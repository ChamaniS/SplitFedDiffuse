"""
@author: Fahim Ahmed Zaman
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import math
from utilities.misc import perturb_flip

from tensorflow.keras import Model
from utilities.gaussianBlock import GaussianDiffusion
#from utilities.model import Denoiser

#%%
''' 
This module is used for initializing the model components

'''

#%% Kernel initialization
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )

#%% Up and Down-sampling
def DownSample(width):
    def apply(x):
        x = layers.Conv2D(width, kernel_size=3, strides=2, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply

#%% Convolutional blocks
def res_conv_block(x, 
                   filter_size, 
                   size, 
                   dropout, 
                   batch_norm=False, 
                   activation='relu', 
                   layer_name=False):
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same', kernel_initializer='he_uniform')(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=-1)(conv)
    conv = keras.activations.swish(conv) if activation == 'swish' else layers.Activation('relu')(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same', kernel_initializer='he_uniform')(conv)
    if batch_norm:
        conv = layers.BatchNormalization(axis=-1)(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm:
        shortcut = layers.BatchNormalization(axis=-1)(shortcut)

    res_path = layers.add([shortcut, conv])
    if activation == 'swish':
        res_path = keras.activations.swish(res_path) if not layer_name else layers.Activation(keras.activations.swish, name=layer_name)(res_path)
    else:
        res_path = layers.Activation('relu')(res_path) if not layer_name else layers.Activation('relu', name=layer_name)(res_path)
    return res_path

def conv_block(x, 
               filter_size, 
               kernel_size, 
               activation_fn, 
               groups=4, 
               dropout_rate=True):
    
    residual = layers.Conv2D(filter_size, kernel_size=1, padding='same', kernel_initializer=kernel_init(1.0))(x)
    x = activation_fn(x)
    x = layers.Conv2D(filter_size, kernel_size=kernel_size, padding="same", kernel_initializer=kernel_init(1.0))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = activation_fn(x)
    x = layers.Conv2D(filter_size, kernel_size=kernel_size, padding="same", kernel_initializer=kernel_init(0.0))(x)
    x = layers.Add()([x, residual])
    x = activation_fn(x)
    x = DownSample(filter_size)(x)
    x = layers.GroupNormalization(groups=groups)(x)
    return x

def ResidualBlock(width, 
                  groups=8, 
                  activation_fn=keras.activations.swish):
    
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[-1]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1, kernel_initializer=kernel_init(1.0))(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[:, None, None, :]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)

        x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0))(x)
        x = layers.Add()([x, residual])
        return x

    return apply

#%% Attention mechanisms
class MultiHeadAttentionBlock(layers.Layer):
    """Applies multi-head self-attention.

    Args:
        units: Number of units in the dense layers.
        num_heads: Number of attention heads.
        groups: Number of groups for GroupNormalization layer.
    """
    def __init__(self, units, num_heads=8, groups=8, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.groups = groups

        # Define the GroupNormalization and Dense layers for query, key, value, and projection
        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def split_heads(self, x, batch_size):
        """Splits the last dimension into (num_heads, depth) and transposes the result
        to shape (batch_size, num_heads, height, width, depth).
        """
        depth = self.units // self.num_heads
        x = tf.reshape(x, (batch_size, -1, self.num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, tokens, depth)

    def call(self, inputs):
        shape = tf.shape(inputs)
        batch_size, height, width, _ = shape[0], shape[1], shape[2], shape[3]
        num_tokens = height * width
        scale = tf.cast(self.units // self.num_heads, tf.float32) ** (-0.5)
    
        # Normalize inputs
        inputs_norm = self.norm(inputs)
    
        # Compute Q, K, V
        q = self.query(inputs_norm)  # (B, H, W, units)
        k = self.key(inputs_norm)    # (B, H, W, units)
        v = self.value(inputs_norm)  # (B, H, W, units)
    
        # Reshape and split into heads
        q = tf.reshape(q, (batch_size, num_tokens, self.units))  # (B, H*W, units)
        k = tf.reshape(k, (batch_size, num_tokens, self.units))
        v = tf.reshape(v, (batch_size, num_tokens, self.units))
    
        q = self.split_heads(q, batch_size)  # (B, num_heads, H*W, depth)
        k = self.split_heads(k, batch_size)  # (B, num_heads, H*W, depth)
        v = self.split_heads(v, batch_size)  # (B, num_heads, H*W, depth)
    
        # Compute attention scores
        attn_score = tf.einsum("bhid,bhjd->bhij", q, k) * scale  # (B, num_heads, H*W, H*W)
        attn_score = tf.nn.softmax(attn_score, axis=-1)  # Softmax over the last dimension (H*W)
    
        # Apply attention to values
        attn_output = tf.einsum("bhij,bhjd->bhid", attn_score, v)  # (B, num_heads, H*W, depth)
    
        # Concatenate heads and project back to the original dimension
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])  # (B, H*W, num_heads, depth)
        attn_output = tf.reshape(attn_output, (batch_size, height, width, self.units))  # (B, H, W, units)
    
        # Final projection layer
        proj_output = self.proj(attn_output)
    
        # Ensure residual connection has matching shape
        if inputs.shape[-1] == proj_output.shape[-1]:
            return inputs + proj_output
        else:
            raise ValueError("Shape mismatch between input and projected output for residual connection")
            
class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        
        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])
        
        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])
        
        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj

#%% Time embeddings
class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb
    return apply

#%% Model components
'''
def encoder(input_shape,
            elayers = [1, 2, 4, 4, 2],
            dropout_rate = 0.2, 
            batch_norm = True, 
            FILTER_SIZE = 3, 
            FILTER_NUM = 16, 
            activation = 'swish'):
                  
    inputs = layers.Input(input_shape, dtype=tf.float32, name='encoder_input')
    
    for i in range(len(elayers)):
        if i==0:
            x = res_conv_block(inputs, 
                               FILTER_SIZE, 
                               elayers[i]*FILTER_NUM, 
                               dropout_rate, 
                               batch_norm, 
                               activation=activation)
        else:
            x = res_conv_block(x, 
                               FILTER_SIZE, 
                               elayers[i]*FILTER_NUM, 
                               dropout_rate, 
                               batch_norm, 
                               activation=activation)
        if i!=len(elayers)-1:
            x = layers.MaxPooling2D(pool_size=(2, 2)) (x)
                
    x = layers.Conv2D(1, kernel_size=(1, 1), padding='same') (x)
    encoded = layers.LayerNormalization(axis=(1, 2, 3), center=True, scale=True, name='encoder_output')(x)
    model = models.Model(inputs, encoded, name="Label-Encoder")
    return model
'''
def encoder(input_shape,
            elayers=[1, 2, 4, 4, 2],
            dropout_rate=0.2,
            batch_norm=True,
            FILTER_SIZE=3,
            FILTER_NUM=16,
            activation='swish'):

    inputs = layers.Input(input_shape, dtype=tf.float32, name='encoder_input')
    x = inputs

    for i in range(len(elayers)):
        x = res_conv_block(
            x,
            filter_size=FILTER_SIZE,
            size=elayers[i] * FILTER_NUM,
            dropout=dropout_rate,
            batch_norm=batch_norm,
            activation=activation,
        )

        # ?? NaN Check after each conv block
        x = tf.debugging.check_numerics(x, message=f"NaNs after res_conv_block layer {i}")

        if i != len(elayers) - 1:
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(1, kernel_size=(1, 1), padding='same')(x)
    x = layers.LayerNormalization(axis=(1, 2, 3), center=True, scale=True)(x)

    # ?? Final NaN check before output
    x = tf.debugging.check_numerics(x, message="NaNs in final encoder output")

    encoded = layers.Activation('linear', name='encoder_output')(x)
    model = models.Model(inputs, encoded, name="Label-Encoder")
    return model





def decoder(input_shape,
            dlayers=[2, 4, 4, 2],
            dropout_rate=0.2, 
            batch_norm=True, 
            FILTER_SIZE=3, 
            FILTER_NUM=16, 
            activation='swish',
            num_classes=3):

    inputs = layers.Input(input_shape, dtype=tf.float32, name='decoder_input')
    x = inputs

    for i in range(len(dlayers)):
        # Ensure channel alignment for the first layer
        expected_channels = dlayers[i] * FILTER_NUM
        if i == 0 and input_shape[-1] != expected_channels:
            x = layers.Conv2D(expected_channels, kernel_size=1, padding="same")(x)

        x = layers.Conv2DTranspose(filters=expected_channels, 
                                   kernel_size=(3, 3), 
                                   padding="same", 
                                   strides=2)(x)
        x = res_conv_block(x, 
                           FILTER_SIZE, 
                           expected_channels, 
                           dropout_rate, 
                           batch_norm, 
                           activation=activation)

    x = keras.activations.swish(x) if activation == 'swish' else layers.Activation('relu')(x)
    x = layers.Conv2D(num_classes, kernel_size=(1, 1))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    decoded = layers.Softmax(axis=-1, name='decoder_output')(x)
    model = models.Model(inputs, decoded, name="Label-Decoder")
    return model


def ImgEncoder(input_shape,
               filter_size=16,
               kernel_size=3, 
               dropout=0.2, 
               groups=4,
               channels = 1,
               activation=keras.activations.swish):  
    
    inputs = layers.Input(input_shape, dtype=tf.float32, name='image_input')
    
    x = layers.Conv2D(filter_size, kernel_size=3, padding='same', kernel_initializer=kernel_init(1.0))(inputs)
    x = conv_block(x, 2*filter_size, kernel_size, dropout_rate=dropout, activation_fn=activation)
    x = conv_block(x, 4*filter_size, kernel_size, dropout_rate=dropout, activation_fn=activation)
    x = conv_block(x, 4*filter_size, kernel_size, dropout_rate=dropout, activation_fn=activation)
    x = MultiHeadAttentionBlock(4*filter_size)(x)
    x = conv_block(x, 2*filter_size, kernel_size, dropout_rate=dropout, activation_fn=activation)
    x = MultiHeadAttentionBlock(2*filter_size)(x)
    x = layers.Conv2D(channels, kernel_size=1, padding="same", kernel_initializer=kernel_init(0.0))(x)
    x = MultiHeadAttentionBlock(channels, num_heads=channels, groups=1)(x)
    x = activation(x)
    outputs = layers.BatchNormalization(axis=-1, name='embedding_output')(x)
    
    # Model 
    model = models.Model(inputs, outputs, name="Image-Encoder")
    return model

def Denoiser(input_shape_lv,
             input_shape_ie,
             first_conv_channels = 16,
             widths = [16, 32, 64],
             has_attention = [False, True, True],
             num_res_blocks=2,
             norm_groups=4,
             channels=1,
             interpolation="nearest",
             activation_fn=keras.activations.swish, num_classes=3):
    
    lv_input = layers.Input(input_shape_lv, dtype=tf.float32)
    img_input = layers.Input(input_shape_ie, dtype=tf.float32)
    time_input = layers.Input((), dtype=tf.float32)
    
    inputs = [lv_input, img_input, time_input]
    
    x = layers.Concatenate(axis=-1, name='denoiser_input')([lv_input, img_input])
    
    x = layers.Conv2D(first_conv_channels, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_init(1.0))(x)
    
    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    
    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(widths[i], 
                              groups=norm_groups, 
                              activation_fn=activation_fn)([x, temb])
            
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)
        

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    
    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(widths[i], 
                              groups=norm_groups, 
                              activation_fn=activation_fn)([x, temb])
            
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    outputs = layers.Conv2D(channels, (3, 3), padding="same", name='denoiser_output')(x)
    return keras.Model(inputs, outputs, name="Denoiser")
#%% load model components

def loadModel(images, labels, filepath='./savedModels', loadCheckpoint=True, num_classes=3):
    tf.keras.backend.clear_session()
    labelEncoder = encoder(labels.shape[1:])
    denoised_shape = (labelEncoder.output.shape[1], labelEncoder.output.shape[2], num_classes)
    labelDecoder = decoder(denoised_shape, num_classes=num_classes)
    imageEncoder = ImgEncoder(images.shape[1:])
    denoiser = Denoiser(labelEncoder.output.shape[1:], imageEncoder.output.shape[1:], channels=num_classes)
    
    # model components
    models = (labelEncoder, labelDecoder, imageEncoder, denoiser)
    
    # load checkpoints
    if loadCheckpoint == True:
        modelNames = ['labelEncoder.hdf5', 'labelDecoder.hdf5', 'imageEncoder.hdf5', 'denoiser.hdf5']
        savedmodels = os.listdir(filepath)
        if all(item in savedmodels for item in modelNames):
            for modl, path in zip(models, modelNames):
                modl.load_weights(os.path.join(filepath, path))
                print(f'\n{os.path.splitext(path)[0]} weights loaded...')
        else:
            print('\nModel weights are unavailable. Please train LDSeg...\n')
    return models
    


class LDSeg(Model):
    def __init__(self, network, ema_network, imgEncoder, encoder, decoder, 
                 timesteps, gdf_util, filepath, augmentation=False, ema=0.999, lamda=1):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.imgEncoder = imgEncoder
        self.encoder = encoder
        self.decoder = decoder
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.filepath = filepath
        self.aug = augmentation
        self.ema = ema
        self.lamda = lamda

        self.encoderPath = os.path.join(filepath, 'labelEncoder.hdf5')
        self.decoderPath = os.path.join(filepath, 'labelDecoder.hdf5')
        self.imgEncoderPath = os.path.join(filepath, 'imageEncoder.hdf5')
        self.denoiserPath = os.path.join(filepath, 'denoiser.hdf5')
        self.modelPaths = [self.encoderPath, self.decoderPath, self.imgEncoderPath, self.denoiserPath]

        print("\nModel save paths:")
        for path in self.modelPaths:
            print(f"  - {path}")
        print()

    def compile(self, optimizerMaskEC, optimizerMaskDC, optimizerImgEC, optimizerDen, 
                loss1, loss2, score, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.optimizerMaskEC = optimizerMaskEC
        self.optimizerMaskDC = optimizerMaskDC
        self.optimizerImgEC = optimizerImgEC
        self.optimizerDen = optimizerDen
        self.loss1 = loss1
        self.loss2 = loss2
        self.score = score

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training=training)
        x = self.decoder(x, training=training)
        return x

    def train_step(self, data):
        X, y = data
        bsize = tf.shape(X)[0]
        Bx, Hx, Wx, Cx = X.get_shape()
    
        if self.aug:
            flips = tf.random.uniform(minval=0, maxval=4, shape=(bsize, 1), dtype=tf.int64)
            X = perturb_flip(X, flips)
            y = perturb_flip(y, flips)
    
        X = tf.ensure_shape(X, [None, Hx, Wx, Cx])
        y = tf.ensure_shape(y, [None, Hx, Wx, 1])
        y = tf.squeeze(y, axis=-1)
    
        y = tf.clip_by_value(y, 0, self.decoder.output_shape[-1] - 1)
        tf.debugging.check_numerics(tf.cast(y, tf.float32), "NaNs in y before encoder")

        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(bsize, 1), dtype=tf.int64)
    
        with tf.GradientTape(persistent=True) as tape:
            y_prime = tf.expand_dims(y, axis=-1)
            z_l0 = self.encoder(y_prime, training=True)
            
            z_l0 = tf.clip_by_value(z_l0, -10.0, 10.0)
            tf.debugging.check_numerics(z_l0, "z_l0 contains NaNs after encoder")
            
            
            z_i = self.imgEncoder(X, training=True)
    
            epsilon = tf.random.normal(shape=tf.shape(z_l0))
            z_lt = self.gdf_util.q_sample(z_l0, t, epsilon)
            tf.debugging.check_numerics(z_lt, "NaNs in z_lt before denoising")
            
            z_nt = self.network([z_lt, z_i, t], training=True)
            z_dn = z_lt - z_nt
            
            # ?? Normalize z_dn before feeding to decoder to avoid NaNs in softmax
            z_dn = (z_dn - tf.reduce_mean(z_dn)) / (tf.math.reduce_std(z_dn) + 1e-6)
            # Or alternatively clip extreme values:
            # z_dn = tf.clip_by_value(z_dn, -10.0, 10.0)
            
            tf.debugging.check_numerics(z_dn, "NaNs in decoder input z_dn")
            
            y_hat = self.decoder(z_dn, training=True)
            tf.debugging.check_numerics(y_hat, "NaNs in decoder output y_hat")

    
            # ?? Final check before loss
            tf.debugging.check_numerics(tf.reduce_sum(y_hat), message="NaNs in y_hat!")
    
            loss1 = self.loss1(y, y_hat)
            loss2 = self.loss2(epsilon, z_nt)
            loss = loss1 + self.lamda * loss2
            score = self.score(y, y_hat)
    
        self.optimizerDen.apply_gradients(zip(tape.gradient(loss, self.network.trainable_weights), self.network.trainable_weights))
        self.optimizerMaskEC.apply_gradients(zip(tape.gradient(loss, self.encoder.trainable_weights), self.encoder.trainable_weights))
        self.optimizerMaskDC.apply_gradients(zip(tape.gradient(loss, self.decoder.trainable_weights), self.decoder.trainable_weights))
        self.optimizerImgEC.apply_gradients(zip(tape.gradient(loss, self.imgEncoder.trainable_weights), self.imgEncoder.trainable_weights))
    
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
    
        del tape
    
        return {
            "Loss(L)": loss,
            "Loss(L1)": loss1,
            "Loss(L2)": loss2,
            "IoU_nb": score
        }



    def test_step(self, data):
        X, y = data
        bsize = tf.shape(X)[0]
        Bx, Hx, Wx, Cx = X.get_shape()

        X = tf.ensure_shape(X, [None, Hx, Wx, Cx])
        y = tf.ensure_shape(y, [None, Hx, Wx, 1])
        y = tf.squeeze(y, axis=-1)
        y = tf.cast(y, tf.int32)
        y = tf.clip_by_value(y, 0, self.decoder.output_shape[-1] - 1)
        #tf.debugging.assert_less_equal(tf.reduce_max(y), self.decoder.output_shape[-1] - 1, message="Label out of range")
        tf.debugging.assert_less_equal(
            tf.cast(tf.reduce_max(y), tf.int32),
            tf.cast(self.decoder.output_shape[-1] - 1, tf.int32),
            message="Label out of range"
        )

        
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(bsize, 1), dtype=tf.int64)

        y_prime = tf.expand_dims(y, axis=-1)
        z_l0 = self.encoder(y_prime, training=False)
        z_i = self.imgEncoder(X, training=False)

        epsilon = tf.random.normal(shape=tf.shape(z_l0))
        z_lt = self.gdf_util.q_sample(z_l0, t, epsilon)
        z_nt = self.ema_network([z_lt, z_i, t], training=False)
        z_dn = z_lt - z_nt
        y_hat = self.decoder(z_dn, training=False)

        loss1 = self.loss1(y, y_hat)
        loss2 = self.loss2(epsilon, z_nt)
        loss = loss1 + self.lamda * loss2
        score = self.score(y, y_hat)

        return {
            "Val_Loss(L)": loss,
            "Val_Loss(L1)": loss1,
            "Val_Loss(L2)": loss2,
            "Val_IoU_nb": score
        }

    def save_model(self, epoch, logs=None):
        if epoch % 100 == 0:
            self.encoder.save_weights(self.modelPaths[0])
            self.decoder.save_weights(self.modelPaths[1])
            self.imgEncoder.save_weights(self.modelPaths[2])
            self.ema_network.save_weights(self.modelPaths[3])
            print("\n? Model weights saved.\n")


    def save_model(self, epoch, logs=None):
        if epoch % 100 == 0:
            self.encoder.save_weights(self.encoderPath)
            self.decoder.save_weights(self.decoderPath)
            self.imgEncoder.save_weights(self.imgEncoderPath)
            self.ema_network.save_weights(self.denoiserPath)
            print("\n?? Model weights saved.")
    