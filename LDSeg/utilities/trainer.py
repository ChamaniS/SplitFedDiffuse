"""
@author: Fahim Ahmed Zaman
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from utilities.gaussianBlock import GaussianDiffusion
from utilities.model import Denoiser, LDSeg

#%% Data augmentation

def data_flip(imbatch, rots):
    n = rots.shape[0]
    imaugs = []
    for i in range(n):
        if rots[i] == 0:
            imaugs.append(imbatch[i])
        elif rots[i] == 3:
            imaugs.append(np.flip(imbatch[i], axis=(0, 1)))
        else:
            imaugs.append(np.flip(imbatch[i], axis=rots[i] - 1))
    return np.array(imaugs, dtype=np.float32)

def perturb_flip(img, rot):
    return tf.py_function(data_flip, [img, rot], tf.float32)

#%% Losses

def dice_coef_score(y_true, y_pred, num_labels=3, epsilon=1e-6):
    y_true_f = tf.cast(tf.one_hot(tf.cast(y_true, dtype=tf.uint8), num_labels)[..., 1:], tf.float32)
    y_pred_f = tf.cast(y_pred[..., 1:], tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon)

def dice_coef(y_true, y_pred, num_labels=3, epsilon=1e-6):
    y_true_f = tf.cast(tf.one_hot(tf.cast(y_true, dtype=tf.uint8), num_labels), tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    return (2. * tf.reduce_sum(y_true_f * y_pred_f)) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon)

def CE_loss(y_true, y_pred):
    return keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true, y_pred)


def DSC_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def reconstructionLoss(y_true, y_pred, gamma=2):
    return CE_loss(y_true, y_pred) + gamma * DSC_loss(y_true, y_pred)


#%% IoU without background

def iou_wo_background(y_true, y_pred, num_classes=3, epsilon=1e-6):
    if y_pred.shape.ndims == 4:
        y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    ious = []
    for class_id in range(1, num_classes):
        true_mask = tf.equal(y_true, class_id)
        pred_mask = tf.equal(y_pred, class_id)
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(true_mask, pred_mask), tf.float32))
        ious.append((intersection + epsilon) / (union + epsilon))
    return tf.reduce_mean(ious)

#%% LDSeg Trainer

def train_models(train_dataset, val_dataset, test_dataset,
                 labelEncoder, labelDecoder, imageEncoder, denoiser,
                 epochs=1500, batch_size=4, lrate=1e-3,
                 minbeta=1e-4, maxbeta=0.02, total_timesteps=1000, scheduler='cosine',
                 modelDirectory='./savedModels', num_classes=3):

    tf.keras.backend.clear_session()

    #ema = Denoiser(labelEncoder.output.shape[1:], imageEncoder.output.shape[1:])
    #denoiser.set_weights(ema.get_weights())
    ema = Denoiser(labelEncoder.output.shape[1:], imageEncoder.output.shape[1:], channels=num_classes)
    denoiser = Denoiser(labelEncoder.output.shape[1:], imageEncoder.output.shape[1:], channels=num_classes)
    denoiser.set_weights(ema.get_weights())

    gdf_util = GaussianDiffusion(beta_start=minbeta,
                                 beta_end=maxbeta,
                                 timesteps=total_timesteps,
                                 schedule=scheduler)

    model = LDSeg(network=ema,
                  ema_network=denoiser,
                  imgEncoder=imageEncoder,
                  encoder=labelEncoder,
                  decoder=labelDecoder,
                  timesteps=total_timesteps,
                  gdf_util=gdf_util,
                  augmentation=True,
                  filepath=modelDirectory)

    model.compile(optimizerMaskEC=Adam(learning_rate=lrate),
                  optimizerMaskDC=Adam(learning_rate=lrate),
                  optimizerImgEC=Adam(learning_rate=lrate),
                  optimizerDen=Adam(learning_rate=lrate),
                  loss1=reconstructionLoss,
                  loss2=keras.losses.MeanSquaredError(),
                  score=lambda y_true, y_pred: iou_wo_background(y_true, y_pred, num_classes=num_classes))

    model.fit(train_dataset,
              validation_data=val_dataset,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=model.save_model)])
