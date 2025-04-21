"""
@author: Fahim Ahmed Zaman
"""

import numpy as np
import os
from tqdm import tqdm
from utilities.misc import addNoise, mean_variance_normalization, plot_seg, plot_sampling
from utilities.misc import noiseScheduler, plot_noise_parameters
import tensorflow as tf

#%% Evaluation Metrics
def dice_coef_score(y_true, y_pred, num_labels=3, epsilon=1e-6):
    y_true_f = tf.cast(tf.one_hot(tf.cast(y_true, dtype=tf.uint8), num_labels)[..., 1:], dtype=tf.float32)
    y_pred_f = tf.cast(tf.one_hot(tf.cast(y_pred, dtype=tf.uint8), num_labels)[..., 1:], dtype=tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon)
    return dice.numpy()

def iou_score(y_true, y_pred, num_labels=3, epsilon=1e-6):
    y_true_f = tf.cast(tf.one_hot(tf.cast(y_true, dtype=tf.uint8), num_labels)[..., 1:], dtype=tf.float32)
    y_pred_f = tf.cast(tf.one_hot(tf.cast(y_pred, dtype=tf.uint8), num_labels)[..., 1:], dtype=tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    iou = intersection / (union + epsilon)
    return iou.numpy()

def iou_wo_background(y_true, y_pred, num_classes=3, epsilon=1e-6):
    if isinstance(y_pred, (tuple, list)):
        y_pred = y_pred[0]
    y_pred = tf.convert_to_tensor(y_pred)
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
        iou = (intersection + epsilon) / (union + epsilon)
        ious.append(iou)
    return tf.reduce_mean(ious).numpy()

#%% Main segmentation function
def segment(data,
            labelDecoder,
            imageEncoder,
            denoiser,
            true_labels=None,
            model_directory='./savedModels',
            totalTimesteps=1000,
            samplingSteps=10,
            sampler='DDIM',
            scheduler='cosine',
            sigma=0,
            batch_size=1,
            plots=False):

    alpha_cumprod, alphas, betas, times = noiseScheduler(T_all=totalTimesteps, total_timesteps=samplingSteps)
    alpha_cumprod1 = np.roll(alpha_cumprod, 1)

    models = (labelDecoder, imageEncoder, denoiser)
    modelNames = ['labelDecoder.hdf5', 'imageEncoder.hdf5', 'denoiser.hdf5']
    savedmodels = os.listdir(model_directory)
    if all(item in savedmodels for item in modelNames):
        for modl, path in zip(models, modelNames):
            modl.load_weights(os.path.join(model_directory, path))
            print(f'\n? {os.path.splitext(path)[0]} weights loaded.')
    else:
        raise Exception('? Model weights are unavailable. Please train LDSeg...')

    if sigma != 0:
        data = addNoise(data, sigma)

    z_i = imageEncoder.predict(data, batch_size=batch_size, verbose=0)
    z_lt = mean_variance_normalization(np.random.normal(0, 1, z_i.shape))

    for t, beta, acum, acum1 in tqdm(reversed(list(zip(times, betas, alpha_cumprod, alpha_cumprod1))), desc="Sampling timesteps"):
        z_n = denoiser.predict([z_lt, z_i, np.repeat(t, z_lt.shape[0], axis=0)], batch_size=batch_size, verbose=0)
        epsilon = np.random.normal(scale=1, size=z_lt.shape)
        if t == times[0]:
            z_lt = (z_lt - np.sqrt(1 - acum) * z_n) / np.sqrt(acum) + np.sqrt(beta) * epsilon
        else:
            if sampler == 'DDPM':
                s = np.sqrt(((1 - acum1) / (1 - acum)) * (1 - acum / acum1))
            else:
                s = 0
            z_lt = np.sqrt(acum1) * ((z_lt - np.sqrt(1 - acum) * z_n) / np.sqrt(acum)) + \
                   np.sqrt(1 - acum1 - s ** 2) * z_n + s * epsilon
        z_lt = mean_variance_normalization(z_lt)

    yhat = labelDecoder.predict(z_lt, batch_size=batch_size, verbose=0)
    yhat = np.argmax(yhat, axis=-1)

    if true_labels is not None:
        dice = dice_coef_score(true_labels, yhat)
        iou = iou_score(true_labels, yhat)
        iou_wb = iou_wo_background(true_labels, yhat)
        print(f"\n? Test Dice Score    : {dice:.4f}")
        print(f"? Test IoU Score     : {iou:.4f}")
        print(f"? Test IoU w/o BG    : {iou_wb:.4f}")

    return yhat

#%% tf.data.Dataset version

def segment_tfdata(dataset,
                    labelDecoder,
                    imageEncoder,
                    denoiser,
                    model_directory='./savedModels',
                    totalTimesteps=1000,
                    samplingSteps=10,
                    sampler='DDIM',
                    scheduler='cosine',
                    sigma=0,
                    batch_size=1,num_classes=3):

    images = []
    labels = []
    for img, lbl in dataset:
        images.append(img.numpy())
        labels.append(lbl.numpy())
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    if labels.ndim == 4 and labels.shape[-1] == 1:
        labels = np.squeeze(labels, axis=-1)

    return segment(images,
                   labelDecoder=labelDecoder,
                   imageEncoder=imageEncoder,
                   denoiser=denoiser,
                   true_labels=labels,
                   model_directory=model_directory,
                   totalTimesteps=totalTimesteps,
                   samplingSteps=samplingSteps,
                   sampler=sampler,
                   scheduler=scheduler,
                   sigma=sigma,
                   batch_size=batch_size,
                   plots=False)
