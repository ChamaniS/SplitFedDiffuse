"""
@author: Fahim Zaman

email: fahim-zaman@uiowa.edu
"""

from utilities import dataLoader, model, trainer, sampler, misc
from configparser import ConfigParser
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
#%% Load Configurations
config = ConfigParser()
config.read('cfg.ini')

GPU = config.getboolean('GPU', 'Multiple')
misc.setGPU(GPU)

MINBETA = config.getfloat('NoiseScheduler', 'beta_0')
MAXBETA = config.getfloat('NoiseScheduler', 'beta_T')
SCHEDULER = config.get('NoiseScheduler', 'Scheduler')
TOTAL_STEPS = config.getint('NoiseScheduler', 'Timesteps')

MODEL_TRAINING = config.getboolean('Parameters', 'Training')
CHECKPOINT = config.getboolean('Parameters', 'Checkpoint')
BATCH_SIZE = config.getint('Parameters', 'BatchSize')
EPOCH = config.getint('Parameters', 'Epoch')
LEARNING_RATE = config.getfloat('Parameters', 'LearningRate')

SAMPLING_STEPS = config.getint('Sampler', 'Sampling_Steps')
SAMPLER = config.get('Sampler', 'Sampler')
SIGMA = config.getfloat('Sampler', 'Sigma')

#%% Data Read
#images, labels = dataLoader.dataRead()
images, labels = dataLoader.dataRead(dataset='HAM10K')
images = np.array(images)
labels = np.array(labels)

test_images = images[-1000:]
test_labels = labels[-1000:]

train_val_images = images[:-1000]
train_val_labels = labels[:-1000]

train_images, val_images, train_labels, val_labels = train_test_split(train_val_images, train_val_labels, test_size=0.15, random_state=42)

#%% Load Model Components
components = model.loadModel(images,
                             labels,
                             loadCheckpoint = CHECKPOINT)

labelEncoder, labelDecoder, imageEncoder, denoiser = components

#%% Model Training
if MODEL_TRAINING == True:
    trainer.train_models(train_images = train_images,
                         train_labels = train_labels,
                         val_images = val_images,
                         val_labels = val_labels,
                         labelEncoder = labelEncoder,
                         labelDecoder = labelDecoder,
                         imageEncoder = imageEncoder,
                         denoiser = denoiser,
                         minbeta = MINBETA,
                         maxbeta = MAXBETA,
                         total_timesteps = TOTAL_STEPS,
                         scheduler = SCHEDULER,
                         epochs = EPOCH,
                         batch_size = BATCH_SIZE,
                         lrate = LEARNING_RATE)

    import gc
    import tensorflow as tf
    gc.collect()
    tf.keras.backend.clear_session()

    del labelEncoder, labelDecoder, imageEncoder, denoiser
    gc.collect()
    tf.keras.backend.clear_session()


    # ? Reload model components (weights already saved)
    components = model.loadModel(images, labels, loadCheckpoint=True)
    labelEncoder, labelDecoder, imageEncoder, denoiser = components

#%% Model inference
'''
yhat = sampler.segment(data = images,
                       labelDecoder = labelDecoder, 
                       imageEncoder = imageEncoder, 
                       denoiser = denoiser, 
                       totalTimesteps = TOTAL_STEPS, 
                       samplingSteps = SAMPLING_STEPS, 
                       sampler = SAMPLER,
                       scheduler = SCHEDULER,
                       sigma = SIGMA, 
                       batch_size = BATCH_SIZE)                                       
'''
# Inference on test set
yhat = sampler.segment(data=test_images,
                       labelDecoder=labelDecoder,
                       imageEncoder=imageEncoder,
                       denoiser=denoiser,
                       true_labels=test_labels,
                       totalTimesteps=TOTAL_STEPS,
                       samplingSteps=SAMPLING_STEPS,
                       sampler=SAMPLER,
                       scheduler=SCHEDULER,
                       sigma=SIGMA,
                       batch_size=BATCH_SIZE)

#%% Write Segmentations
dataLoader.segWrite(segmentation = yhat)