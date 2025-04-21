from utilities import dataLoader, model, trainer, sampler, misc
from configparser import ConfigParser
import tensorflow as tf
import gc

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

#%% Data Read using tf.data
train_dataset = dataLoader.create_dataset(
    image_dir='/local-scratch/localhome/csj5/HAM_Data/HAM_NEW/train_img',
    mask_dir='/local-scratch/localhome/csj5/HAM_Data/HAM_NEW/train_mask',
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_dataset = dataLoader.create_dataset(
    image_dir='/local-scratch/localhome/csj5/HAM_Data/HAM_NEW/val_img',
    mask_dir='/local-scratch/localhome/csj5/HAM_Data/HAM_NEW/val_mask',
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = dataLoader.create_dataset(
    image_dir='/local-scratch/localhome/csj5/HAM_Data/HAM_NEW/test_img',
    mask_dir='/local-scratch/localhome/csj5/HAM_Data/HAM_NEW/test_mask',
    batch_size=BATCH_SIZE,
    shuffle=False
)

#%% Load Model Components (using a few samples to initialize)
example_batch = next(iter(train_dataset))
example_images, example_labels = example_batch
components = model.loadModel(example_images.numpy(),
                             example_labels.numpy(),
                             loadCheckpoint=CHECKPOINT)
labelEncoder, labelDecoder, imageEncoder, denoiser = components

#%% Model Training
if MODEL_TRAINING:
    trainer.train_models(train_dataset=train_dataset,
                         val_dataset=val_dataset,
                         test_dataset=test_dataset,
                         labelEncoder=labelEncoder,
                         labelDecoder=labelDecoder,
                         imageEncoder=imageEncoder,
                         denoiser=denoiser,
                         minbeta=MINBETA,
                         maxbeta=MAXBETA,
                         total_timesteps=TOTAL_STEPS,
                         scheduler=SCHEDULER,
                         epochs=EPOCH,
                         batch_size=BATCH_SIZE,
                         lrate=LEARNING_RATE)

    gc.collect()
    tf.keras.backend.clear_session()

    del labelEncoder, labelDecoder, imageEncoder, denoiser
    gc.collect()
    tf.keras.backend.clear_session()

    # Reload model components
    #components = model.loadModel(example_images.numpy(), example_labels.numpy(), loadCheckpoint=True)
    example_batch = next(iter(train_dataset))
    example_images, example_masks = example_batch
    components = model.loadModel(example_images.numpy(), example_masks.numpy(), loadCheckpoint=CHECKPOINT)
    
    labelEncoder, labelDecoder, imageEncoder, denoiser = components

#%% Model inference on test set (in batches)
yhat = sampler.segment_tfdata(
    dataset=test_dataset,
    labelDecoder=labelDecoder,
    imageEncoder=imageEncoder,
    denoiser=denoiser,
    model_directory='./savedModels',
    totalTimesteps=TOTAL_STEPS,
    samplingSteps=SAMPLING_STEPS,
    sampler=SAMPLER,
    scheduler=SCHEDULER,
    sigma=SIGMA,
    batch_size=BATCH_SIZE
)


#%% Write Segmentations
dataLoader.segWrite(
    segmentation=yhat,
    img_dir='/local-scratch/localhome/csj5/HAM_Data/HAM_NEW/test_img',
    seg_dir='/local-scratch/localhome/csj5/LDSeg-main/Data/HAM10K/Segmentation/test_img'
)


