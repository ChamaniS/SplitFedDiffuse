import os
import cv2
import tensorflow as tf
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted
from skimage.measure import label
import imageio


IMG_SIZE = (512, 512)
AUTOTUNE = tf.data.AUTOTUNE

def process_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

'''
def process_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    mask = tf.where(mask > 0, 1, 0)
    mask = tf.cast(mask, tf.uint8)
    return mask

def process_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    mask = tf.cast(mask, tf.uint8)
    return mask
'''
def process_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    
    # Assign classes based on intensity (example: 0=background, 1=class1, 2=class2)
    mask = tf.where(mask == 255, tf.constant(2, dtype=mask.dtype), mask)
    mask = tf.where(mask == 128, tf.constant(1, dtype=mask.dtype), mask)
    
    return tf.cast(mask, tf.uint8)


def process_pair(img_path, mask_path):
    img = process_image(img_path)
    mask = process_mask(mask_path)
    return img, mask

def create_dataset(image_dir, mask_dir, batch_size=32, shuffle=True):
    image_paths = natsorted([str(p) for p in Path(image_dir).glob("*")])
    mask_paths = natsorted([str(p) for p in Path(mask_dir).glob("*")])
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.map(process_pair, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

def writeImage(segs, imgpath, segpath):
    for n in tqdm(range(len(segs))):
        img = imageio.imread(imgpath[n])

        if img.ndim == 2:
            H, W = img.shape
        elif img.ndim == 3:
            H, W, _ = img.shape
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        lbl = cv2.resize(segs[n], (W, H), interpolation=cv2.INTER_NEAREST)
        lbl = np.where(lbl > 0, 1, 0)
        lbl = label(lbl, connectivity=2).astype(np.uint8)

        imageio.imwrite(segpath[n], lbl)

def segWrite(segmentation, img_dir, seg_dir):
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    tag = '_segmentation.'
    imgfilenames = natsorted(os.listdir(img_dir))
    imgpath = [os.path.join(img_dir, i) for i in imgfilenames]
    segfilenames = [i.split(os.extsep, 1)[0] + tag + i.split(os.extsep, 1)[1] for i in imgfilenames]
    segpath = [os.path.join(seg_dir, i) for i in segfilenames]

    writeImage(segmentation, imgpath, segpath)
