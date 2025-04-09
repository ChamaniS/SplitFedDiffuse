import torch
import torchvision
from dataset import CVCINDataset
from torch.utils.data import DataLoader
device = "cuda"
import random
import time
import torch.nn as nn
import numpy as np

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory=False

):
    train_ds = CVCINDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_ds = CVCINDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader


def get_loaders_test(
        test_dir,
        test_maskdir,
        test_transform
):
    test_ds = CVCINDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_ds,
    )
    return test_loader