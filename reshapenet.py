import torch
import torch.nn as nn
import numpy as np

# Define a simple model with only the convolutional layer
class reshapenet(nn.Module):
    def __init__(self,in_channels=1):
        super(reshapenet, self).__init__()
        # Only the convolutional layer
        self.conv_layer= nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)

    def forward(self, x):
        # Pass through the convolutional layer
        x = self.conv_layer(x)
        return x