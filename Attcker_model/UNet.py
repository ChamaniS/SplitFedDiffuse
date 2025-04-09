import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetAutoencoder(nn.Module):
    def __init__(self, feature_dim, img_channels=3, img_size=256):
        super(UNetAutoencoder, self).__init__()

        self.img_size = img_size
        self.start_size = img_size // 16
        self.start_channels = 512

        self.encoder_fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.start_channels * self.start_size * self.start_size),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # 4x4 ? 8x8
            nn.ReLU()
        )
        self.reduce1 = nn.Conv2d(384, 128, kernel_size=1)  # Expecting 128+256 = 384 channels

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8 ? 16x16
            nn.ReLU()
        )
        self.reduce2 = nn.Conv2d(768, 256, kernel_size=1)  # ?? Fix: Expecting 512+256 = 768 channels

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1),  # 16x16 ? 32x32
            nn.ReLU()
        )
        self.reduce3 = nn.Conv2d(1024, 512, kernel_size=1)  # Expecting 512+512 = 1024 channels

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(512, img_channels, kernel_size=4, stride=2, padding=1),  # 32x32 ? 64x64
            nn.Sigmoid()
        )

    def forward(self, features):
        batch_size = features.shape[0]
        latent = self.encoder_fc(features)
        latent = latent.view(batch_size, 512, self.start_size, self.start_size)
        skip1 = latent
        x = self.conv1(latent)
        skip2 = x
        x = self.conv2(x)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = torch.cat((x, skip2), dim=1)
        x = self.reduce1(x)
        x = self.up2(x)
        x = torch.cat((x, skip1), dim=1)
        x = self.reduce2(x)
        x = self.up3(x)
        latent_upsampled = F.interpolate(latent, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, latent_upsampled), dim=1)
        x = self.reduce3(x)
        x = self.final_layer(x)
        return x

