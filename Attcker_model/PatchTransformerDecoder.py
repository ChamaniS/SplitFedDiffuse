import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchTransformerDecoder(nn.Module):
    def __init__(self,
                 in_channels=64,
                 input_resolution=14,
                 img_size=256,
                 out_channels=3,
                 depth=2,
                 heads=2,
                 mlp_dim=64,
                 dropout_rate=0.2):
        super().__init__()

        self.input_resolution = input_resolution
        self.num_patches = input_resolution * input_resolution
        self.embed_dim = in_channels
        self.token_compressor = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # --- Transformer Token Preparation ---
        self.flatten = nn.Flatten(2)                          # [B, C, H, W] -> [B, C, N]
        self.transpose = lambda x: x.transpose(1, 2)          # [B, C, N] -> [B, N, C]
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # --- Residual CNN Branch ---
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Dropout2d(dropout_rate)
        )

        # --- Unflatten Tokens Back to Spatial ---
        self.unflatten = lambda x: x.transpose(1, 2).reshape(-1, self.embed_dim, self.input_resolution, self.input_resolution)

        # --- Upsampling Decoder Head ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.input_resolution and W == self.input_resolution, \
            f"Expected input resolution ({self.input_resolution}, {self.input_resolution}), got ({H}, {W})"

        # Flatten and add positional embeddings
        x_compressed = self.token_compressor(x)  # [B, C, H, W]
        x_tokens = self.flatten(x_compressed)  # [B, C, H*W]
        # [B, C, H*W]
        x_tokens = self.transpose(x_tokens)     # [B, N, D]
        x_tokens = x_tokens + self.pos_embedding

        # Transformer
        x_transformed = self.transformer(x_tokens)

        # Unflatten tokens to spatial
        x_unflattened = self.unflatten(x_transformed)   # [B, C, H, W]

        # Residual CNN branch
        residual = self.cnn_branch(x_unflattened)
        x = x_unflattened + residual

        # Decode to image
        out = self.decoder(x + x_unflattened)
        return out
