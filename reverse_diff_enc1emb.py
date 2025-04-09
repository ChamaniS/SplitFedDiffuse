import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    """Encodes time steps into embeddings using sinusoidal functions."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # Ensure t is a tensor and has the correct device
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.unsqueeze(0)

        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * 
                        (-torch.log(torch.tensor(10000.0, dtype=torch.float32, device=t.device)) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class UNetBlock(nn.Module):
    """Basic UNet Block with Conv + GN + SiLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()  # Swish activation

    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        return x


class ReverseDiffusion(nn.Module):
    """U-Net-based Reverse Diffusion Model."""
    def __init__(self, input_dim, hidden_dim, output_dim, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.encoder1 = UNetBlock(input_dim, hidden_dim)
        self.encoder2 = UNetBlock(hidden_dim, hidden_dim * 2)
        self.bottleneck = UNetBlock(hidden_dim * 2, hidden_dim * 2)
        self.decoder1 = UNetBlock(hidden_dim * 2, hidden_dim)
        self.decoder2 = UNetBlock(hidden_dim, output_dim)

        self.final_conv = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x_t, t):
        # Ensure t is a tensor with the correct shape and device
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.float32, device=x_t.device)
        elif isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.unsqueeze(0)

        t_embedding = self.time_mlp(t).view(t.shape[0], -1, 1, 1)  # Expand for spatial use
        
        enc1 = self.encoder1(x_t) + t_embedding
        enc2 = self.encoder2(enc1)
        bottleneck = self.bottleneck(enc2)
        dec1 = self.decoder1(bottleneck) + enc1  # Skip connection
        dec2 = self.decoder2(dec1)
        
        noise_pred = self.final_conv(dec2)  
        return noise_pred


def denoise(x_t, denoiser, alpha_cum_t, t, device):
    x_t = x_t.to(device)

    # Ensure t is a tensor with correct type and device
    if isinstance(t, int):
        t = torch.tensor([t], dtype=torch.float32, device=device)
    elif isinstance(t, torch.Tensor) and t.dim() == 0:
        t = t.unsqueeze(0)

    t = t.long()  # Fix: Convert t to long for indexing

    noise_pred = denoiser(x_t, t)  # Predict the noise from the noisy input
    sqrt_alpha_cum_t = torch.sqrt(alpha_cum_t[t]).to(device)  # Index correctly
    sqrt_one_minus_alpha_cum_t = torch.sqrt(1 - alpha_cum_t[t]).to(device)

    # Calculate the denoised output by subtracting the predicted noise
    x_0_hat = (x_t - sqrt_one_minus_alpha_cum_t * noise_pred) / sqrt_alpha_cum_t
    return x_0_hat



def initialize_denoiser(input_dim, output_dim, hidden_dim, device="cuda"):
    denoiser = ReverseDiffusion(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    return denoiser
