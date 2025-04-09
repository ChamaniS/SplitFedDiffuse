import torch
import torchvision
from dataset import EmbryoDataset
from torch.utils.data import DataLoader

device = "cuda"
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import torch.nn.functional as F
import random
import time
from forward_diff import forward_diffusion
from reverse_diff1 import initialize_denoiser, denoise
import torch.nn as nn
from scipy.linalg import sqrtm

# Define the noise schedule (linear schedule in this case)
T = 1000  # Total number of timesteps
st_noise = 0.0001
end_noise = 0.02
beta = torch.linspace(st_noise, end_noise, T).to(device)  # Linear noise schedule
alpha = 1 - beta
alpha_cum = torch.cumprod(alpha, dim=0)  # Cumulative product of alpha values
alpha_cum_t = alpha_cum.to(device)  # Move to the correct device
t = 0  # current timestep (it determines how much noise has been removed from the data at that point)


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
    train_ds = EmbryoDataset(
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

    val_ds = EmbryoDataset(
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
    test_ds = EmbryoDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_ds,
    )
    return test_loader


# Normalize function
def normalize_features(features):
    mean = features.mean(dim=(2, 3), keepdim=True)
    std = features.std(dim=(2, 3), keepdim=True)
    normalized_features = (features - mean) / (std + 1e-8)
    return normalized_features

def normalize_to_01(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)


def calculate_psnr(original, reconstructed, max_pixel_value=1.0):
    mse = F.mse_loss(original, reconstructed, reduction='mean')
    if mse == 0:
        return float('inf')  # Avoid log(0) error

    psnr_value = 10 * torch.log10((max_pixel_value ** 2) / mse)
    return psnr_value.item()

def eval(loader, local_model1, local_model2, local_model3, vdenoiser1, vdenoiser2, loss_fn):
    val_running_loss, valid_running_correct, valid_f1_score, valid_accuracy = 0.0, 0.0, 0.0, 0.0
    iou_scores = torch.zeros(5)  # For IoU class scores

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.type(torch.LongTensor).to(device)

            # Forward pass with denoising
            enc1, predictions1 = local_model1(x)
            denoised_pred1 = denoise(normalize_features(forward_diffusion(predictions1, beta, alpha_cum, t)), vdenoiser1, alpha_cum, t, device)
            recon_loss1 = F.mse_loss(normalize_features(denoised_pred1), normalize_features(predictions1))

            predictions2 = local_model2(denoised_pred1)
            denoised_pred2 = denoise(normalize_features(forward_diffusion(predictions2, beta, alpha_cum, t)), vdenoiser2, alpha_cum, t, device)
            recon_loss2 = F.mse_loss(normalize_features(denoised_pred2), normalize_features(predictions2))

            predictions3 = local_model3(enc1, denoised_pred2)

            # Compute losses
            seg_loss = loss_fn(predictions3, y)
            loss = seg_loss + 0.5 * (recon_loss1 + recon_loss2)

            # Compute accuracy and IoU
            preds = torch.argmax(predictions3, dim=1)
            valid_running_correct += (preds == y).float().mean().item()
            val_running_loss += loss.item()
            valid_f1_score += f1_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            valid_accuracy += accuracy_score(y.cpu().flatten(), preds.cpu().flatten())

            # Compute IoU scores
            iou_scores += torch.tensor(jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None))

    # Compute epoch-level metrics
    num_samples = len(loader.dataset)
    epoch_loss = val_running_loss / num_samples
    epoch_acc = 100. * (valid_running_correct / num_samples)
    epoch_iou_scores = iou_scores / num_samples
    epoch_iou_withbackground = epoch_iou_scores.mean().item()
    epoch_iou_nobackground = epoch_iou_scores[1:].mean().item()
    return (epoch_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground) + tuple(epoch_iou_scores.tolist())


def test(loader, modelclientFE, modelserver, modelclientBE, tdenoiser1, tdenoiser2, loss_fn, folder):
    val_running_loss = 0.0
    seg_run_loss = 0.0
    recon1_run_loss = 0.0
    recon2_run_loss = 0.0
    valid_running_correct = 0.0
    valid_accuracy = 0.0
    valid_f1_score = 0.0
    iou_scores = [0.0] * 5
    test_running_psnr = 0.0
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.type(torch.LongTensor).to(device=device)

            # Forward pass
            enc1, predictions1 = modelclientFE(x)
            noisy_pred1 = forward_diffusion(predictions1, beta, alpha_cum, t)
            torch.save(noisy_pred1, folder + './noisy_pred1.pth')
            denoised_pred1 = denoise(normalize_features(noisy_pred1), tdenoiser1, alpha_cum, t, device)
            recon_loss1 = F.mse_loss(normalize_features(denoised_pred1), normalize_features(predictions1))
            predictions2 = modelserver(denoised_pred1)
            noisy_pred2 = forward_diffusion(predictions2, beta, alpha_cum, t)
            torch.save(noisy_pred2, folder + './noisy_pred2.pth')
            denoised_pred2 = denoise(normalize_features(noisy_pred2), tdenoiser2, alpha_cum, t, device)
            recon_loss2 = F.mse_loss(normalize_features(denoised_pred2), normalize_features(predictions2))
            predictions3 = modelclientBE(enc1, denoised_pred2)

            # Loss calculations
            seg_loss = loss_fn(predictions3, y)

            loss = seg_loss + 0.5 * (recon_loss1 + recon_loss2)

            # Accuracy and IoU calculations
            preds = torch.argmax(predictions3, dim=1)

            print("preds size:", preds.size())
            print("y size:", y.size())
            splitdfeddiffuse_psnr = calculate_psnr(y.float(),preds.float())
            print(f"splitdfeddiffuse_psnr: {splitdfeddiffuse_psnr}")
            test_running_psnr=splitdfeddiffuse_psnr

            valid_running_correct += (preds == y).float().mean().item()
            val_running_loss += loss.item()
            seg_run_loss += seg_loss.item()
            recon1_run_loss += recon_loss1.item()
            recon2_run_loss += recon_loss2.item()
            valid_f1_score += f1_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            valid_accuracy += accuracy_score(y.cpu().flatten(), preds.cpu().flatten())

            # Compute IoU scores
            iou_sklearn = jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None)
            for i in range(5):
                iou_scores[i] += iou_sklearn[i]

            # Save images
            torchvision.utils.save_image(preds.float(), f"{folder}/pred_{idx}.BMP", padding=0, scale_each=True, normalize=True)

    # Compute epoch-level metrics
    num_samples = len(loader.dataset)
    epoch_loss = val_running_loss / num_samples
    epoch_seg_loss = seg_run_loss / num_samples
    epoch_recon1_loss = recon1_run_loss / num_samples
    epoch_recon2_loss = recon2_run_loss / num_samples
    epoch_acc = 100. * (valid_running_correct / num_samples)
    epoch_iou_scores = [iou / num_samples for iou in iou_scores]
    epoch_iou_withbackground = sum(epoch_iou_scores) / 5
    epoch_iou_nobackground = sum(epoch_iou_scores[1:]) / 4
    epoch_psnr = test_running_psnr/num_samples
    print(f"Testing accuracy: {epoch_acc:.2f}%")
    print(f"Mean IoU with background: {epoch_iou_withbackground:.4f}")
    print(f"Mean IoU without background: {epoch_iou_nobackground:.4f}")
    print("epoch psnr:",epoch_psnr)
    for i, label in enumerate(["Background", "ZP", "TE", "ICM", "Blastocoel"]):
        print(f"IoU of {label}: {epoch_iou_scores[i]:.4f}")
    return epoch_loss, epoch_seg_loss, epoch_recon1_loss, epoch_recon2_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground

