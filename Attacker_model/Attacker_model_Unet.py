import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
torch.cuda.empty_cache()
device = "cuda"
import segmentation_models_pytorch as smp
from skimage.metrics import structural_similarity as ssim
import numpy as np
from AE import SimpleDecoder
from UNet2 import UNetAutoencoder
from scipy.linalg import sqrtm
from torch.utils.data import Subset
import shutil
CUDA_LAUNCH_BLOCKING=1
# Load Features from Checkpoint
#splitfed_features = "C:/Users/csj5/Projects/Attackmodel/InputFeatureCheckpoints/allpredictions1_Blasto.pth"
splitfeddifuse_features = "C:/Users/csj5/Projects/Attackmodel/InputFeatureCheckpoints/allnoisy_pred1_Covid_V1.pth"

CS1 = torch.load(splitfeddifuse_features, map_location="cuda")
if isinstance(CS1[0], torch.Tensor):
    featuresS1 = torch.stack(CS1)
elif isinstance(CS1[0], dict) and 'features' in CS1[0]:
    featuresS1 = torch.stack([item['features'] for item in CS1])
else:
    raise TypeError("Unexpected checkpoint format. Expected list of tensors or list of dicts.")
featuresS1 = F.adaptive_avg_pool3d(featuresS1, (4, 8, 8))  # Reduce spatial dimensions
featuresS1 = featuresS1.view(featuresS1.shape[0], -1)  # Flatten features

# Load Images
#imagesS1_path = "C:/Users/csj5/Projects/Data/BlastocystDATA/datafed/data/shuffled/federated/test_imgs_new"
imagesS1_path = "C:/Users/csj5/Projects/Data/mosmedplus/mosmedplus_federated/test_imgs"
#imagesS1_path ="C:/Users/csj5/Projects/Data/fhpsaop_512_federated/federated/test_imgs"
#imagesS1_path = "C:/Users/csj5/Projects/Data/HAM10000/HAM10000/datafed/highly_nonuniform/test_imgs"
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.01),  # Color jitte
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

image_filenames = sorted(os.listdir(imagesS1_path))
test_size = 23
train_filenames = image_filenames[:-test_size]
test_filenames = image_filenames[-test_size:]
train_images = [train_transform(Image.open(os.path.join(imagesS1_path, f)).convert("RGB")) for f in train_filenames]
test_images = [test_transform(Image.open(os.path.join(imagesS1_path, f)).convert("RGB")) for f in test_filenames]
train_images = torch.stack(train_images)
test_images = torch.stack(test_images)
print(f"Loaded train images: {train_images.shape}, test images: {test_images.shape}")

train_features = featuresS1[:-test_size]
test_features = featuresS1[-test_size:]

class FeatureImageDataset(Dataset):
    def __init__(self, features, images):
        self.features = features
        self.images = images

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.images[idx]

train_dataset = FeatureImageDataset(train_features, train_images)
test_dataset = FeatureImageDataset(test_features, test_images)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(f"Dataset loaded: {len(train_dataset)} train samples, {len(test_dataset)} test samples.")

feature_dim = featuresS1.shape[1]
img_channels = 3
img_size = 256
#attacker_model = SimpleDecoder(feature_dim, img_channels, img_size).to(device)
attacker_model = UNetAutoencoder(feature_dim, img_channels, img_size).to(device)

optimizer = optim.Adam(attacker_model.parameters(), lr=0.0001)

def calculate_fid(tensor1, tensor2):
    tensor1_np = tensor1.detach().cpu().numpy()
    tensor2_np = tensor2.detach().cpu().numpy()
    tensor1_np = tensor1_np.reshape(tensor1_np.shape[0], -1)
    tensor2_np = tensor2_np.reshape(tensor2_np.shape[0], -1)
    mu1, mu2 = tensor1_np.mean(axis=0), tensor2_np.mean(axis=0)
    if tensor1_np.shape[0] > 1:
        sigma1 = np.cov(tensor1_np, rowvar=False)
        sigma2 = np.cov(tensor2_np, rowvar=False)
    else:
        sigma1 = np.var(tensor1_np, axis=0, keepdims=True)
        sigma2 = np.var(tensor2_np, axis=0, keepdims=True)
    diff = mu1 - mu2
    try:
        covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except ValueError:
        covmean = np.zeros_like(sigma1)
    fid_distance = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid_distance

def save_training_image(images, epoch, batch_idx, output_folder="train_outputs"):
    os.makedirs(output_folder, exist_ok=True)
    image_grid = vutils.make_grid(images, normalize=True)
    save_path = os.path.join(output_folder, f"batch_{batch_idx+1}.png")
    vutils.save_image(image_grid, save_path)

def train_attacker(autoencoder, train_dataloader, optimizer, device="cuda", epochs=200,
                   image_save_dir="train_outputs", checkpoint_dir="checkpoints"):
    autoencoder.train()
    scaler = torch.cuda.amp.GradScaler()
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        epoch_fid = 0.0
        num_samples = 0

        for batch_idx, (features, original_images) in enumerate(train_dataloader):
            features, original_images = features.to(device), original_images.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                reconstructed_images = autoencoder(features)
                reconstructed_images = F.interpolate(reconstructed_images, size=(256, 256), mode="bilinear", align_corners=False)

                mse_loss = F.mse_loss(reconstructed_images, original_images, reduction='mean')
                psnr_value = 10 * torch.log10(1.0 / mse_loss)
                fid_value = calculate_fid(reconstructed_images, original_images)

                orig_img = original_images[0].detach().permute(1, 2, 0).cpu().numpy()
                recon_img = reconstructed_images[0].detach().permute(1, 2, 0).cpu().numpy()
                ssim_value = ssim(orig_img, recon_img, multichannel=True, data_range=1.0)

                loss = mse_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Save reconstructed training image
            save_training_image(reconstructed_images, epoch, batch_idx, image_save_dir)

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_psnr += psnr_value.item()
            epoch_fid += fid_value
            epoch_ssim += ssim_value
            num_samples += 1

            preds = (reconstructed_images > 0.5).int().cpu().numpy().flatten()
            targets = (original_images > 0.5).int().cpu().numpy().flatten()
            iou_sklearn = jaccard_score(targets, preds, average="macro")
            epoch_iou += iou_sklearn

        # Epoch-wise averages
        avg_loss = epoch_loss / num_samples
        avg_iou = epoch_iou / num_samples
        avg_psnr = epoch_psnr / num_samples
        avg_fid = epoch_fid / num_samples
        avg_ssim = epoch_ssim / num_samples

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}, IoU: {avg_iou:.4f}, "
              f"Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}, Avg FID: {avg_fid:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"attacker.pth")
        torch.save(autoencoder.state_dict(), checkpoint_path)

    print("Training completed.")

def save_generated_images(images, batch_idx, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    grid = vutils.make_grid(images, normalize=True)
    output_path = os.path.join(output_folder, f"pred_{batch_idx+1}.png")
    vutils.save_image(grid, output_path)
    print(f"Saved generated test images at: {output_path}")

def generate_images_from_features(autoencoder, test_dataloader, device="cuda"):
    #autoencoder.eval()
    total_loss = 0
    total_psnr = 0
    total_iou = 0
    total_mse = 0
    total_ssim = 0
    total_fid = 0
    num_samples = 0
    output_folder = "generated_test_images"
    os.makedirs(output_folder, exist_ok=True)
    with torch.no_grad():
        for batch_idx, (features, original_images) in enumerate(test_dataloader):
            features = features.to(device)
            original_img_path = os.path.join(imagesS1_path, image_filenames[batch_idx])
            with Image.open(original_img_path) as img:
                original_size = img.size
            reconstructed_images = autoencoder(features)
            reconstructed_images = F.interpolate(reconstructed_images, size=original_size[::-1], mode="bilinear", align_corners=False)
            original_images = F.interpolate(original_images.to(device), size=original_size[::-1], mode="bilinear", align_corners=False)
            save_generated_images(reconstructed_images, batch_idx, output_folder)
            fid_value = calculate_fid(reconstructed_images, original_images)
            mse = F.mse_loss(reconstructed_images, original_images, reduction='mean')
            psnr = 10 * torch.log10(1.0 /mse)
            total_psnr += psnr.item()
            orig_img = original_images[0].detach().permute(1, 2, 0).cpu().numpy()  # Single Image
            recon_img = reconstructed_images[0].detach().permute(1, 2, 0).cpu().numpy()  # Single Image
            ssim_value = ssim(orig_img, recon_img, multichannel=True, data_range=1.0)
            total_ssim +=ssim_value
            preds = (reconstructed_images > 0.5).int().cpu().numpy().flatten()
            targets = (original_images > 0.5).int().cpu().numpy().flatten()
            iou_sklearn = jaccard_score(targets, preds, average="macro")
            total_iou += iou_sklearn
            loss = mse
            total_loss += loss.item()
            total_fid += fid_value
            total_mse += mse
            num_samples += 1
    avg_loss = total_loss / num_samples
    avg_psnr = total_psnr / num_samples
    avg_fid = total_fid / num_samples
    avg_iou = total_iou / num_samples
    avg_mse = total_mse / num_samples
    avg_ssim = total_ssim / num_samples
    print(f"Testing Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.4f}, IoU: {avg_iou:.4f}, MSE: {avg_mse:.4f}, SSIM: {avg_ssim:.4f}, FID: {avg_fid:.4f}")
    print("Generated test images are saved.")


# Run Training & Testing
torch.cuda.empty_cache()
train_attacker(attacker_model, train_dataloader, optimizer, device="cuda", epochs=200)
generate_images_from_features(attacker_model, test_dataloader, device="cuda")

'''
#Testing
checkpoint =torch.load('attackermodel_checkpoint.pth',map_location=device)
if "model_state_dict" in checkpoint:
    attacker_model.load_state_dict(checkpoint["model_state_dict"])
else:
    attacker_model.load_state_dict(checkpoint)  # If directly saved as a state_dict
generate_images_from_features(attacker_model, test_dataloader, criterion, device="cuda")
'''