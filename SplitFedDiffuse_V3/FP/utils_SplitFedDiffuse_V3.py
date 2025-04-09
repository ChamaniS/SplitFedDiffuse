import torch
import torchvision
from dataset import HAMDataset
from torch.utils.data import DataLoader
device = "cuda"
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import torch.nn.functional as F
from forward_diff import forward_diffusion
from reverse_diff1 import initialize_denoiser, denoise

# Define the noise schedule (linear schedule in this case)
T = 1000  # Total number of timesteps
st_noise = 0.0001
end_noise = 0.02
beta = torch.linspace(st_noise, end_noise, T).to(device)  # Linear noise schedule
alpha = 1 - beta
alpha_cum = torch.cumprod(alpha, dim=0)  # Cumulative product of alpha values
alpha_cum_t = alpha_cum.to(device)  # Move to the correct device
t = torch.randint(0, T, (1,)).item()  # current timestep (it determines how much noise has been removed from the data at that point)
#t=0

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
    train_ds = HAMDataset(
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

    val_ds = HAMDataset(
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
    test_ds = HAMDataset(
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
    """
    Normalize features to have zero mean and unit variance along the feature dimensions.
    Args:
        features (torch.Tensor): Input feature tensor.
    Returns:
        torch.Tensor: Normalized features.
    """
    mean = features.mean(dim=(2, 3), keepdim=True)
    std = features.std(dim=(2, 3), keepdim=True)
    normalized_features = (features - mean) / (std + 1e-8)
    return normalized_features

def eval(loader, local_model1, local_model2, local_model3, vdenoiser1, vdenoiser2, global_dn_pred1, global_dn_pred2,
         loss_fn):
    # local_model1.eval()
    # local_model2.eval()
    # local_model3.eval()
    val_running_loss = 0.0
    valid_running_correct = 0.0
    valid_iou_score_class0 = 0.0
    valid_iou_score_class1 = 0.0
    valid_accuracy = 0.0
    valid_f1_score = 0.0
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.type(torch.LongTensor).to(device)
            enc1, predictions1 = local_model1(x)
            noisy_pred1 = forward_diffusion(predictions1, beta, alpha_cum, t)

            denoised_pred1 = denoise(normalize_features(noisy_pred1), vdenoiser1, alpha_cum, t, device)
            recon_loss1 = F.mse_loss(normalize_features(denoised_pred1), normalize_features(predictions1))

            #attention_weights2 = torch.sigmoid(torch.mean(denoised_pred1, dim=(2, 3), keepdim=True))
            #combined_denoised_pred1 = attention_weights2 * normalize_features(denoised_pred1) + (1 - attention_weights2) * normalize_features(global_dn_pred1)
            predictions2 = local_model2(denoised_pred1)
            noisy_pred2 = forward_diffusion(predictions2, beta, alpha_cum, t)

            denoised_pred2 = denoise(normalize_features(noisy_pred2), vdenoiser2, alpha_cum, t, device)
            recon_loss2 = F.mse_loss(normalize_features(denoised_pred2), normalize_features(predictions2))
            #attention_weights4 = torch.sigmoid(torch.mean(denoised_pred2, dim=(2, 3), keepdim=True))
            #combined_denoised_pred2 = attention_weights4 * normalize_features(denoised_pred2) + (1 - attention_weights4) * normalize_features(global_dn_pred2)

            predictions3 = local_model3(enc1, denoised_pred2)

            seg_loss = loss_fn(predictions3, y)
            kl_weight = 0.00001
            recon_weight = 0.5
            loss = seg_loss + recon_weight * (recon_loss2 + recon_loss1)
            preds = torch.argmax(predictions3, dim=1)
            equals = preds == y
            valid_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()
            val_running_loss += loss.item()
            valid_f1_score += f1_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            valid_accuracy += accuracy_score(y.cpu().flatten(), preds.cpu().flatten())
            iou_sklearn = jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None)
            valid_iou_score_class0 += iou_sklearn[0]
            valid_iou_score_class1 += iou_sklearn[1]
            # torchvision.utils.save_image(preds.float(), f"{folder}/pred_{idx}.BMP", padding=0,scale_each=True,normalize=True)
    epoch_loss = val_running_loss / len(loader.dataset)
    epoch_acc = 100. * (valid_running_correct / len(loader.dataset))
    epoch_iou_class0 = (valid_iou_score_class0 / len(loader.dataset))
    epoch_iou_class1 = (valid_iou_score_class1 / len(loader.dataset))

    epoch_iou_withbackground = (epoch_iou_class0 + epoch_iou_class1) / 2
    epoch_iou_nobackground = epoch_iou_class1
    return epoch_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground, epoch_iou_class0, epoch_iou_class1


def test(loader, modelclientFE, modelserver, modelclientBE, tdenoiser1, tdenoiser2, loss_fn, folder):
    # modelclientFE.eval()
    # modelserver.eval()
    # modelclientBE.eval()
    val_running_loss = 0.0
    seg_run_loss = 0.0
    recon1_run_loss = 0.0
    recon2_run_loss = 0.0
    valid_running_correct = 0.0
    valid_iou_score_class0 = 0.0
    valid_iou_score_class1 = 0.0
    valid_accuracy = 0.0
    valid_f1_score = 0.0
    allpredictions1 = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.type(torch.LongTensor).to(device=device)
            enc1, predictions1 = modelclientFE(x)
            noisy_pred1 = forward_diffusion(predictions1, beta, alpha_cum, t)
            allpredictions1.append(noisy_pred1)
            denoised_pred1 = denoise(normalize_features(noisy_pred1), tdenoiser1, alpha_cum, t, device)
            recon_loss1 = F.mse_loss(normalize_features(denoised_pred1), normalize_features(predictions1))

            predictions2 = modelserver(denoised_pred1)
            noisy_pred2 = forward_diffusion(predictions2, beta, alpha_cum, t)

            denoised_pred2 = denoise(normalize_features(noisy_pred2), tdenoiser2, alpha_cum, t, device)
            recon_loss2 = F.mse_loss(normalize_features(denoised_pred2), normalize_features(predictions2))

            predictions3 = modelclientBE(enc1, denoised_pred2)

            seg_loss = loss_fn(predictions3, y)
            kl_weight = 0.00001
            recon_weight = 0.5
            loss = seg_loss + recon_weight * (recon_loss2 + recon_loss1)

            preds = torch.argmax(predictions3, dim=1)
            equals = preds == y
            valid_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()
            val_running_loss += loss.item()
            seg_run_loss += seg_loss.item()
            recon1_run_loss += recon_loss1.item()
            recon2_run_loss += recon_loss2.item()
            valid_f1_score += f1_score(y.cpu().flatten(), preds.cpu().flatten(), average='micro')
            valid_accuracy += accuracy_score(y.cpu().flatten(), preds.cpu().flatten())
            iou_sklearn = jaccard_score(y.cpu().flatten(), preds.cpu().flatten(), average=None)
            valid_iou_score_class0 += iou_sklearn[0]
            valid_iou_score_class1 += iou_sklearn[1]
            torchvision.utils.save_image(preds.float(), f"{folder}/pred_{idx}.BMP", padding=0, scale_each=True,
                                         normalize=True)
    epoch_loss = val_running_loss / len(loader.dataset)
    epoch_seg_loss = seg_run_loss / len(loader.dataset)
    epoch_recon1_loss = recon1_run_loss / len(loader.dataset)
    epoch_recon2_loss = recon2_run_loss / len(loader.dataset)
    epoch_acc = 100. * (valid_running_correct / len(loader.dataset))
    epoch_iou_class0 = (valid_iou_score_class0 / len(loader.dataset))
    epoch_iou_class1 = (valid_iou_score_class1 / len(loader.dataset))
    epoch_iou_withbackground = (epoch_iou_class0 + epoch_iou_class1 ) / 2
    epoch_iou_nobackground = epoch_iou_class1
    print("Testing accuracy score:", epoch_acc)
    print("Testing mean IoU withbackground:", epoch_iou_withbackground)
    print("Testing mean IoU withoutbackground:", epoch_iou_nobackground)
    print("IoU of Background:", epoch_iou_class0)
    print("IoU of Area 1:", epoch_iou_class1)
    torch.save(allpredictions1, '/lustre06/project/6008975/csj5/narvalenv/allnoisy_pred1_HAM_V3.pth')
    return epoch_loss, epoch_seg_loss, epoch_recon1_loss, epoch_recon2_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground, denoised_pred1, denoised_pred2
