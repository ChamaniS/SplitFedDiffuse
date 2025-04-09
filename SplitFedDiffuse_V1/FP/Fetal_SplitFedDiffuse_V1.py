import copy
import math
import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # Restrict OpenMP threads
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from utils_SplitFedDiffuse_V1 import (get_loaders, eval, get_loaders_test, test)
from forward_diff import forward_diffusion
from reverse_diff1 import initialize_denoiser, denoise
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.metrics import jaccard_score
from options import args_parser
from agg.Fed_Avg import fedAvg
import numpy as np
import pandas as pd
import random
import time
import torch.nn as nn
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
#from torchvision.models import vit_b_16
import sys
import timm

from reshapenet import reshapenet

CUDA_LAUNCH_BLOCKING = 1

# Hyperparameters
LEARNING_RATE = 0.0001
device = "cuda"
NUM_WORKERS = 1
SHUFFLE = False
NUM_CLASSES = 3
PIN_MEMORY = False

# Define the noise schedule (linear schedule in this case)
T = 1000  # Total number of timesteps for forward diffusion
st_noise = 0.0001
end_noise = 0.02
beta = torch.linspace(st_noise, end_noise, T).to(device)  # Linear noise schedule
alpha = 1 - beta
alpha_cum = torch.cumprod(alpha, dim=0)  # Cumulative product of alpha values
alpha_cum_t = alpha_cum.to(device)  # Move to the correct device
t = torch.randint(0, T, (1,)).item()  # Sample a timestep
# t=0

parentF="C:/Users/csj5/Projects/Fetal_segmentation"
DataF = "C:/Users/csj5/Projects/Data/fhpsaop_512_federated/federated"
save_F = "C:/Users/csj5/Projects/Fetal_segmentation/Saved"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.cuda.empty_cache()
ifcompressed = 1
from models.clientmodel_FE2 import UNET_FE
from models.clientmodel_BE2 import UNET_BE
from models.servermodel2 import UNET_server

TRAIN_IMG_DIR_C1 = DataF+"./client1/train_imgs/"
TRAIN_MASK_DIR_C1 = DataF+"./client1/train_masks/"
VAL_IMG_DIR_C1 = DataF+"./client4/val_imgs/"
VAL_MASK_DIR_C1 = DataF+"./client4/val_masks/"

#client 2
TRAIN_IMG_DIR_C2 = DataF+"./client4/train_imgs/"
TRAIN_MASK_DIR_C2 = DataF+"./client4/train_masks/"
VAL_IMG_DIR_C2 = DataF+"./client4/val_imgs/"
VAL_MASK_DIR_C2 = DataF+"./client4/val_masks/"

#client 3
TRAIN_IMG_DIR_C3 = DataF+"./client4/train_imgs/"
TRAIN_MASK_DIR_C3 = DataF+"./client4/train_masks/"
VAL_IMG_DIR_C3 = DataF+"./client4/val_imgs/"
VAL_MASK_DIR_C3 = DataF+"./client4/val_masks/"

#client 4
TRAIN_IMG_DIR_C4 = DataF+"./client4/train_imgs/"
TRAIN_MASK_DIR_C4 = DataF+"./client4/train_masks/"
VAL_IMG_DIR_C4 = DataF+"./client4/val_imgs/"
VAL_MASK_DIR_C4 = DataF+"./client4/val_masks/"

#client 5
TRAIN_IMG_DIR_C5 = DataF+"./client4/train_imgs/"
TRAIN_MASK_DIR_C5 = DataF+"./client4/train_masks/"
VAL_IMG_DIR_C5 = DataF+"./client4/val_imgs/"
VAL_MASK_DIR_C5 = DataF+"./client4/val_masks/"

TEST_IMG_DIR = DataF+"./attacker_test_images/"
TEST_MASK_DIR = DataF+"./attacker_test_masks/"

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


def train(train_loader, local_model1, local_model2, local_model3,local_ViTmodel,reshapenet, denoiser1, denoiser2, global_dn_pred1,
          global_dn_pred2, optimizer1, optimizer2, optimizer3, loss_fn):
    # Backward hook for modelclientBE
    grads3 = 0
    def grad_hook1(model, grad_input, grad_output):
        nonlocal grads3
        grads3 = grad_input[0].clone().detach()
    local_model3.decoder2_2.register_full_backward_hook(grad_hook1)

    grads2 = 0
    def grad_hook2(model, grad_input, grad_output):
        nonlocal grads2
        grads2 = grad_input[0].clone().detach()
    local_model2.encoder2_2.register_full_backward_hook(grad_hook2)

    loop = tqdm(train_loader)
    train_running_loss = 0.0
    train_running_correct = 0.0
    train_iou_score_class0 = 0.0
    train_iou_score_class1 = 0.0
    train_iou_score_class2 = 0.0
    recon_loss1_running = 0.0
    LDDM1_loss_running = 0.0
    all_predictions1 = []
    all_predictions1_noisy = []
    all_ViT_out_noisy = []
    all_ViT_out_noisy_resized = []
    all_combined_pred1 = []
    all_denoised_pred1 = []
    all_combined_denoised_pred1 = []
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.type(torch.LongTensor).to(device)
        enc1, predictions1 = local_model1(data)
        all_predictions1.append(predictions1.detach().cpu())
        targetsR = targets.unsqueeze(1)
        targetsR = targetsR.repeat(1, 3, 1, 1)
        targetsR = targetsR.float().to(device)
        targetsR = F.interpolate(targetsR, size=(224, 224), mode="bilinear", align_corners=False)
        ViT_out = local_ViTmodel(targetsR)

        ViT_out = ViT_out.detach()
        ViT_out = ViT_out[:, :1, None, None]
        ViT_out_reshaped = reshapenet(ViT_out)
        ViT_out_resized = F.interpolate(ViT_out_reshaped, size=(112, 112), mode="bilinear", align_corners=False)
        ViT_out_noisy = forward_diffusion(ViT_out_resized, beta, alpha_cum, t)

        attention_weights1 = torch.sigmoid(torch.mean(predictions1, dim=(2, 3), keepdim=True))
        combined_pred1 = attention_weights1 * normalize_features(predictions1) + (
                    1 - attention_weights1) * normalize_features(ViT_out_noisy)
        all_combined_pred1.append(combined_pred1.detach().cpu())
        # combined_pred1 = normalize_features(predictions1) + normalize_features(ViT_out_noisy_resized)

        # Denoising model 1
        denoised_pred1 = denoise(normalize_features(combined_pred1), denoiser1, alpha_cum, t, device)
        recon_loss1 = F.mse_loss(normalize_features(denoised_pred1),
                                 normalize_features(predictions1))  # reconstruction loss of LDDM-1
        all_denoised_pred1.append(denoised_pred1.detach().cpu())
        recon_loss1_running += recon_loss1.item()

        LDDM1_loss = F.mse_loss(normalize_features(denoised_pred1), normalize_features(combined_pred1))
        LDDM1_loss_running += LDDM1_loss.item()

        attention_weights2 = torch.sigmoid(torch.mean(denoised_pred1, dim=(2, 3), keepdim=True))
        combined_denoised_pred1 = attention_weights2 * normalize_features(denoised_pred1) + (
                    1 - attention_weights2) * normalize_features(global_dn_pred1)

        predictions2 = local_model2(combined_denoised_pred1)
        attention_weights3 = torch.sigmoid(torch.mean(predictions2, dim=(2, 3), keepdim=True))
        combined_pred2 = attention_weights3 * normalize_features(predictions2) + (
                    1 - attention_weights3) * normalize_features(ViT_out_noisy_resized)

        denoised_pred2 = denoise(normalize_features(combined_pred2), denoiser2, alpha_cum, t, device)
        recon_loss2 = F.mse_loss(normalize_features(denoised_pred2),
                                 normalize_features(predictions2))  # reconstruction loss of LDDM-2
        attention_weights4 = torch.sigmoid(torch.mean(denoised_pred2, dim=(2, 3), keepdim=True))
        combined_denoised_pred2 = attention_weights4 * normalize_features(denoised_pred2) + (
                    1 - attention_weights4) * normalize_features(global_dn_pred2)

        predictions3 = local_model3(enc1, combined_denoised_pred2)
        seg_loss = loss_fn(predictions3, targets)
        recon_weight = 0.5
        loss = seg_loss + recon_weight * (recon_loss2 + recon_loss1)

        preds = torch.argmax(predictions3, dim=1)
        equals = preds == targets
        train_running_correct += torch.mean(equals.type(torch.FloatTensor)).item()
        train_running_loss += loss.item()
        iou_sklearn = jaccard_score(targets.cpu().flatten(), preds.cpu().flatten(), average=None)
        train_iou_score_class0 += iou_sklearn[0]
        train_iou_score_class1 += iou_sklearn[1]
        train_iou_score_class2 += iou_sklearn[2]
        loss.backward(retain_graph=True)
        optimizer3.step()
        optimizer3.zero_grad()
        mygrad3 = grads3
        predictions2.backward(mygrad3, retain_graph=True)
        optimizer2.step()
        optimizer2.zero_grad()
        mygrad2 = grads2
        predictions1.backward(mygrad2)
        optimizer1.step()
        optimizer1.zero_grad()
        loop.set_postfix(loss=loss.item())
    epoch_loss = train_running_loss / len(train_loader.dataset)
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    epoch_iou_class0 = (train_iou_score_class0 / len(train_loader.dataset))
    epoch_iou_class1 = (train_iou_score_class1 / len(train_loader.dataset))
    epoch_iou_class2 = (train_iou_score_class2 / len(train_loader.dataset))
    epoch_iou_withbackground = (epoch_iou_class0 + epoch_iou_class1 + epoch_iou_class2) / 3
    epoch_iou_nobackground = (epoch_iou_class1 + epoch_iou_class2) / 2
    return epoch_loss, epoch_acc, epoch_iou_withbackground, epoch_iou_nobackground, epoch_iou_class0, epoch_iou_class1, epoch_iou_class2


# 2. Main function
def main():
    args = args_parser()
    start_time = time.time()
    train_transform = A.Compose(
        [
            A.Resize(height=args.image_height, width=args.image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=args.image_height, width=args.image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_loader_C1, val_loader_C1 = get_loaders(
        TRAIN_IMG_DIR_C1,
        TRAIN_MASK_DIR_C1,
        VAL_IMG_DIR_C1,
        VAL_MASK_DIR_C1,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY

    )

    train_loader_C2, val_loader_C2 = get_loaders(
        TRAIN_IMG_DIR_C2,
        TRAIN_MASK_DIR_C2,
        VAL_IMG_DIR_C2,
        VAL_MASK_DIR_C2,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY

    )

    train_loader_C3, val_loader_C3 = get_loaders(
        TRAIN_IMG_DIR_C3,
        TRAIN_MASK_DIR_C3,
        VAL_IMG_DIR_C3,
        VAL_MASK_DIR_C3,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    train_loader_C4, val_loader_C4 = get_loaders(
        TRAIN_IMG_DIR_C4,
        TRAIN_MASK_DIR_C4,
        VAL_IMG_DIR_C4,
        VAL_MASK_DIR_C4,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    train_loader_C5, val_loader_C5 = get_loaders(
        TRAIN_IMG_DIR_C5,
        TRAIN_MASK_DIR_C5,
        VAL_IMG_DIR_C5,
        VAL_MASK_DIR_C5,
        args.local_bs,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    test_loader = get_loaders_test(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        train_transform
    )

    global_model1_fed = UNET_FE(in_channels=3).to(device)
    global_model2_fed = UNET_server(in_channels=32).to(device)
    global_model3_fed = UNET_BE(out_channels=NUM_CLASSES).to(device)

    global_ViTmodel =timm.create_model("vit_base_patch16_224", pretrained=True)

    test_folder = save_F + "/testingsaved"
    loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
    global_denoiser1 = initialize_denoiser(input_dim=64, output_dim=64, hidden_dim=128, device=device)
    global_denoiser2 = initialize_denoiser(input_dim=64, output_dim=64, hidden_dim=128, device=device)
    global_reshapenet = reshapenet(in_channels=1).to(device)
    test_epoch_loss, test_epoch_seg_loss,test_epoch_recon1_loss,test_epoch_recon2_loss,test_epoch_acc, test_epoch_accwithback,test_epoch_accnoback,global_dn_pred1,global_dn_pred2 = test(test_loader, global_model1_fed, global_model2_fed, global_model3_fed,global_ViTmodel,global_reshapenet,global_denoiser1,global_denoiser2, loss_fn, test_folder)

    # global round
    client1_train_acc, client1_train_loss, client1_train_withbackiou, client1_train_nobackiou, client1_val_acc, client1_val_loss, client1_val_withbackiou, client1_val_nobackiou, client1_g_val_acc, client1_g_val_loss, client1_g_val_iouwithback, client1_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client2_train_acc, client2_train_loss, client2_train_withbackiou, client2_train_nobackiou, client2_val_acc, client2_val_loss, client2_val_withbackiou, client2_val_nobackiou, client2_g_val_acc, client2_g_val_loss, client2_g_val_iouwithback, client2_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client3_train_acc, client3_train_loss, client3_train_withbackiou, client3_train_nobackiou, client3_val_acc, client3_val_loss, client3_val_withbackiou, client3_val_nobackiou, client3_g_val_acc, client3_g_val_loss, client3_g_val_iouwithback, client3_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client4_train_acc, client4_train_loss, client4_train_withbackiou, client4_train_nobackiou, client4_val_acc, client4_val_loss, client4_val_withbackiou, client4_val_nobackiou, client4_g_val_acc, client4_g_val_loss, client4_g_val_iouwithback, client4_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    client5_train_acc, client5_train_loss, client5_train_withbackiou, client5_train_nobackiou, client5_val_acc, client5_val_loss, client5_val_withbackiou, client5_val_nobackiou, client5_g_val_acc, client5_g_val_loss, client5_g_val_iouwithback, client5_g_val_iounoback = [], [], [], [], [], [], [], [], [], [], [], []
    test_Acc, test_Iou_withback, test_Iou_noback, test_Loss, test_seg_loss, test_recon1_loss, test_recon2_loss, test_KL1_loss, test_KL2_loss = [], [], [], [], [], [], [], [], []
    least_lossg = 100000000;
    least_losst = 100000000;

    for com_round in (range(args.rounds)):
        local_weights1,local_weights2,local_weights3,ViT_weights, den1_weights,den2_weights,reshape_weights =[],[],[],[],[],[],[]
        least_lossC1, least_lossC2, least_lossC3, least_lossC4, least_lossC5 = 100000000, 100000000, 100000000, 100000000, 100000000;
        round_idx = com_round + 1

        # --------------------------------------LOCAL TRAINING & VALIDATING---------------------------------------------------------------------------
        print(f'\n | Global Training Round : {round_idx} |\n')
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local_model1 =copy.deepcopy(global_model1_fed)
            local_model2 =copy.deepcopy(global_model2_fed)
            local_model3 = copy.deepcopy(global_model3_fed)
            local_ViTmodel = copy.deepcopy(global_ViTmodel)
            local_denoiser1 = copy.deepcopy(global_denoiser1)
            local_denoiser2 = copy.deepcopy(global_denoiser2)
            local_reshapenet = copy.deepcopy(global_reshapenet)

            optimizer1 = optim.Adam(local_model1.parameters(), lr=LEARNING_RATE)
            optimizer2 = optim.Adam(local_model2.parameters(), lr=LEARNING_RATE)
            optimizer3 = optim.Adam(local_model3.parameters(), lr=LEARNING_RATE)

            cl_idx = idx + 1
            print("Selected client:", cl_idx)
            if cl_idx == 1:
                train_loader = train_loader_C1
                val_loader = val_loader_C1
                #folder = save_F + "./Saved/local_models/client1"
            elif cl_idx == 2:
                train_loader = train_loader_C2
                val_loader = val_loader_C2
                #folder = save_F + "./Saved/local_models/client2"
            elif cl_idx == 3:
                train_loader = train_loader_C3
                val_loader = val_loader_C3
                #folder = save_F + "./Saved/local_models/client3"
            elif cl_idx == 4:
                train_loader = train_loader_C4
                val_loader = val_loader_C4
                #folder = save_F + "./Saved/local_models/client4"
            elif cl_idx == 5:
                train_loader = train_loader_C5
                val_loader = val_loader_C5
                #folder = save_F + "./Saved/local_models/client5"

            # local epoch
            for epoch in range(args.local_ep):
                print(f"[INFO]: Epoch {epoch + 1} of {args.local_ep}")
                print("Client", cl_idx, " training.........")
                if cl_idx == 1:
                    train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1,trainepoch_iou_class2= train(train_loader, local_model1 , local_model2, local_model3,local_ViTmodel,local_reshapenet,local_denoiser1,local_denoiser2,global_dn_pred1,global_dn_pred2,optimizer1,optimizer2,optimizer3,loss_fn)
                    print("Client", cl_idx, "local validating.........")
                    val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1,valepoch_iou_class2 = eval(
                        val_loader, local_model1 , local_model2, local_model3,local_ViTmodel,local_reshapenet, local_denoiser1,local_denoiser2,global_dn_pred1,global_dn_pred2,loss_fn)
                    client1_train_acc.append(train_epoch_acc)
                    client1_train_loss.append(train_epoch_loss)
                    client1_train_withbackiou.append(trainepoch_iou_withbackground)
                    client1_train_nobackiou.append(trainepoch_iou_nobackground)
                    client1_val_acc.append(val_epoch_acc)
                    client1_val_loss.append(val_epoch_loss)
                    client1_val_withbackiou.append(valepoch_iou_withbackground)
                    client1_val_nobackiou.append(valepoch_iou_nobackground)
                    if least_lossC1 > val_epoch_loss:
                        least_lossC1 = val_epoch_loss
                        torch.save(local_model1.state_dict(),'./Saved/Checkpoints/C1M1_localcheckpoint.pth')
                        torch.save(local_model2.state_dict(),'./Saved/Checkpoints/C1M2_localcheckpoint.pth')
                        torch.save(local_model3.state_dict(),'./Saved/Checkpoints/C1M3_localcheckpoint.pth')
                        torch.save(local_ViTmodel.state_dict(),'./Saved/Checkpoints/C1_local_ViTmodel.pth')
                        torch.save(local_denoiser1.state_dict(),'./Saved/Checkpoints/C1_denoiser1.pth')
                        torch.save(local_denoiser2.state_dict(),'./Saved/Checkpoints/C1_denoiser2.pth')
                        torch.save(local_reshapenet.state_dict(),'./Saved/Checkpoints/C1_local_reshapenet.pth')
                        print('C1localmodel saved')
                if cl_idx == 2:
                    train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1,trainepoch_iou_class2= train(train_loader, local_model1 , local_model2, local_model3,local_ViTmodel,local_reshapenet,local_denoiser1,local_denoiser2,global_dn_pred1,global_dn_pred2,optimizer1,optimizer2,optimizer3,loss_fn)
                    print("Client", cl_idx, "local validating.........")
                    val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1,valepoch_iou_class2 = eval(
                        val_loader, local_model1 , local_model2, local_model3,local_ViTmodel,local_reshapenet, local_denoiser1,local_denoiser2,global_dn_pred1,global_dn_pred2,loss_fn)
                    client2_train_acc.append(train_epoch_acc)
                    client2_train_loss.append(train_epoch_loss)
                    client2_train_withbackiou.append(trainepoch_iou_withbackground)
                    client2_train_nobackiou.append(trainepoch_iou_nobackground)
                    client2_val_acc.append(val_epoch_acc)
                    client2_val_loss.append(val_epoch_loss)
                    client2_val_withbackiou.append(valepoch_iou_withbackground)
                    client2_val_nobackiou.append(valepoch_iou_nobackground)
                    if least_lossC2 > val_epoch_loss:
                        least_lossC2 = val_epoch_loss
                        torch.save(local_model1.state_dict(),'./Saved/Checkpoints/C2M1_localcheckpoint.pth')
                        torch.save(local_model2.state_dict(),'./Saved/Checkpoints/C2M2_localcheckpoint.pth')
                        torch.save(local_model3.state_dict(),'./Saved/Checkpoints/C2M3_localcheckpoint.pth')
                        torch.save(local_ViTmodel.state_dict(),'./Saved/Checkpoints/C2_local_ViTmodel.pth')
                        torch.save(local_denoiser1.state_dict(),'./Saved/Checkpoints/C2_denoiser1.pth')
                        torch.save(local_denoiser2.state_dict(),'./Saved/Checkpoints/C2_denoiser2.pth')
                        torch.save(local_reshapenet.state_dict(),'./Saved/Checkpoints/C2_local_reshapenet.pth')
                        print('C2localmodel saved')
                if cl_idx == 3:
                    train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1,trainepoch_iou_class2= train(train_loader, local_model1 , local_model2, local_model3,local_ViTmodel,local_reshapenet,local_denoiser1,local_denoiser2,global_dn_pred1,global_dn_pred2,optimizer1,optimizer2,optimizer3,loss_fn)
                    print("Client", cl_idx, "local validating.........")
                    val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1,valepoch_iou_class2 = eval(
                        val_loader, local_model1 , local_model2, local_model3,local_ViTmodel,local_reshapenet, local_denoiser1,local_denoiser2,global_dn_pred1,global_dn_pred2,loss_fn)
                    client3_train_acc.append(train_epoch_acc)
                    client3_train_loss.append(train_epoch_loss)
                    client3_train_withbackiou.append(trainepoch_iou_withbackground)
                    client3_train_nobackiou.append(trainepoch_iou_nobackground)
                    client3_val_acc.append(val_epoch_acc)
                    client3_val_loss.append(val_epoch_loss)
                    client3_val_withbackiou.append(valepoch_iou_withbackground)
                    client3_val_nobackiou.append(valepoch_iou_nobackground)
                    if least_lossC3 > val_epoch_loss:
                        least_lossC3 = val_epoch_loss
                        torch.save(local_model1.state_dict(),'./Saved/Checkpoints/C3M1_localcheckpoint.pth')
                        torch.save(local_model2.state_dict(),'./Saved/Checkpoints/C3M2_localcheckpoint.pth')
                        torch.save(local_model3.state_dict(),'./Saved/Checkpoints/C3M3_localcheckpoint.pth')
                        torch.save(local_ViTmodel.state_dict(),'./Saved/Checkpoints/C3_local_ViTmodel.pth')
                        torch.save(local_denoiser1.state_dict(),'./Saved/Checkpoints/C3_denoiser1.pth')
                        torch.save(local_denoiser2.state_dict(),'./Saved/Checkpoints/C3_denoiser2.pth')
                        torch.save(local_reshapenet.state_dict(),'./Saved/Checkpoints/C3_local_reshapenet.pth')
                        print('C3localmodel saved')
                if cl_idx == 4:
                    train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1,trainepoch_iou_class2= train(train_loader, local_model1 , local_model2, local_model3,local_ViTmodel,local_reshapenet,local_denoiser1,local_denoiser2,global_dn_pred1,global_dn_pred2,optimizer1,optimizer2,optimizer3,loss_fn)
                    print("Client", cl_idx, "local validating.........")
                    val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1,valepoch_iou_class2 = eval(
                        val_loader, local_model1 , local_model2, local_model3,local_ViTmodel,local_reshapenet, local_denoiser1,local_denoiser2,global_dn_pred1,global_dn_pred2,loss_fn)
                    client4_train_acc.append(train_epoch_acc)
                    client4_train_loss.append(train_epoch_loss)
                    client4_train_withbackiou.append(trainepoch_iou_withbackground)
                    client4_train_nobackiou.append(trainepoch_iou_nobackground)
                    client4_val_acc.append(val_epoch_acc)
                    client4_val_loss.append(val_epoch_loss)
                    client4_val_withbackiou.append(valepoch_iou_withbackground)
                    client4_val_nobackiou.append(valepoch_iou_nobackground)
                    if least_lossC4 > val_epoch_loss:
                        least_lossC4 = val_epoch_loss
                        torch.save(local_model1.state_dict(),'./Saved/Checkpoints/C4M1_localcheckpoint.pth')
                        torch.save(local_model2.state_dict(),'./Saved/Checkpoints/C4M2_localcheckpoint.pth')
                        torch.save(local_model3.state_dict(),'./Saved/Checkpoints/C4M3_localcheckpoint.pth')
                        torch.save(local_ViTmodel.state_dict(),'./Saved/Checkpoints/C4_local_ViTmodel.pth')
                        torch.save(local_denoiser1.state_dict(),'./Saved/Checkpoints/C4_denoiser1.pth')
                        torch.save(local_denoiser2.state_dict(),'./Saved/Checkpoints/C4_denoiser2.pth')
                        torch.save(local_reshapenet.state_dict(),'./Saved/Checkpoints/C4_local_reshapenet.pth')
                        print('C4localmodel saved')
                if cl_idx == 5:
                    train_epoch_loss, train_epoch_acc, trainepoch_iou_withbackground, trainepoch_iou_nobackground, trainepoch_iou_class0, trainepoch_iou_class1,trainepoch_iou_class2= train(train_loader, local_model1 , local_model2, local_model3,local_ViTmodel,local_reshapenet,local_denoiser1,local_denoiser2,global_dn_pred1,global_dn_pred2,optimizer1,optimizer2,optimizer3,loss_fn)
                    print("Client", cl_idx, "local validating.........")
                    val_epoch_loss, val_epoch_acc, valepoch_iou_withbackground, valepoch_iou_nobackground, valepoch_iou_class0, valepoch_iou_class1,valepoch_iou_class2 = eval(
                        val_loader, local_model1 , local_model2, local_model3,local_ViTmodel,local_reshapenet, local_denoiser1,local_denoiser2,global_dn_pred1,global_dn_pred2,loss_fn)
                    client5_train_acc.append(train_epoch_acc)
                    client5_train_loss.append(train_epoch_loss)
                    client5_train_withbackiou.append(trainepoch_iou_withbackground)
                    client5_train_nobackiou.append(trainepoch_iou_nobackground)
                    client5_val_acc.append(val_epoch_acc)
                    client5_val_loss.append(val_epoch_loss)
                    client5_val_withbackiou.append(valepoch_iou_withbackground)
                    client5_val_nobackiou.append(valepoch_iou_nobackground)
                    if least_lossC5 > val_epoch_loss:
                        least_lossC5 = val_epoch_loss
                        torch.save(local_model1.state_dict(),'./Saved/Checkpoints/C5M1_localcheckpoint.pth')
                        torch.save(local_model2.state_dict(),'./Saved/Checkpoints/C5M2_localcheckpoint.pth')
                        torch.save(local_model3.state_dict(),'./Saved/Checkpoints/C5M3_localcheckpoint.pth')
                        torch.save(local_ViTmodel.state_dict(),'./Saved/Checkpoints/C5_local_ViTmodel.pth')
                        torch.save(local_denoiser1.state_dict(),'./Saved/Checkpoints/C5_denoiser1.pth')
                        torch.save(local_denoiser2.state_dict(),'./Saved/Checkpoints/C5_denoiser2.pth')
                        torch.save(local_reshapenet.state_dict(),'./Saved/Checkpoints/C5_local_reshapenet.pth')
                        print('C5localmodel saved')

                print(
                    f"Training dice loss: {train_epoch_loss:.3f}, Training accuracy: {train_epoch_acc:.3f},Training iou Score with background: {trainepoch_iou_withbackground:.3f},Training iou Score without background: {trainepoch_iou_nobackground:.3f}")
                print("\n Training IoUs Client:", cl_idx)
                print("T: Background:", trainepoch_iou_class0)
                print("T: Area 1:", trainepoch_iou_class1)
                print("T: Area 2:", trainepoch_iou_class2)

                print(
                    f"Validating dice loss: {val_epoch_loss:.3f}, Validating accuracy: {val_epoch_acc:.3f},Validating iou Score with background: {valepoch_iou_withbackground:.3f},Validating iou Score without background: {valepoch_iou_nobackground:.3f}")
                print("\n Validating IoUs Client:", cl_idx)
                print("V: Background:", valepoch_iou_class0)
                print("V: Area 1:", valepoch_iou_class1)
                print("V: Area 2:", valepoch_iou_class2)

        C1M1localbest = torch.load(save_F+'./Checkpoints/C1M1_localcheckpoint.pth')
        C1M2localbest = torch.load(save_F+'./Checkpoints/C1M2_localcheckpoint.pth')
        C1M3localbest = torch.load(save_F+'./Checkpoints/C1M3_localcheckpoint.pth')
        C2M1localbest = torch.load(save_F+'./Checkpoints/C2M1_localcheckpoint.pth')
        C2M2localbest = torch.load(save_F+'./Checkpoints/C2M2_localcheckpoint.pth')
        C2M3localbest = torch.load(save_F+'./Checkpoints/C2M3_localcheckpoint.pth')
        C3M1localbest = torch.load(save_F+'./Checkpoints/C3M1_localcheckpoint.pth')
        C3M2localbest = torch.load(save_F+'./Checkpoints/C3M2_localcheckpoint.pth')
        C3M3localbest = torch.load(save_F+'./Checkpoints/C3M3_localcheckpoint.pth')
        C4M1localbest = torch.load(save_F+'./Checkpoints/C4M1_localcheckpoint.pth')
        C4M2localbest = torch.load(save_F+'./Checkpoints/C4M2_localcheckpoint.pth')
        C4M3localbest = torch.load(save_F+'./Checkpoints/C4M3_localcheckpoint.pth')
        C5M1localbest = torch.load(save_F+'./Checkpoints/C5M1_localcheckpoint.pth')
        C5M2localbest = torch.load(save_F+'./Checkpoints/C5M2_localcheckpoint.pth')
        C5M3localbest = torch.load(save_F+'./Checkpoints/C5M3_localcheckpoint.pth')
        C1localViT = torch.load(save_F+'./Checkpoints/C1_local_ViTmodel.pth')
        C2localViT = torch.load(save_F+'./Checkpoints/C2_local_ViTmodel.pth')
        C3localViT = torch.load(save_F+'./Checkpoints/C3_local_ViTmodel.pth')
        C4localViT = torch.load(save_F+'./Checkpoints/C4_local_ViTmodel.pth')
        C5localViT = torch.load(save_F+'./Checkpoints/C5_local_ViTmodel.pth')
        C1reshapenet = torch.load(save_F+'./Checkpoints/C1_local_reshapenet.pth')
        C2reshapenet = torch.load(save_F+'./Checkpoints/C2_local_reshapenet.pth')
        C3reshapenet = torch.load(save_F+'./Checkpoints/C3_local_reshapenet.pth')
        C4reshapenet = torch.load(save_F+'./Checkpoints/C4_local_reshapenet.pth')
        C5reshapenet = torch.load(save_F+'./Checkpoints/C5_local_reshapenet.pth')
        C1localden1 = torch.load(save_F+'./Checkpoints/C1_denoiser1.pth')
        C2localden1 = torch.load(save_F+'./Checkpoints/C2_denoiser1.pth')
        C3localden1 = torch.load(save_F+'./Checkpoints/C3_denoiser1.pth')
        C4localden1 = torch.load(save_F+'./Checkpoints/C4_denoiser1.pth')
        C5localden1 = torch.load(save_F+'./Checkpoints/C5_denoiser1.pth')
        C1localden2 = torch.load(save_F+'./Checkpoints/C1_denoiser2.pth')
        C2localden2 = torch.load(save_F+'./Checkpoints/C2_denoiser2.pth')
        C3localden2 = torch.load(save_F+'./Checkpoints/C3_denoiser2.pth')
        C4localden2 = torch.load(save_F+'./Checkpoints/C4_denoiser2.pth')
        C5localden2 = torch.load(save_F+'./Checkpoints/C5_denoiser2.pth')

        tot_loader = len(train_loader_C1) + len(train_loader_C2) + len(train_loader_C3) + len(train_loader_C4) + len(
            train_loader_C5)
        D1 = len(train_loader_C1) / tot_loader;
        D2 = len(train_loader_C2) / tot_loader;
        D3 = len(train_loader_C3) / tot_loader;
        D4 = len(train_loader_C4) / tot_loader;
        D5 = len(train_loader_C5) / tot_loader;

        # updated parameters
        C1M1localbest.update((x, y * D1) for x, y in C1M1localbest.items())
        C1M2localbest.update((x, y * D1) for x, y in C1M2localbest.items())
        C1M3localbest.update((x, y * D1) for x, y in C1M3localbest.items())
        C2M1localbest.update((x, y * D2) for x, y in C2M1localbest.items())
        C2M2localbest.update((x, y * D2) for x, y in C2M2localbest.items())
        C2M3localbest.update((x, y * D2) for x, y in C2M3localbest.items())
        C3M1localbest.update((x, y * D3) for x, y in C3M1localbest.items())
        C3M2localbest.update((x, y * D3) for x, y in C3M2localbest.items())
        C3M3localbest.update((x, y * D3) for x, y in C3M3localbest.items())
        C4M1localbest.update((x, y * D4) for x, y in C4M1localbest.items())
        C4M2localbest.update((x, y * D4) for x, y in C4M2localbest.items())
        C4M3localbest.update((x, y * D4) for x, y in C4M3localbest.items())
        C5M1localbest.update((x, y * D5) for x, y in C5M1localbest.items())
        C5M2localbest.update((x, y * D5) for x, y in C5M2localbest.items())
        C5M3localbest.update((x, y * D5) for x, y in C5M3localbest.items())
        C1localViT.update((x, y * D1) for x, y in C1localViT.items())
        C2localViT.update((x, y * D2) for x, y in C2localViT.items())
        C3localViT.update((x, y * D3) for x, y in C3localViT.items())
        C4localViT.update((x, y * D4) for x, y in C4localViT.items())
        C5localViT.update((x, y * D5) for x, y in C5localViT.items())
        C1reshapenet.update((x, y * D1) for x, y in C1reshapenet.items())
        C2reshapenet.update((x, y * D2) for x, y in C2reshapenet.items())
        C3reshapenet.update((x, y * D3) for x, y in C3reshapenet.items())
        C4reshapenet.update((x, y * D4) for x, y in C4reshapenet.items())
        C5reshapenet.update((x, y * D5) for x, y in C5reshapenet.items())

        C1localden1.update((x, y * D1) for x, y in C1localden1.items())
        C2localden1.update((x, y * D2) for x, y in C2localden1.items())
        C3localden1.update((x, y * D3) for x, y in C3localden1.items())
        C4localden1.update((x, y * D4) for x, y in C4localden1.items())
        C5localden1.update((x, y * D5) for x, y in C5localden1.items())
        C1localden2.update((x, y * D1) for x, y in C1localden2.items())
        C2localden2.update((x, y * D2) for x, y in C2localden2.items())
        C3localden2.update((x, y * D3) for x, y in C3localden2.items())
        C4localden2.update((x, y * D4) for x, y in C4localden2.items())
        C5localden2.update((x, y * D5) for x, y in C5localden2.items())

        # Model1Averaging
        M1dict = [C1M1localbest, C2M1localbest, C3M1localbest, C4M1localbest, C5M1localbest, ]
        M2dict = [C1M2localbest, C2M2localbest, C3M2localbest, C4M2localbest, C5M2localbest]
        M3dict = [C1M3localbest, C2M3localbest, C3M3localbest, C4M3localbest, C5M3localbest]
        local_weights1.extend(M1dict)
        local_weights2.extend(M2dict)
        local_weights3.extend(M3dict)

        #ViT weights
        ViT_dict = [C1localViT, C2localViT, C3localViT, C4localViT, C5localViT]
        ViT_weights.extend(ViT_dict)

        #reshapenet weights
        reshape_dict = [C1reshapenet, C2reshapenet, C3reshapenet, C4reshapenet, C5reshapenet]
        reshape_weights.extend(reshape_dict)

        #den1 weights
        den1_dict = [C1localden1, C2localden1, C3localden1, C4localden1, C5localden1]
        den1_weights.extend(den1_dict)

        #den2 weights
        den2_dict = [C1localden2, C2localden2, C3localden2, C4localden2, C5localden2]
        den2_weights.extend(den2_dict)


        # averaging parameters
        global_fed_weights1 = fedAvg(local_weights1)
        global_fed_weights2 = fedAvg(local_weights2)
        global_fed_weights3 = fedAvg(local_weights3)
        global_ViT_weights = fedAvg(ViT_weights)
        global_reshape_weights = fedAvg(reshape_weights)
        global_den1_weights = fedAvg(den1_weights)
        global_den2_weights = fedAvg(den2_weights)

        # load the new parameters - FedAvg
        global_model1_fed.load_state_dict(global_fed_weights1)
        global_model2_fed.load_state_dict(global_fed_weights2)
        global_model3_fed.load_state_dict(global_fed_weights3)
        global_ViTmodel.load_state_dict(global_ViT_weights)
        global_reshapenet.load_state_dict(global_reshape_weights)
        global_denoiser1.load_state_dict(global_den1_weights)
        global_denoiser2.load_state_dict(global_den2_weights)
        print("Weights averaged, loaded new weights")

        # ------------------------------------------VALIDATING USING THE GLOBAL MODEL-----------------------------------------------------------------------
        # Validating using the global model
        m1 = max(int(args.frac * args.num_users), 1)
        idxs_users1 = np.random.choice(range(args.num_users), m1, replace=False)
        for idx in idxs_users1:

            cl_idx = idx + 1
            print("Selected client:", cl_idx)
            if cl_idx == 1:
                val_loader = val_loader_C1
                #folder = save_F + "./Saved/global_model/val/client1"
            elif cl_idx == 2:
                val_loader = val_loader_C2
                #folder = save_F + "./Saved/global_model/val/client2"
            elif cl_idx == 3:
                val_loader = val_loader_C3
                #folder = save_F + "./Saved/global_model/val/client3"
            elif cl_idx == 4:
                val_loader = val_loader_C4
                #folder = save_F + "./Saved/global_model/val/client4"
            elif cl_idx == 5:
                val_loader = val_loader_C5
                #folder = save_F + "./Saved/global_model/val/client5"

            best_epoch = 0
            for epoch in range(args.val_global_ep):
                print(f"[INFO]: Epoch {epoch + 1} of {args.val_global_ep}")
                print("Client", cl_idx, " validating.........")
                if cl_idx == 1:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1,g_valepoch_iou_class2 = eval(
                        val_loader, global_model1_fed, global_model2_fed, global_model3_fed,global_ViTmodel,global_reshapenet,global_denoiser1,global_denoiser2, global_dn_pred1,global_dn_pred2,loss_fn)
                    client1_g_val_acc.append(g_val_epoch_acc)
                    client1_g_val_loss.append(g_val_epoch_loss)
                    client1_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client1_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 2:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1,g_valepoch_iou_class2= eval(
                        val_loader, global_model1_fed, global_model2_fed, global_model3_fed,global_ViTmodel,global_reshapenet,global_denoiser1,global_denoiser2, global_dn_pred1,global_dn_pred2,loss_fn)
                    client2_g_val_acc.append(g_val_epoch_acc)
                    client2_g_val_loss.append(g_val_epoch_loss)
                    client2_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client2_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 3:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1,g_valepoch_iou_class2 = eval(
                        val_loader, global_model1_fed, global_model2_fed, global_model3_fed,global_ViTmodel,global_reshapenet,global_denoiser1,global_denoiser2, global_dn_pred1,global_dn_pred2,loss_fn)
                    client3_g_val_acc.append(g_val_epoch_acc)
                    client3_g_val_loss.append(g_val_epoch_loss)
                    client3_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client3_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 4:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1,g_valepoch_iou_class2 = eval(
                        val_loader, global_model1_fed, global_model2_fed, global_model3_fed,global_ViTmodel,global_reshapenet,global_denoiser1,global_denoiser2, global_dn_pred1,global_dn_pred2,loss_fn)
                    client4_g_val_acc.append(g_val_epoch_acc)
                    client4_g_val_loss.append(g_val_epoch_loss)
                    client4_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client4_g_val_iounoback.append(g_val_epoch_iounoback)
                if cl_idx == 5:
                    g_val_epoch_loss, g_val_epoch_acc, g_val_epoch_iouwithback, g_val_epoch_iounoback, g_valepoch_iou_class0, g_valepoch_iou_class1,g_valepoch_iou_class2= eval(
                        val_loader, global_model1_fed, global_model2_fed, global_model3_fed,global_ViTmodel,global_reshapenet,global_denoiser1,global_denoiser2, global_dn_pred1,global_dn_pred2,loss_fn)
                    client5_g_val_acc.append(g_val_epoch_acc)
                    client5_g_val_loss.append(g_val_epoch_loss)
                    client5_g_val_iouwithback.append(g_val_epoch_iouwithback)
                    client5_g_val_iounoback.append(g_val_epoch_iounoback)

                print(
                    f"Global Validating dice loss: {g_val_epoch_loss:.3f}, Global Validating accuracy: {g_val_epoch_acc:.3f},Global Validating iou Score with background: {g_val_epoch_iouwithback:.3f},Global Validating iou Score without background: {g_val_epoch_iounoback:.3f}")
                print("\n Global Validating IoUs Client:", cl_idx)
                print("GV: Background:", g_valepoch_iou_class0)
                print("GV: Area 1:", g_valepoch_iou_class1)
                print("GV: Area 2:", g_valepoch_iou_class2)


        tot_gloss = client1_g_val_loss[-1] + client2_g_val_loss[-1] + client3_g_val_loss[-1] + client4_g_val_loss[-1] + \
                    client5_g_val_loss[-1]
        avg_g_val_loss = tot_gloss / 5;
        print("avg g val loss:", avg_g_val_loss)
        if least_lossg > avg_g_val_loss:
            least_lossg = avg_g_val_loss
            torch.save(global_model1_fed.state_dict(), save_F + './Checkpoints/M1_globalcheckpoint.pth')
            torch.save(global_model2_fed.state_dict(), save_F + './Checkpoints/M2_globalcheckpoint.pth')
            torch.save(global_model3_fed.state_dict(), save_F + './Checkpoints/M3_globalcheckpoint.pth')
            torch.save(global_ViTmodel.state_dict(), save_F + './Checkpoints/globalViT.pth')
            torch.save(global_reshapenet.state_dict(), save_F + './Checkpoints/global_reshapenet.pth')
            torch.save(global_denoiser1.state_dict(), save_F + './Checkpoints/global_denoiser1.pth')
            torch.save(global_denoiser2.state_dict(), save_F + './Checkpoints/global_denoiser2.pth')
            print('Global best model saved')
            print('-' * 50)

        # ------------------------------------------TESTING USING THE GLOBAL MODEL-----------------------------------------------------------------------

        test_folder = save_F + "/testingsaved"
        M1_test = copy.deepcopy(global_model1_fed)
        M2_test = copy.deepcopy(global_model2_fed)
        M3_test = copy.deepcopy(global_model3_fed)
        ViT_test = copy.deepcopy(global_ViTmodel)
        test_denoiser1 = copy.deepcopy(global_denoiser1)
        test_denoiser2 = copy.deepcopy(global_denoiser2)
        test_reshapenet = copy.deepcopy(global_reshapenet)

        M1_test.load_state_dict(torch.load(save_F + './Checkpoints/M1_globalcheckpoint.pth'))
        M2_test.load_state_dict(torch.load(save_F + './Checkpoints/M2_globalcheckpoint.pth'))
        M3_test.load_state_dict(torch.load(save_F + './Checkpoints/M3_globalcheckpoint.pth'))
        ViT_test.load_state_dict(torch.load(save_F + './Checkpoints/globalViT.pth'))
        test_reshapenet.load_state_dict(torch.load(save_F + './Checkpoints/global_reshapenet.pth'))
        test_denoiser1.load_state_dict(torch.load(save_F + './Checkpoints/global_denoiser1.pth'))
        test_denoiser2.load_state_dict(torch.load(save_F + './Checkpoints/global_denoiser2.pth'))

        for epoch in range(args.val_global_ep):
            print("Global testing.........")
            test_epoch_loss, test_epoch_seg_loss, test_epoch_recon1_loss, test_epoch_recon2_loss, test_epoch_acc, test_epoch_accwithback, test_epoch_accnoback, global_dn_pred1, global_dn_pred2 = test(
                test_loader, M1_test, M2_test, M3_test, ViT_test, test_reshapenet, test_denoiser1, test_denoiser2,
                loss_fn, test_folder)
            print('\n')
            print(
                f"Testing dice loss: {test_epoch_loss:.3f}, Testing accuracy: {test_epoch_acc:.3f},Testing iou Score with background: {test_epoch_accwithback:.3f},Testing iou Score without background: {test_epoch_accnoback:.3f}")
            test_Acc.append(test_epoch_acc)
            test_Iou_withback.append(test_epoch_accwithback)
            test_Iou_noback.append(test_epoch_accnoback)
            test_Loss.append(test_epoch_loss)
            test_seg_loss.append(test_epoch_seg_loss)
            test_recon1_loss.append(test_epoch_recon1_loss)
            test_recon2_loss.append(test_epoch_recon2_loss)

        # -------------------------------------------------PLOTTING RESULTS-----------------------------------------------------------------------

        alltest_acc, alltest_iouwithback, alltest_iounoback, alltest_loss, alltest_segloss, alltest_rec1loss, alltest_rec2loss = [], [], [], [], [], [], []
        alltest_acc.append(test_Acc)
        alltest_loss.append(test_Loss)
        alltest_segloss.append(test_seg_loss)
        alltest_rec1loss.append(test_recon1_loss)
        alltest_rec2loss.append(test_recon2_loss)
        alltest_iouwithback.append(test_Iou_withback)
        alltest_iounoback.append(test_Iou_noback)

        alltest_acc = pd.DataFrame(alltest_acc)
        alltest_loss = pd.DataFrame(alltest_loss)
        alltest_segloss = pd.DataFrame(alltest_segloss)
        alltest_rec1loss = pd.DataFrame(alltest_rec1loss)
        alltest_rec2loss = pd.DataFrame(alltest_rec2loss)
        alltest_iouwithback = pd.DataFrame(alltest_iouwithback)
        alltest_iounoback = pd.DataFrame(alltest_iounoback)

        alltest_acc.to_csv(save_F + './Outputs/alltest_acc.csv')
        alltest_loss.to_csv(save_F + './Outputs/alltest_loss.csv')
        alltest_segloss.to_csv(save_F + './Outputs/alltest_segloss.csv')
        alltest_rec1loss.to_csv(save_F + './Outputs/alltest_rec1loss.csv')
        alltest_rec2loss.to_csv(save_F + './Outputs/alltest_rec2loss.csv')
        alltest_iouwithback.to_csv(save_F + './Outputs/alltest_iouwithback.csv')
        alltest_iounoback.to_csv(save_F + './Outputs/alltest_iouwithoutback.csv')

    print('TRAINING COMPLETE')
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))


if __name__ == "__main__":
    main()

