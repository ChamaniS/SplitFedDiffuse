# splitfed_gms_ham10k.py
import os
import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
import albumentations as A
from models.gms_encoder import GMSEncoder
from models.latent_mapping import LatentMappingNetwork
from models.gms_decoder import GMSDecoder
from utils.dataloader import HAMDataset
from agg.Fed_Avg import fedAvg
from sklearn.metrics import jaccard_score

# Config
ROUNDS = 10
EPOCHS = 12
BATCH_SIZE =1
LR = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

train_transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    #A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
    ToTensorV2()
])


DataF = "/lustre06/project/6008975/csj5/highlynonuniform/"
save_F = "/lustre06/project/6008975/csj5/narvalenv/Saved2/baseline/Fed_Avg/"

client_paths = {
    "client1": (DataF + "/client_1/train_img/", DataF + "/client_1/train_mask/"),
    "client2": (DataF + "/client_2/train_img/", DataF + "/client_2/train_mask/"),
    "client3": (DataF + "/client_3/train_img/", DataF + "/client_3/train_mask/"),
    "client4": (DataF + "/client_4/train_img/", DataF + "/client_4/train_mask/"),
    "client5": (DataF + "/client_5/train_img/", DataF + "/client_5/train_mask/")
}

TEST_IMG_DIR = DataF + "/mutual_img/"
TEST_MASK_DIR = DataF + "/mutual_mask/"

# Model init
RESNET_WEIGHTS_PATH = "/lustre06/project/6008975/csj5/narvalenv/GMS/pretrained/resnet50-0676ba61.pth" 

# Model init
def init_model():
    encoder = GMSEncoder(pretrained=True).to(device)
    latent_mapper = LatentMappingNetwork().to(device)
    decoder = GMSDecoder().to(device)
    return encoder, latent_mapper, decoder

# Local training
def local_train(encoder, mapper, decoder, loader, epochs):
    encoder.eval()  # frozen
    mapper.train()
    decoder.eval()  # frozen
    optimizer = torch.optim.Adam(mapper.parameters(), lr=LR)

    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        for img, mask in loop:
            img, mask = img.to(device), mask.to(device)
            with torch.no_grad():
                feats = encoder(img)
            mapped_feats = mapper(feats)
            output = decoder(mapped_feats)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=output.shape[2:], mode="bilinear", align_corners=False)
            loss = F.binary_cross_entropy_with_logits(output, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
    return mapper.state_dict()

# Evaluation function
def evaluate(encoder, mapper, decoder, loader):
    encoder.eval()
    mapper.eval()
    decoder.eval()
    total_iou = 0
    total_loss = 0
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            feats = encoder(img)
            mapped_feats = mapper(feats)
            output = decoder(mapped_feats)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=output.shape[2:], mode="bilinear", align_corners=False)
            loss = F.binary_cross_entropy_with_logits(output, mask)
            pred = (torch.sigmoid(output) > 0.5).long().squeeze(1)
            true_bin = (mask > 0.5).long().view(-1).cpu().numpy()
            pred_bin = pred.long().view(-1).cpu().numpy()
            iou = jaccard_score(true_bin, pred_bin, average='binary')

            total_iou += iou
            total_loss += loss.item()
    return total_loss / len(loader), total_iou / len(loader)

# Main loop
if __name__ == "__main__":
    global_encoder, global_mapper, global_decoder = init_model()

    # Create dataloaders for each client
    clients = []
    for name, (img_dir, mask_dir) in client_paths.items():
        dataset = HAMDataset(img_dir, mask_dir, transform=train_transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        clients.append((name, loader))

    # Test loader
    test_dataset = HAMDataset(TEST_IMG_DIR, TEST_MASK_DIR, transform=train_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for rnd in range(ROUNDS):
        print(f"\n--- Round {rnd+1} ---")
        mapper_weights = []

        for client_id, (client_name, loader) in enumerate(clients):
            print(f"Client {client_name} training...")
            _, mapper, _ = init_model()
            mapper.load_state_dict(global_mapper.state_dict())
            trained_mapper = local_train(global_encoder, mapper, global_decoder, loader, EPOCHS)
            mapper_weights.append(trained_mapper)

            print(f"Evaluating {client_name} locally...")
            loss, iou = evaluate(global_encoder, mapper, global_decoder, loader)
            print(f"[Train Eval] {client_name} - Loss: {loss:.4f}, IoU: {iou:.4f}")

        avg_mapper = fedAvg(mapper_weights)
        global_mapper.load_state_dict(avg_mapper)

        print("\nGlobal model evaluation:")
        for client_name, loader in clients:
            loss, iou = evaluate(global_encoder, global_mapper, global_decoder, loader)
            print(f"[Global Eval] {client_name} - Loss: {loss:.4f}, IoU: {iou:.4f}")

        print("\nTesting on shared test set:")
        test_loss, test_iou = evaluate(global_encoder, global_mapper, global_decoder, test_loader)
        print(f"[Test Eval] Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}")

    torch.save(global_encoder.state_dict(), save_F + "gms_encoder_ham10k.pth")
    torch.save(global_mapper.state_dict(), save_F + "gms_mapper_ham10k.pth")
    torch.save(global_decoder.state_dict(), save_F + "gms_decoder_ham10k.pth")
    print("Training complete. Models saved.")