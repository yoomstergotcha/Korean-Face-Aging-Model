import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.sam_lite import SAMLiteGenerator
from datasets.kor_aging_dataset import KorAgingDataset
from scripts.train_sam_lite import criterion_recon, criterion_age, perceptual_loss, identity_loss

def compute_losses(checkpoint, data_dir, labels_csv, img_size=128, batch_size=8):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 불러오기
    model = SAMLiteGenerator(latent_dim=256, img_size=img_size).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    dataset = KorAgingDataset(
        img_dir=data_dir,
        labels_csv=labels_csv,
        img_size=img_size,
        split="train"
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0
    total_recon = 0
    total_lpips = 0
    total_id = 0
    total_age = 0

    with torch.no_grad():
        for imgs, ages in tqdm(loader):
            imgs = imgs.to(device)
            ages = ages.to(device).float()

            outputs, age_pred = model(imgs, ages)

            loss_recon = criterion_recon(outputs, imgs)
            loss_lpips = perceptual_loss(outputs, imgs)
            loss_id = identity_loss(outputs, imgs)
            loss_age = criterion_age(age_pred.squeeze(), ages)

            loss = (
                1.0 * loss_recon +
                0.2 * loss_lpips +
                0.1 * loss_id +
                0.5 * loss_age
            )

            total_loss += loss.item() * imgs.size(0)
            total_recon += loss_recon.item() * imgs.size(0)
            total_lpips += loss_lpips.item() * imgs.size(0)
            total_id += loss_id.item() * imgs.size(0)
            total_age += loss_age.item() * imgs.size(0)

    N = len(dataset)

    print("=== Loss Summary ===")
    print(f"Total Loss    : {total_loss / N:.4f}")
    print(f"Recon Loss    : {total_recon / N:.4f}")
    print(f"LPIPS Loss    : {total_lpips / N:.4f}")
    print(f"ID Loss       : {total_id / N:.4f}")
    print(f"Age Loss      : {total_age / N:.4f}")


if __name__ == "__main__":
    compute_losses(
        checkpoint="./checkpoints/sam_lite_kor.pt",
        data_dir="./data/train",
        labels_csv="./data/labels.csv",
        img_size=128,
        batch_size=8
    )
