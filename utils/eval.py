import torch
import numpy as np
from tqdm import tqdm

from torchvision import transforms
from PIL import Image

# Optional: LPIPS / FID imports
import lpips
from pytorch_fid.fid_score import calculate_fid_given_paths


# =========================
# Image utils
# =========================
def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def load_image(path, img_size=128):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0)


# =========================
# Age MAE
# =========================
@torch.no_grad()
def compute_age_mae(generator, age_estimator, csv_rows, device):
    errs = []

    for row in tqdm(csv_rows, desc="Age MAE"):
        x = load_image(row["input_path"]).to(device)
        age_tgt = torch.tensor([row["target_age"]], device=device)

        x_fake, _ = generator(x, age_tgt, age_tgt)
        age_pred = age_estimator(x_fake)

        errs.append(torch.abs(age_pred - age_tgt).item())

    return float(np.mean(errs))


# =========================
# LPIPS
# =========================
@torch.no_grad()
def compute_lpips(generator, csv_rows, device):
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    scores = []

    for row in tqdm(csv_rows, desc="LPIPS"):
        x = load_image(row["input_path"]).to(device)
        y = load_image(row["output_path"]).to(device)

        s = lpips_fn(x, y).mean().item()
        scores.append(s)

    return float(np.mean(scores))


# =========================
# FID
# =========================
def compute_fid(real_dir, fake_dir, device):
    return calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=32,
        device=device,
        dims=2048
    )
