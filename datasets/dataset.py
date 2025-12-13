import os
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# =========================
# Default Transforms
# =========================
def get_default_transform(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])


# =========================
# K-SAM Dataset
# =========================
class KSAMDataset(Dataset):
    """
    Dataset structure assumption:

    DATA_ROOT/
      ├── train/
      │    ├── subject_id/
      │    │     ├── xxxx_yyyy_age_xxxx_F_sam128.jpg
      │    │     ├── xxxx_yyyy_age_xxxx_F_sam128.npy   (optional landmarks)
      ├── validation/
      └── test/

    Returns:
      x_src   : source image
      age_src : source age (float tensor)
      age_tgt : target age (float tensor)
      x_tgt   : (optional, None for unpaired)
      lm_src  : source landmarks (68×2) or zeros
    """

    def __init__(
        self,
        data_root,
        split="train",
        img_size=128,
        paired=False,
        use_landmarks=True,
        transform=None,
        age_min=0.0,
        age_max=80.0
    ):
        self.data_root = data_root
        self.split = split
        self.paired = paired
        self.use_landmarks = use_landmarks
        self.age_min = age_min
        self.age_max = age_max

        self.transform = transform or get_default_transform(img_size)

        self.records = self._build_index()

    def _build_index(self):
        split_dir = os.path.join(self.data_root, self.split)
        records = []

        for sid in os.listdir(split_dir):
            sid_dir = os.path.join(split_dir, sid)
            if not os.path.isdir(sid_dir):
                continue

            img_paths = sorted(
                glob.glob(os.path.join(sid_dir, "*_sam128.jpg"))
            )

            for img_path in img_paths:
                fname = os.path.basename(img_path)
                parts = fname.split("_")

                # Example: 0013_1972_06_00000001_F_sam128.jpg
                age = float(parts[2])

                lm_path = img_path.replace(".jpg", ".npy")

                records.append({
                    "subject_id": sid,
                    "img_path": img_path,
                    "lm_path": lm_path,
                    "age": age
                })

        return records

    def __len__(self):
        return len(self.records)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def _load_landmarks(self, path):
        if self.use_landmarks and os.path.exists(path):
            lm = np.load(path).astype(np.float32)
            return torch.from_numpy(lm)
        else:
            return torch.zeros((68, 2), dtype=torch.float32)

    def __getitem__(self, idx):
        src = self.records[idx]

        x_src = self._load_image(src["img_path"])
        lm_src = self._load_landmarks(src["lm_path"])

        age_src = torch.tensor([src["age"]], dtype=torch.float32)

        # -------- target sampling --------
        if self.paired:
            # same subject, different image
            same_sid = [
                r for r in self.records
                if r["subject_id"] == src["subject_id"]
                and r["img_path"] != src["img_path"]
            ]
            tgt = random.choice(same_sid) if same_sid else src
        else:
            tgt = random.choice(self.records)

        age_tgt = torch.tensor([tgt["age"]], dtype=torch.float32)

        # We do NOT load x_tgt (unpaired training)
        return x_src, age_src, age_tgt, None, lm_src


# =========================
# Dataloader helper
# =========================
def build_dataloader(
    data_root,
    split,
    batch_size,
    shuffle=True,
    num_workers=4,
    **dataset_kwargs
):
    dataset = KSAMDataset(
        data_root=data_root,
        split=split,
        **dataset_kwargs
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    )
    return loader
