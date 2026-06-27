import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROOF_TYPES = ["tile", "tin", "flat"]
CONNECTION_TYPES = ["single_phase", "three_phase"]

def encode_meta(row: pd.Series) -> torch.Tensor:
    vec = torch.zeros(8)
    if row["roof_type"] in ROOF_TYPES:
        vec[ROOF_TYPES.index(row["roof_type"])] = 1.0
    if row["connection_type"] in CONNECTION_TYPES:
        vec[3 + CONNECTION_TYPES.index(row["connection_type"])] = 1.0
    vec[5] = float(row["num_panels"]) / 70.0
    vec[6] = float(row["angle"]) / 90
    vec[7] = float(row["num_strings"]) / 10.0
    return vec

def get_transforms(img_size, train=True):
    h, w = img_size
    if train:
        return A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.2),
            A.ElasticTransform(alpha=30, sigma=5, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
            A.RandomShadow(p=0.2),
            A.Resize(h, w),
            A.Normalize(),
            ToTensorV2(),
        ])
    return A.Compose([A.Resize(h, w), A.Normalize(), ToTensorV2()])

class SolarDataset(Dataset):
    def __init__(self, meta_df, image_dir, mask_dir, img_size, train=True):
        self.meta = meta_df.reset_index(drop = True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = get_transforms(img_size, train)

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_filename"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.mask_dir, row["mask_filename"])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot open mask: {mask_path}")
        mask = (mask > 127).astype(np.uint8)
        out = self.transform(image=img, mask=mask)
        image_t = out["image"]
        mask_t = out["mask"].unsqueeze(0).float()
        return image_t, mask_t, encode_meta(row)

def make_datasets(cfg):
    meta = pd.read_csv(cfg.data.metadata_csv).sample(frac=1, random_state=42)
    n_val = max(1, int(len(meta) * cfg.data.val_split))
    train_ds = SolarDataset(meta.iloc[n_val:], cfg.data.image_dir, cfg.data.mask_dir, 
                            cfg.data.img_size, train=True)
    val_ds = SolarDataset(meta.iloc[:n_val], cfg.data.image_dir, cfg.data.mask_dir,
                          cfg.data.img_size, train=False)
    return train_ds, val_ds
    
