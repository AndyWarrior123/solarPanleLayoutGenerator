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


def encode_meta(row: "pd.Series | dict") -> torch.Tensor:
    vec = torch.zeros(8)
    if row["roof_type"] in ROOF_TYPES:
        vec[ROOF_TYPES.index(row["roof_type"])] = 1.0
    if row["connection_type"] in CONNECTION_TYPES:
        vec[3 + CONNECTION_TYPES.index(row["connection_type"])] = 1.0
    vec[5] = float(row["num_panels"]) / 70.0
    vec[6] = float(row["angle"]) / 90
    vec[7] = float(row["num_strings"]) / 10.0
    return vec


def _spatial_transforms(img_size, train: bool):
    """Transforms applied to BOTH the roof image and the panel-cut target."""
    h, w = img_size
    if train:
        return A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.CoarseDropout(num_holes_range=(1, 4),
                            hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.2),
            A.ElasticTransform(alpha=30, sigma=5, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
            A.RandomShadow(p=0.2),
            A.Resize(h, w),
        ], additional_targets={"target": "image"})
    return A.Compose([A.Resize(h, w)], additional_targets={"target": "image"})


def _image_transforms(train: bool):
    """Colour / noise transforms applied to the ROOF IMAGE only."""
    if train:
        return A.Compose([
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                 val_shift_limit=10, p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    return A.Compose([A.Normalize(), ToTensorV2()])


class SolarDataset(Dataset):
    def __init__(self, meta_df, image_dir, mask_dir, img_size, train=True):
        self.meta         = meta_df.reset_index(drop=True)
        self.image_dir    = image_dir
        self.mask_dir     = mask_dir
        self.spatial_tfm  = _spatial_transforms(img_size, train)
        self.image_tfm    = _image_transforms(train)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]

        # ── Input: clean roof image (BGR → RGB) ──────────────────────────────
        img_path = os.path.join(self.image_dir, row["image_filename"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── Target: RGB panel cut (BGR → RGB) ────────────────────────────────
        # The file is a 3-channel colour PNG produced by prepare_single --layout-cut.
        # Pixels inside detected panels carry the actual panel colour; elsewhere = 0.
        target_path = os.path.join(self.mask_dir, row["mask_filename"])
        target = cv2.imread(target_path)
        if target is None:
            raise FileNotFoundError(f"Cannot open target: {target_path}")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # ── Spatial transforms: applied to both roof image and target ─────────
        out    = self.spatial_tfm(image=img, target=target)
        img    = out["image"]
        target = out["target"]

        # ── Colour transforms + normalise: roof image only ───────────────────
        image_t = self.image_tfm(image=img)["image"]                        # (3, H, W) normalised

        # Target: scale to [0, 1]; shape (3, H, W)
        target_t = torch.from_numpy(target).permute(2, 0, 1).float() / 255.0

        return image_t, target_t, encode_meta(row)


def make_datasets(cfg):
    meta  = pd.read_csv(cfg.data.metadata_csv).sample(frac=1, random_state=42)
    n_val = max(1, int(len(meta) * cfg.data.val_split))
    train_ds = SolarDataset(meta.iloc[n_val:], cfg.data.image_dir, cfg.data.mask_dir,
                            cfg.data.img_size, train=True)
    val_ds   = SolarDataset(meta.iloc[:n_val], cfg.data.image_dir, cfg.data.mask_dir,
                            cfg.data.img_size, train=False)
    return train_ds, val_ds
