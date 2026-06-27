import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def forward(self, logits, targets, smooth = 1.0):
        p = torch.sigmoid(logits)
        inter = (p*targets).sum(dim=(2, 3))
        union = p.sum(dim=(2,3)) + targets.sum(dim = (2, 3))
        return 1 - ((2 * inter + smooth) / (union + smooth)).mean()
    

class CountLoss(nn.Module):
    # MSE between predicted panel fraction and target fraction.
    # Both sides normalised to [0, 1] so this loss stays on the same scale
    # as BCE/Dice (≈0-2) instead of exploding to thousands at init.
    # pixels_per_panel tuned for 512x512 input (original ~700x1424 ≈ 2200px/panel,
    # scaled down by 512²/700/1424 ≈ 0.27 → ~600px/panel).

    def forward(self, logits, meta, pixels_per_panel=600):
        pred_count_norm = torch.sigmoid(logits).sum(dim=(1, 2, 3)) / (pixels_per_panel * 70.0)
        return F.mse_loss(pred_count_norm, meta[:, 5])  # meta[:,5] is already num_panels/70
    

class SetbackLoss(nn.Module):
    def __init__(self, margin=20):
        super().__init__()
        self.m = margin
    
    def forward(self, logits):
        p = torch.sigmoid(logits)
        m = self.m
        top    = p[:, :, :m,    :   ].mean()
        bottom = p[:, :, -m:,   :   ].mean()
        left   = p[:, :, m:-m,  :m  ].mean()
        right  = p[:, :, m:-m,  -m: ].mean()
        return (top + bottom + left + right) / 4
    
class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ~95% background, ~5% panels → upweight panel pixels so BCE doesn't
        # reward predicting all-background. 10x balances without over-predicting blobs.
        self.register_buffer("pos_w", torch.tensor([10.0]))
        self.bce = None  # built lazily in forward once device is known
        self.dice = DiceLoss()
        self.count = CountLoss()
        self.setback = SetbackLoss()
        self.w_dice = cfg.training.dice_weight
        self.w_count = cfg.training.count_loss_weight
        self.w_setback = cfg.training.setback_loss_weight

    def forward(self, logits, masks, meta):
        bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_w.to(logits.device))
        seg = bce(logits, masks) + self.dice(logits, masks)
        return (self.w_dice * seg
                + self.w_count * self.count(logits, meta)
                + self.w_setback * self.setback(logits))