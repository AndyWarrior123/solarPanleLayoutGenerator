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
    # MSE between predicted panel area and expected count from metadata

    def forward(self, logits, meta, pixels_per_panel=800):
        pred_count = torch.sigmoid(logits).sum(dim=(1, 2, 3)) / pixels_per_panel
        target_count = meta[:, 5] * 30.0
        return F.mse_loss(pred_count, target_count)
    

class SetbackLoss(nn.Module):
    def __init__(self, margin=20):
        super().__init__()
        self.m = margin
    
    def forward(self, logits):
        p = torch.sigmoid(logits)
        m = self.m
        border = torch.cat([p[:,:,:m,:], p[:,:,-m:,:],
                            p[:,:,m:-m, :m], p[:, :, m:-m, -m:]], dim=2)
        return border.mean()
    
class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.count = CountLoss()
        self.setback = SetbackLoss()
        self.w_dice = cfg.training.dice_weight
        self.w_count = cfg.training.count_loss_weight
        self.w_setback = cfg.training.setback_loss_weight

    def forward(self, logits, masks, meta):
        seg = self.bce(logits, masks) + self.dice(logits, masks)
        return (self.w_dice * seg
                + self.w_count * self.count(logits, meta)
                + self.w_setback * self.setback(logits))