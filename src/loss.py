import torch
import torch.nn as nn


class RGBPanelLoss(nn.Module):
    def __init__(self, panel_pixel_weight: float = 10.0):
        super().__init__()
        self.w = panel_pixel_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        # panel pixels: where any channel of target is active
        panel_mask = (targets.sum(dim=1, keepdim=True) > 0.02).float()
        weights = 1.0 + (self.w - 1.0) * panel_mask
        return (weights * (pred - targets).abs()).mean()
