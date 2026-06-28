import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class FiLMGenerator(nn.Module):
    def __init__(self, meta_dim, hidden_dim, num_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*num_channels),
        )

    def forward(self, features, meta):
        gamma, beta = self.net(meta).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return features * (1 + gamma) + beta # RESIDUAL FiLM
    

class SolarUNet(nn.Module):
    DECODER_CHANNELS = 16
    
    def __init__(self, cfg):
        super().__init__()

        self.base = smp.Unet(
            encoder_name=cfg.model.encoder,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=3,
            classes=cfg.model.out_channels,
            activation=None,
        )
        self.film = FiLMGenerator(cfg.model.meta_dim,
                                  cfg.model.film_hidden_dim,
                                  self.DECODER_CHANNELS)
    
    def forward(self, x, meta):
        features = self.base.encoder(x)
        decoder_out = self.base.decoder(features)       # (B, 16, H, W)
        conditioned = self.film(decoder_out, meta)
        return self.base.segmentation_head(conditioned) # (B, 1, H, W)