import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class FusionCNN(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.enc_roi = Encoder(bands)
        self.enc_sat = Encoder(bands)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, bands, 3, padding=1)
        )

    def forward(self, roi, sat):
        f_roi = self.enc_roi(roi)
        f_sat = self.enc_sat(sat)
        fused = torch.cat([f_roi, f_sat], dim=1)
        return self.dec(fused)
