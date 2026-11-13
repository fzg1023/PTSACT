"""Original dual-input network definition; training paths excluded."""
import torch
from torch import nn
from nn_prior import AttentionUNet


class DualInputResidual(nn.Module):
    def __init__(self, base_ch=24, dropout=0.1):
        super().__init__()
        self.tof_branch = nn.Sequential(nn.Linear(120, 256), nn.ReLU(), nn.Linear(256, 1024))
        self.refiner = AttentionUNet(in_channels=2, base_ch=base_ch, dropout_p=dropout, use_attention=True)

    def forward(self, initial, tof):
        features = self.tof_branch(tof).reshape(-1, 1, 32, 32)
        return self.refiner(torch.cat((initial, features), dim=1))
