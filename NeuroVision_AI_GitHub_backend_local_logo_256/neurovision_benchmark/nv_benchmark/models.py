"""CSI-only, camera-only and late-fusion comparison with a shared binary task."""
import torch
from torch import nn


class CSIEncoder(nn.Module):
    def __init__(self, channels: int = 12, dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class PerceptionClassifier(nn.Module):
    def __init__(self, mode: str, channels: int = 12, visual_dim: int = 8, n_outputs: int = 1):
        super().__init__()
        assert mode in {"csi", "camera", "fusion"}
        self.mode = mode
        if mode != "camera":
            self.csi = CSIEncoder(channels)
        if mode != "csi":
            self.camera = nn.Sequential(
                nn.Linear(visual_dim, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.ReLU()
            )
        dim = {"csi": 64, "camera": 32, "fusion": 96}[mode]
        self.head = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, n_outputs))

    def forward(self, csi, camera):
        if self.mode == "csi":
            z = self.csi(csi)
        elif self.mode == "camera":
            z = self.camera(camera)
        else:
            z = torch.cat([self.csi(csi), self.camera(camera)], dim=1)
        out = self.head(z)
        return out.squeeze(1) if out.shape[1] == 1 else out
