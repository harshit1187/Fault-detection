import numpy as np, torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, n_ch=4, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 8, 15, stride=4, padding=7), nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, 16, 9, stride=4, padding=4), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, n_cls),
        )

    def forward(self, x):
        return self.net(x)

def prep(X, stats=None):
    """X: (N, 1024, 4) -> (N, 4, 1024), normalized.
    Accel channels: per-window mean removed (kills gravity).
    Current channel: kept as is, then standardized with TRAIN stats."""
    X = X.astype(np.float32).copy()
    X[:, :, :3] -= X[:, :, :3].mean(axis=1, keepdims=True)
    X = X.transpose(0, 2, 1)
    if stats is None:
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True) + 1e-8
        stats = (mean, std)
    mean, std = stats
    return ((X - mean) / std).astype(np.float32), stats