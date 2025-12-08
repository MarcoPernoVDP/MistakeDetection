import torch
import torch.nn as nn

class BaselineV1_MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 512, p: float = 0.5):
        super().__init__()

        self.norm = nn.LayerNorm(in_features)      # ★ Normalizzazione cruciale

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),

            nn.Linear(hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p),

            nn.Linear(256, 1)                      # Output logit
        )

    def forward(self, x):
        x = self.norm(x)           # stabilizza feature Omnivore/SlowFast
        x = self.mlp(x)
        return x
