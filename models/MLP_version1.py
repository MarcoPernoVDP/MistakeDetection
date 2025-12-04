import torch.nn as nn

class MLP_version1(nn.Module):
    def __init__(self, in_features: int, p: float = 0.5):
        super().__init__()

        self.fc1 = nn.Linear(in_features, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)       # Dropout layer con probabilità p
        self.fc2 = nn.Linear(256, 1)       # Output logit (senza sigmoid)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)                # Applica Dropout solo in TRAIN
        x = self.fc2(x)                    # Output logit
        return x                           # no Sigmoid qui
