import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]
        

class BaselineV2_Transformer(nn.Module):
    def __init__(self, feature_dim=1024, n_heads=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.pos_enc = PositionalEncoding(feature_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # MLP HEAD per 1 output
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, x):
        """
        x: (B=1, T, D)
        returns: (B=1) probabilità per l'intero step
        """
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, T, D)

        # mean pooling sui sub-segmenti
        x = x.mean(dim=1)  # (B, D)

        logits = self.mlp(x).squeeze(-1)  # (B,)
        probs = torch.sigmoid(logits)
        return probs, logits