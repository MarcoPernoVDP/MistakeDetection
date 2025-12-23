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
    def __init__(self, feature_dim=1024, n_heads=8, num_layers=2, dropout=0.5):
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

    def forward(self, x, attention_mask=None):
        """
        x: (B, T, D) - input features (può essere padded)
        attention_mask: (B, T) - boolean tensor, True per token reali, False per padding
        returns: (B,) probabilità per l'intero video
        """
        x = self.pos_enc(x)
        
        # Converti attention_mask per il transformer
        # TransformerEncoder usa src_key_padding_mask dove True = IGNORA
        # Il nostro attention_mask ha True = REALE, quindi invertiamo
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask  # Inverti: True dove c'è padding
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, D)

        # Mean pooling solo sui token reali (non sul padding)
        if attention_mask is not None:
            # Maschera i token di padding prima del pooling
            mask_expanded = attention_mask.unsqueeze(-1)  # (B, T, 1)
            x_masked = x * mask_expanded  # (B, T, D)
            # Somma e dividi per il numero di token reali
            sum_embeddings = x_masked.sum(dim=1)  # (B, D)
            num_real_tokens = attention_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
            x = sum_embeddings / num_real_tokens.clamp(min=1.0)  # (B, D)
        else:
            x = x.mean(dim=1)  # (B, D)

        logits = self.mlp(x).squeeze(-1)  # (B,)
        probs = torch.sigmoid(logits)
        return probs, logits