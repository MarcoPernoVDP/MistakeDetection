import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineV3_LSTM(nn.Module):
    """
    Nuova Baseline (V3) basata su LSTM puro.
    """
    def __init__(self, feature_dim=1024, hidden_size=1024, num_layers=2, dropout=0.5):
        """
        Args:
            feature_dim (int): Dimensione delle feature di input (D).
            hidden_size (int): Dimensione dell'hidden state dell'LSTM.
                               (Uguale a feature_dim per coerenza con l'MLP del Transformer)
            num_layers (int): Numero di strati LSTM impilati.
            dropout (float): Dropout applicato tra gli strati LSTM.
        """
        super().__init__()

        # Modulo LSTM
        # batch_first=True si aspetta input (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # MLP HEAD (Testa di Classificazione)
        # Identica a quella del Transformer, riceve un vettore di dimensione hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        """
        x: (B, T, D) -> Batch, Time, Dimensions
        returns: probs, logits (B,)
        """
        # 1. Passaggio attraverso LSTM
        # self.lstm restituisce: output, (hidden_state, cell_state)
        # Ci interessa solo l'ultimo hidden_state (h_n) per catturare il contesto finale.
        # h_n shape: (num_layers, Batch, hidden_size)
        _, (h_n, _) = self.lstm(x)

        # 2. Estrazione del contesto
        # Prendiamo l'hidden state dell'ULTIMO layer (-1)
        # x shape diventa: (Batch, hidden_size)
        x = h_n[-1]

        # 3. Classificazione
        logits = self.mlp(x).squeeze(-1)  # (B,)
        probs = torch.sigmoid(logits)

        return probs, logits