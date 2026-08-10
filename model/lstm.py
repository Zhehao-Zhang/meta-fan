import torch
import torch.nn as nn


class PureLSTM(nn.Module):
    """Pure LSTM backbone for time-series prediction.

    Input (B, T, F) → LSTM → last timestep hidden → FC → output
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def freeze_lstm(self):
        for p in self.lstm.parameters():
            p.requires_grad = False

    def unfreeze_lstm(self):
        for p in self.lstm.parameters():
            p.requires_grad = True

    def freeze_fc(self):
        for p in self.fc.parameters():
            p.requires_grad = False

    def unfreeze_fc(self):
        for p in self.fc.parameters():
            p.requires_grad = True

    def forward(self, x):
        # x: (B, T, F)
        lstm_out, _ = self.lstm(x)        # (B, T, H)
        last_hidden = lstm_out[:, -1, :]   # (B, H)
        return self.relu(self.fc(last_hidden))
