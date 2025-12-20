import torch
from torch import nn


class Encoder(nn.Module):
    """
    Time-domain convolutional encoder.

    Input:
        waveform: (B, 1, T)

    Output:
        enc_feat: (B, N, L)
    """

    def __init__(self, enc_dim=256, kernel_size=16, stride=8):
        super().__init__()
        self.enc_dim = enc_dim

        self.sequential = nn.Sequential(
            nn.Conv1d(1, enc_dim, kernel_size, stride=stride),
            nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, enc_dim, eps=1e-8),
            nn.PReLU(),
            nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, enc_dim, eps=1e-8),
            nn.PReLU(),
            nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, enc_dim, eps=1e-8),
            nn.PReLU()
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, T)

        Returns:
            enc_feat: (B, N, L)
        """
        return self.sequential(x)


class Decoder(nn.Module):
    """
    Time-domain decoder (inverse of encoder).

    Input:
        masked_feat: (B, C, N, L)

    Output:
        waveforms: (B, C, T)
    """

    def __init__(self, enc_dim=256, kernel_size=16, stride=8):
        super().__init__()
        self.enc_dim = enc_dim

        self.sequential = nn.Sequential(
            nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, enc_dim, eps=1e-8),
            nn.PReLU(),
            nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, enc_dim, eps=1e-8),
            nn.PReLU(),
            nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, enc_dim, eps=1e-8),
            nn.PReLU(),
            nn.ConvTranspose1d(enc_dim, 1, kernel_size=kernel_size, stride=stride, bias=True)
        )

    def forward(self, masked_feat):
        """
        Args:
            masked_feat: (B, C, N, L)

        Returns:
            separated_waveforms: (B, C, T)
        """
        B, C, N, L = masked_feat.size()
        x = masked_feat.view(B * C, N, L)  # (B*C, N, L)
        y = self.sequential(x)  # (B*C, 1, T)
        y = y.view(B, C, -1)  # (B, C, T)
        return y
