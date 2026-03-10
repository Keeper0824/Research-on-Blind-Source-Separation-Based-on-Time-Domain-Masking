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

        self.encoder_conv = nn.Conv1d(in_channels=1, out_channels=enc_dim, kernel_size=kernel_size, stride=stride,
                                      padding=0, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, 1, T)

        Returns:
            enc_feat: (B, N, L)
        """
        return torch.relu(self.encoder_conv(x))


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

        self.decoder_deconv = nn.ConvTranspose1d(
            in_channels=enc_dim,
            out_channels=1,  # reconstruct 1 waveform per speaker
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False
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
        y = self.decoder_deconv(x)  # (B*C, 1, T)
        y = y.view(B, C, -1)  # (B, C, T)
        return y
