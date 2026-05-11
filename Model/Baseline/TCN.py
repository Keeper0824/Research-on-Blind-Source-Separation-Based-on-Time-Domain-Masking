from torch import nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """
    Ensures causality by removing the extra time steps introduced by padding
    on the right side of the sequence.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: (B, C, T + padding)
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Standard TCN block with the structure:
    Conv1d -> Chomp -> ReLU -> Dropout (x2) + Residual connection.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # If input and output channels do not match, use 1x1 conv for residual mapping.
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SeparatorTCN(nn.Module):
    """ Separation module (corresponding to the "Separation" block in the architecture). Input: enc_feat: Encoder feature map, shape (B, N, L) Output: masks: (B, C, N, L) estimated masks masked_feat: (B, C, N, L) masked encoder features Notation: B: batch size N: encoder feature dimension L: sequence length (number of frames) C: number of sources """

    def __init__(
            self,
            enc_dim: int,  # Number of encoder output channels N
            num_sources: int = 2,  # Number of target sources C
            hidden_channels: int = 256,
            num_blocks: int = 8,  # Number of dilated TCN blocks per repeat
            num_repeats: int = 3,  # Number of repeats of the TCN stack
            kernel_size: int = 3,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.enc_dim = enc_dim
        self.num_sources = num_sources

        # 1x1 bottleneck projection: encoder_dim hidden_channels
        self.bottleneck = nn.Conv1d(enc_dim, hidden_channels, 1)

        # Build the dilated TCN stack (similar to Conv-TasNet separation module)
        tcn_blocks = []
        for r in range(num_repeats):
            for b in range(num_blocks):
                dilation = 2 ** b
                padding = (kernel_size - 1) * dilation
                tcn_blocks.append(
                    TemporalBlock(
                        n_inputs=hidden_channels,
                        n_outputs=hidden_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation,
                        padding=padding,
                        dropout=dropout,
                    )
                )
        self.tcn = nn.Sequential(*tcn_blocks)

        # Final 1x1 conv to produce masks:
        # hidden_channels (num_sources enc_dim)
        self.mask_conv = nn.Conv1d(hidden_channels, num_sources * enc_dim, kernel_size=1)

    def forward(self, enc_feat):
        """ Args: enc_feat: (B, N, L) features from the Encoder Returns: masks: (B, C, N, L) masked_feat: (B, C, N, L) """
        B, N, L = enc_feat.size()
        assert N == self.enc_dim, "enc_dim mismatch: check Encoder and SeparationTCN settings."

        h = self.bottleneck(enc_feat)  # (B, H, L)
        h = self.tcn(h)  # (B, H, L)

        masks = self.mask_conv(h)  # (B, C*N, L)
        masks = masks.view(B, self.num_sources, self.enc_dim, L)

        # Softmax over source dimension ensures masks sum to 1 at each time-frequency position
        masks = F.softmax(masks, dim=1)

        # Broadcast multiplication: (B, 1, N, L) * (B, C, N, L)
        enc_feat_expanded = enc_feat.unsqueeze(1)
        masked_feat = masks * enc_feat_expanded

        return masks, masked_feat
