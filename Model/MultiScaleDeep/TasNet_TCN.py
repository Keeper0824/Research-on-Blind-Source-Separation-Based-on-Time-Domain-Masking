from torch import nn

class TCNBlock(nn.Module):
    """
    One single TCN residual block used in Conv-TasNet.
    Input / Output: (B, B_chan, L), where B_chan = bottleneck channels.
    """

    def __init__(self, bottleneck_channels, hidden_channels, kernel_size=3, dilation=1):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel sizes to maintain temporal length."

        self.bottleneck_channels = bottleneck_channels
        self.hidden_channels = hidden_channels

        # 1x1 conv: bottleneck hidden
        self.conv1x1 = nn.Conv1d(bottleneck_channels, hidden_channels, kernel_size=1, bias=False)

        self.prelu1 = nn.PReLU(hidden_channels)
        self.norm1 = nn.GroupNorm(1, hidden_channels, eps=1e-6)

        # Depthwise dilated convolution: hidden hidden
        # Padding ensures output length == input length
        padding = dilation * (kernel_size - 1) // 2
        self.dconv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, dilation=dilation,
                               padding=padding, groups=hidden_channels, bias=False)

        self.prelu2 = nn.PReLU(hidden_channels)
        self.norm2 = nn.GroupNorm(1, hidden_channels, eps=1e-6)

        # 1x1 convs for residual and skip outputs
        self.res_out = nn.Conv1d(hidden_channels, bottleneck_channels, kernel_size=1)
        self.skip_out = nn.Conv1d(hidden_channels, bottleneck_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, B_chan, L)

        Returns:
            res:  (B, B_chan, L) residual branch
            skip: (B, B_chan, L) skip-connection branch
        """
        out = self.conv1x1(x)
        out = self.prelu1(out)
        out = self.norm1(out)

        out = self.dconv(out)
        out = self.prelu2(out)
        out = self.norm2(out)

        res = self.res_out(out)
        skip = self.skip_out(out)
        return res, skip


class SeparatorTCN(nn.Module):
    """
    Separation module of TasNet using a Temporal Convolutional Network (TCN).
    Input:  encoder output (B, N, L)
    Output: masks (B, num_speakers, N, L)
    """

    def __init__(self, enc_channels, bottleneck_channels, hidden_channels=256, kernel_size=3, num_blocks=8,
                 num_repeats=3, num_speakers=2):
        super().__init__()

        self.enc_channels = enc_channels
        self.bottleneck_channels = bottleneck_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.num_repeats = num_repeats
        self.num_speakers = num_speakers

        # Map encoder output N bottleneck B
        self.encoder_1x1 = nn.Conv1d(enc_channels, bottleneck_channels, kernel_size=1)

        # Build TCN blocks: dilation pattern 1, 2, 4, ... per repeat
        blocks = []
        for r in range(num_repeats):
            for b in range(num_blocks):
                dilation = 2 ** b
                blocks.append(TCNBlock(bottleneck_channels=bottleneck_channels, hidden_channels=hidden_channels,
                                       kernel_size=kernel_size, dilation=dilation))

        self.tcn_blocks = nn.ModuleList(blocks)

        # After accumulating all skips, predict masks via 1x1 conv
        self.mask_conv = nn.Conv1d(
            bottleneck_channels,
            num_speakers * enc_channels,
            kernel_size=1
        )
        self.mask_activation = nn.Sigmoid()  # Standard for Conv-TasNet

    def forward(self, enc_feat):
        """
        Args:
            enc_feat: (B, N, L)

        Returns:
            masks: (B, num_speakers, N, L)
        """
        B, N, L = enc_feat.shape

        # Bottleneck projection
        x = self.encoder_1x1(enc_feat)  # (B, B_chan, L)

        skip_sum = 0
        for block in self.tcn_blocks:
            res, skip = block(x)
            x = x + res  # Residual connection
            skip_sum = skip_sum + skip  # Global skip accumulation

        # Predict masks
        out = self.mask_conv(skip_sum)  # (B, num_speakers*N, L)
        out = self.mask_activation(out)

        # Reshape to (B, num_speakers, N, L)
        out = out.view(B, self.num_speakers, self.enc_channels, L)
        return out
