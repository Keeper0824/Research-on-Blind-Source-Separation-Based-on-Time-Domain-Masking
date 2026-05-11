import torch
import torch.nn as nn

class ConvGroupNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.GroupNorm(1, out_channels, eps=1e-8),  # Use GroupNorm (G=1 is equivalent to LayerNorm, more stable for audio)
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)

class ConvGroupNormGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2

        # The convolution output must be 2 times the size of out_channels
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(1, out_channels * 2, eps=1e-6)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.glu(x)  # After GLU is executed, the number of channels changes back to out_channels


class PostRefine(nn.Module):
    def __init__(self, C, depth=3, k=3, scale=0.1):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(ConvGroupNormGLU(C, C, kernel_size=k))

        self.layers = nn.ModuleList(layers)
        self.scales = nn.Parameter(torch.zeros(depth))

    def forward(self, x):
        for i, blk in enumerate(self.layers):
            x = x + torch.tanh(self.scales[i]) * blk(x)
        return x


# --- Encoder part (naming convention corrected) ---
class ConvDownBlock(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super().__init__()
        # Input: 1 -> Block1 -> Down1 -> Block2 -> Down2 -> Block3 -> Down3
        # Assume eventually out_channels is C
        pad = (kernel_size - 1) // 2
        # Stage 1: 1 -> C/4
        self.block1 = ConvGroupNormReLU(1, out_channels // 2, kernel_size)
        self.down1 = nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=kernel_size, stride=2, padding=pad)

        # Stage 2: C/4 -> C/2
        self.block2 = ConvGroupNormReLU(out_channels // 2, out_channels, kernel_size)
        self.down2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=pad)

        # Stage 3: C/2 -> C
        self.block3 = ConvGroupNormReLU(out_channels, out_channels, kernel_size)
        self.down3 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=pad)

    def forward(self, x):
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.down3(x)
        return x


class MultiScaleEncoder(nn.Module):
    """Corresponding to picture 2 (image_676301.jpg) Encoder part on the left Contains three branches: Short, Middle, and Long"""

    def __init__(self, enc_dim=256, kernel_sizes=[5, 11, 21], stride=8):
        super().__init__()
        self.enc_dim = enc_dim

        # Make sure the total dimension is divisible by 3, or manually assign the dimensions of each branch
        # Here we assume that each branch outputs dim // 3
        branch_dim = enc_dim // 3
        self.branch_dim = branch_dim

        # Bugfix: if enc_dim is not divisible by 3, the last branch is completed.
        last_dim = enc_dim - (branch_dim * 2)

        self.short_branch = ConvDownBlock(branch_dim, kernel_size=kernel_sizes[0])
        self.mid_branch = ConvDownBlock(branch_dim, kernel_size=kernel_sizes[1])
        self.long_branch = ConvDownBlock(last_dim, kernel_size=kernel_sizes[2])

        self.post_refine = PostRefine(enc_dim, depth=3)

    def forward(self, x):
        # x: (B, 1, T)
        s_out = self.short_branch(x)
        m_out = self.mid_branch(x)
        l_out = self.long_branch(x)

        # Splicing: (B, C_short, L) + (B, C_mid, L) +... -> (B, enc_dim, L)
        enc_feat = torch.cat([s_out, m_out, l_out], dim=1)
        # enc_feat = self.post_refine(enc_feat)
        return enc_feat


# --- Decoder part (completely rewritten to fix logic) ---

class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=pad, bias=False)
        )
        self.block3 = ConvGroupNormReLU(in_channels, in_channels, kernel_size)

        # Stage 2 Inverse
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=pad, bias=False)
        )
        self.block2 = ConvGroupNormReLU(in_channels, in_channels // 2, kernel_size)

        # Stage 1 Inverse
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(in_channels // 2, in_channels // 2, kernel_size=kernel_size, padding=pad, bias=False)
        )
        self.block1 = ConvGroupNormReLU(in_channels // 2, in_channels // 2, kernel_size)

        self.out_channels = in_channels // 2

    def forward(self, x):
        x = self.up3(x)
        x = self.block3(x)
        x = self.up2(x)
        x = self.block2(x)
        x = self.up1(x)
        x = self.block1(x)
        return x

    # class ConvUpBlock(nn.Module):
    #     def __init__(self, in_channels, kernel_size):
    #         super().__init__()
    # # Here in_channels is the dimension of the Encoder output (i.e. C)
    # # We need to completely reverse the Encoder process:
    #         # Encoder: 1 -> C/4 -> C/2 -> C
    # # Decoder: C -> C/2 -> C/4 -> Out (here we let it output C/4 as a feature)
    # Implementation note.
    # # Stage 3 Inverse (corresponding Down3 reverse): C -> C/2.
    #         pad, outpad = _deconv_padding_outpad(kernel_size)
    #         self.up3 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=kernel_size, stride=2, padding=pad,
    #                                       output_padding=outpad)
    #         self.block3 = ConvGroupNormReLU(in_channels, in_channels // 2, kernel_size)
    # Implementation note.
    # # Stage 2 Inverse (corresponding Down2 reverse): C/2 -> C/4.
    #         self.up2 = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=kernel_size, stride=2,
    #                                       padding=pad, output_padding=outpad)
    #         self.block2 = ConvGroupNormReLU(in_channels // 2, in_channels // 4, kernel_size)
    # Implementation note.
    # # Stage 1 Inverse (corresponding Down1 inverse): C/4 -> C/4 (maintain feature dimensions for final fusion)
    #         self.up1 = nn.ConvTranspose1d(in_channels // 4, in_channels // 4, kernel_size=kernel_size, stride=2,
    #                                       padding=pad, output_padding=outpad)
    #         self.block1 = ConvGroupNormReLU(in_channels // 4, in_channels // 4, kernel_size)
    # Implementation note.
    # # Record the final output channel number for automatic calculation by Decoder
    #         self.out_channels = in_channels // 4


def forward(self, x):
    # Sequence: Up3 -> Block3 -> Up2 -> Block2 -> Up1 -> Block1.
    x = self.up3(x)
    x = self.block3(x)
    x = self.up2(x)
    x = self.block2(x)
    x = self.up1(x)
    x = self.block1(x)
    return x


class MultiScaleDecoder(nn.Module):
    def __init__(self, enc_dim=256, kernel_sizes=[5, 11, 21], stride=8):
        super().__init__()
        # Check whether stride matches: ConvUpBlock internally performs 3 times of upsampling with stride=2, total stride=8.
        if stride != 8:
            print("Warning: Manual stride parameter is ignored, fixed to 8 by architecture.")

        branch_dim = enc_dim // 3
        last_dim = enc_dim - (branch_dim * 2)
        self.splits = [branch_dim, branch_dim, last_dim]

        # Initialize three branches
        self.short_branch = ConvUpBlock(branch_dim, kernel_size=kernel_sizes[0])
        self.mid_branch = ConvUpBlock(branch_dim, kernel_size=kernel_sizes[1])
        self.long_branch = ConvUpBlock(last_dim, kernel_size=kernel_sizes[2])

        # Automatically calculate the number of input channels of the fusion layer (avoid manual calculation errors)
        total_out_channels = (
                self.short_branch.out_channels +
                self.mid_branch.out_channels +
                self.long_branch.out_channels
        )
        self.fusion_conv = nn.Conv1d(total_out_channels, 96, 1)
        self.post_refine = PostRefine(96, depth=3)
        self.final_concat = nn.Conv1d(96, 1, kernel_size=1)

    def forward(self, masked_feat):
        # If the input is 4D (B, Source, N, L), process it first and in the Batch dimension
        is_4d = False
        if masked_feat.dim() == 4:
            is_4d = True
            B, S, N, L = masked_feat.size()
            x = masked_feat.view(B * S, N, L)
        else:
            x = masked_feat



        # 1. Split features
        cur = 0
        s_in = x[:, cur:cur + self.splits[0], :]
        cur += self.splits[0]
        m_in = x[:, cur:cur + self.splits[1], :]
        cur += self.splits[1]
        l_in = x[:, cur:, :]

        # 2. Parallel decoding
        s_out = self.short_branch(s_in)
        m_out = self.mid_branch(m_in)
        l_out = self.long_branch(l_in)

        # 3. Cutting & Stitching.
        decoded_feat = torch.cat([s_out, m_out, l_out], dim=1)
        decoded_feat = self.fusion_conv(decoded_feat)
        # decoded_feat = self.post_refine(decoded_feat)

        # 4. Fusion output
        waveform = self.final_concat(decoded_feat)


        if is_4d:
            waveform = waveform.view(B, S, -1)

        return waveform
