import torch
import torch.nn as nn

class ConvGroupNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.GroupNorm(1, out_channels, eps=1e-8),  # 使用 GroupNorm (G=1 等同于 LayerNorm，对音频更稳)
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)

class ConvGroupNormGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2

        # 卷积输出必须是 out_channels 的 2 倍
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(1, out_channels * 2, eps=1e-6)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.glu(x)  # GLU 执行后，通道数变回 out_channels


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


# --- Encoder 部分 (修正了命名规范) ---
class ConvDownBlock(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super().__init__()
        # Input: 1 -> Block1 -> Down1 -> Block2 -> Down2 -> Block3 -> Down3
        # 假设最终 out_channels 是 C
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
    """
    对应图片2 (image_676301.jpg) 左侧 Encoder 部分
    包含 Short, Middle, Long 三个分支
    """

    def __init__(self, enc_dim=256, kernel_sizes=[5, 11, 21], stride=8):
        super().__init__()
        self.enc_dim = enc_dim

        # 确保总维度能被3整除，或者手动分配每个分支的维度
        # 这里我们假设每个分支输出 dim // 3
        branch_dim = enc_dim // 3
        self.branch_dim = branch_dim

        # 修正：如果 enc_dim 不能被3整除，最后一个分支补齐
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

        # 拼接: (B, C_short, L) + (B, C_mid, L) + ... -> (B, enc_dim, L)
        enc_feat = torch.cat([s_out, m_out, l_out], dim=1)
        # enc_feat = self.post_refine(enc_feat)
        return enc_feat


# --- Decoder 部分 (完全重写以修复逻辑) ---

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
    #         # 这里的 in_channels 是 Encoder 输出的维度 (即 C)
    #         # 我们需要完全逆向 Encoder 的过程:
    #         # Encoder: 1 -> C/4 -> C/2 -> C
    #         # Decoder: C -> C/2 -> C/4 -> Out (我们这里让它输出 C/4 作为特征)
    #
    #         # Stage 3 Inverse (对应的 Down3 逆向): C -> C/2
    #         pad, outpad = _deconv_padding_outpad(kernel_size)
    #         self.up3 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=kernel_size, stride=2, padding=pad,
    #                                       output_padding=outpad)
    #         self.block3 = ConvGroupNormReLU(in_channels, in_channels // 2, kernel_size)
    #
    #         # Stage 2 Inverse (对应的 Down2 逆向): C/2 -> C/4
    #         self.up2 = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=kernel_size, stride=2,
    #                                       padding=pad, output_padding=outpad)
    #         self.block2 = ConvGroupNormReLU(in_channels // 2, in_channels // 4, kernel_size)
    #
    #         # Stage 1 Inverse (对应的 Down1 逆向): C/4 -> C/4 (保持特征维度以便最后融合)
    #         self.up1 = nn.ConvTranspose1d(in_channels // 4, in_channels // 4, kernel_size=kernel_size, stride=2,
    #                                       padding=pad, output_padding=outpad)
    #         self.block1 = ConvGroupNormReLU(in_channels // 4, in_channels // 4, kernel_size)
    #
    #         # 记录最终输出通道数，供 Decoder 自动计算
    #         self.out_channels = in_channels // 4


def forward(self, x):
    # 顺序：Up3 -> Block3 -> Up2 -> Block2 -> Up1 -> Block1
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
        # 检查 stride 是否匹配：ConvUpBlock 内部做了 3 次 stride=2 的上采样，总 stride=8
        if stride != 8:
            print("Warning: Manual stride parameter is ignored, fixed to 8 by architecture.")

        branch_dim = enc_dim // 3
        last_dim = enc_dim - (branch_dim * 2)
        self.splits = [branch_dim, branch_dim, last_dim]

        # 初始化三个分支
        self.short_branch = ConvUpBlock(branch_dim, kernel_size=kernel_sizes[0])
        self.mid_branch = ConvUpBlock(branch_dim, kernel_size=kernel_sizes[1])
        self.long_branch = ConvUpBlock(last_dim, kernel_size=kernel_sizes[2])

        # 自动计算融合层的输入通道数 (避免手动计算出错)
        total_out_channels = (
                self.short_branch.out_channels +
                self.mid_branch.out_channels +
                self.long_branch.out_channels
        )
        self.fusion_conv = nn.Conv1d(total_out_channels, 96, 1)
        self.post_refine = PostRefine(96, depth=3)
        self.final_concat = nn.Conv1d(96, 1, kernel_size=1)

    def forward(self, masked_feat):
        # 如果输入是 4D (B, Source, N, L)，先并在 Batch 维度处理
        is_4d = False
        if masked_feat.dim() == 4:
            is_4d = True
            B, S, N, L = masked_feat.size()
            x = masked_feat.view(B * S, N, L)
        else:
            x = masked_feat



        # 1. 拆分特征
        cur = 0
        s_in = x[:, cur:cur + self.splits[0], :]
        cur += self.splits[0]
        m_in = x[:, cur:cur + self.splits[1], :]
        cur += self.splits[1]
        l_in = x[:, cur:, :]

        # 2. 并行解码
        s_out = self.short_branch(s_in)
        m_out = self.mid_branch(m_in)
        l_out = self.long_branch(l_in)

        # 3. 裁剪 & 拼接
        decoded_feat = torch.cat([s_out, m_out, l_out], dim=1)
        decoded_feat = self.fusion_conv(decoded_feat)
        # decoded_feat = self.post_refine(decoded_feat)

        # 4. 融合输出
        waveform = self.final_concat(decoded_feat)


        if is_4d:
            waveform = waveform.view(B, S, -1)

        return waveform
