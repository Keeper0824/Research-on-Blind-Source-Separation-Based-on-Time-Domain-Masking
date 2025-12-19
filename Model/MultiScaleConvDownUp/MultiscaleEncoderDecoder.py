import torch
import torch.nn as nn


def crop_to_min_len(xs):
    """
    辅助函数：因为不同卷积核会导致输出长度有细微差别，
    这里将所有分支裁剪到最短的那个长度，以便拼接。
    """
    min_len = min(x.size(-1) for x in xs)
    return [x[..., :min_len].contiguous() for x in xs]


def _deconv_padding_outpad(k):
    pad = (k - 1) // 2
    outpad = 1 if (k % 2 == 1) else 0
    return pad, outpad


class ConvGroupNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.GroupNorm(1, out_channels),  # 使用 GroupNorm (G=1 等同于 LayerNorm，对音频更稳)
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)


class PostRefine(nn.Module):
    def __init__(self, C, depth=4, k=3):
        super().__init__()
        pad = (k - 1) // 2
        layers = []
        for _ in range(depth):
            layers += [
                nn.Conv1d(C, C, k, padding=pad),
                nn.GroupNorm(1, C),
                nn.PReLU(C),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- Encoder 部分 (修正了命名规范) ---
class ConvDownBlock(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super().__init__()
        # Input: 1 -> Block1 -> Down1 -> Block2 -> Down2 -> Block3 -> Down3
        # 假设最终 out_channels 是 C
        pad, outpad = _deconv_padding_outpad(kernel_size)
        # Stage 1: 1 -> C/4
        self.block1 = ConvGroupNormReLU(1, out_channels // 4, kernel_size)
        self.down1 = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=kernel_size, stride=2, padding=pad)

        # Stage 2: C/4 -> C/2
        self.block2 = ConvGroupNormReLU(out_channels // 4, out_channels // 2, kernel_size)
        self.down2 = nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=kernel_size, stride=2, padding=pad)

        # Stage 3: C/2 -> C
        self.block3 = ConvGroupNormReLU(out_channels // 2, out_channels, kernel_size)
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

        # 裁剪对齐
        outs = crop_to_min_len([s_out, m_out, l_out])

        # 拼接: (B, C_short, L) + (B, C_mid, L) + ... -> (B, enc_dim, L)
        enc_feat = torch.cat(outs, dim=1)
        enc_feat = self.post_refine(enc_feat)
        return enc_feat


# --- Decoder 部分 (完全重写以修复逻辑) ---
class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        # 这里的 in_channels 是 Encoder 输出的维度 (即 C)
        # 我们需要完全逆向 Encoder 的过程:
        # Encoder: 1 -> C/4 -> C/2 -> C
        # Decoder: C -> C/2 -> C/4 -> Out (我们这里让它输出 C/4 作为特征)

        # Stage 3 Inverse (对应的 Down3 逆向): C -> C/2
        pad, outpad = _deconv_padding_outpad(kernel_size)
        self.up3 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=kernel_size, stride=2, padding=pad,
                                      output_padding=outpad)
        self.block3 = ConvGroupNormReLU(in_channels, in_channels // 2, kernel_size)

        # Stage 2 Inverse (对应的 Down2 逆向): C/2 -> C/4
        self.up2 = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=kernel_size, stride=2,
                                      padding=pad, output_padding=outpad)
        self.block2 = ConvGroupNormReLU(in_channels // 2, in_channels // 4, kernel_size)

        # Stage 1 Inverse (对应的 Down1 逆向): C/4 -> C/4 (保持特征维度以便最后融合)
        self.up1 = nn.ConvTranspose1d(in_channels // 4, in_channels // 4, kernel_size=kernel_size, stride=2,
                                      padding=pad, output_padding=outpad)
        self.block1 = ConvGroupNormReLU(in_channels // 4, in_channels // 4, kernel_size)

        # 记录最终输出通道数，供 Decoder 自动计算
        self.out_channels = in_channels // 4

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

        print(f"Decoder Final Fusion Channels: {total_out_channels}")  # 调试用

        self.post_refine = PostRefine(enc_dim, depth=3)
        self.final_concat = nn.Conv1d(total_out_channels, 1, kernel_size=1)

    def forward(self, masked_feat):
        # 如果输入是 4D (B, Source, N, L)，先并在 Batch 维度处理
        is_4d = False
        if masked_feat.dim() == 4:
            is_4d = True
            B, S, N, L = masked_feat.size()
            x = masked_feat.view(B * S, N, L)
        else:
            x = masked_feat

        x = self.post_refine(x)

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
        outs = crop_to_min_len([s_out, m_out, l_out])
        decoded_feat = torch.cat(outs, dim=1)

        # 4. 融合输出
        waveform = self.final_concat(decoded_feat)

        if is_4d:
            waveform = waveform.view(B, S, -1)

        return waveform
