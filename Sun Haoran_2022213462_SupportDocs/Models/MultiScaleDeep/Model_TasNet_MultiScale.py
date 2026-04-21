from torch import nn
import torch.nn.functional as F
import Model.MultiScaleDeep.MultiscaleEncoderDecoder as EncoderDecoder
import Model.MultiScaleDeep.TasNet_TCN as TCNModel


class Model(nn.Module):
    """
    Wrapper: Encoder + SeparatorTCN + Decoder
    Input:  mixture (B, 1, T)
    Output:  est_sources (B, num_speakers, T_est)
    """

    def __init__(self, enc_dim=256, enc_kernel=[5, 10, 20], enc_stride=8, bottleneck_channels=128, hidden_channels=256,
                 num_blocks=8, num_repeats=3, num_speakers=2):
        super().__init__()
        self.num_speakers = num_speakers

        self.encoder = EncoderDecoder.MultiScaleEncoder(enc_dim=enc_dim, kernel_sizes=enc_kernel,
                                                        stride=enc_stride)
        self.separator = TCNModel.SeparatorTCN(enc_channels=enc_dim, bottleneck_channels=bottleneck_channels,
                                               hidden_channels=hidden_channels, kernel_size=3, num_blocks=num_blocks,
                                               num_repeats=num_repeats, num_speakers=num_speakers)
        self.decoder = EncoderDecoder.MultiScaleDecoder(enc_dim=enc_dim, kernel_sizes=enc_kernel, stride=8)

    def pad_to_divisor(self, x, divisor=8):
        T = x.size(-1)
        rem = T % divisor
        if rem == 0:
            return x, 0
        pad_len = divisor - rem
        return F.pad(x, (0, pad_len), value=0), pad_len

    def unpad(self, x, pad_len):
        if pad_len > 0:
            return x[..., :-pad_len]
        return x

    def forward(self, mixture):
        """
        mixture: (B, 1, T)
        return:  (B, num_speakers, T_est)
        """
        x_padded, pad_len = self.pad_to_divisor(mixture, divisor=8)
        enc_feat = self.encoder(x_padded)  # (B, N, L)
        masks = self.separator(enc_feat)  # (B, S, N, L)
        masked_feat = masks * enc_feat.unsqueeze(1)  # (B, S, N, L)
        out = self.decoder(masked_feat)  # (B, S, T_est)
        out = self.unpad(out, pad_len)
        return out
