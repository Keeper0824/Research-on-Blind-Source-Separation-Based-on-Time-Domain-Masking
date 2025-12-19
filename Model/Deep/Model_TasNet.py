from torch import nn
import Model.Deep.EncoderDecoder as EncoderDecoder
import Model.Deep.TasNet_TCN as TCNModel


class Model(nn.Module):
    """
    Wrapper: Encoder + SeparatorTCN + Decoder
    输入:  mixture (B, 1, T)
    输出:  est_sources (B, num_speakers, T_est)
    """

    def __init__(self, enc_dim=256, enc_kernel=16, enc_stride=8, bottleneck_channels=128, hidden_channels=256,
                 num_blocks=8, num_repeats=3, num_speakers=2):
        super().__init__()
        self.num_speakers = num_speakers

        self.encoder = EncoderDecoder.Encoder(enc_dim, enc_kernel, enc_stride)
        self.separator = TCNModel.SeparatorTCN(enc_channels=enc_dim, bottleneck_channels=bottleneck_channels,
                                               hidden_channels=hidden_channels, kernel_size=3, num_blocks=num_blocks,
                                               num_repeats=num_repeats, num_speakers=num_speakers)
        self.decoder = EncoderDecoder.Decoder(enc_dim, enc_kernel, enc_stride)

    def forward(self, mixture):
        """
        mixture: (B, 1, T)
        return:  (B, num_speakers, T_est)
        """
        enc_feat = self.encoder(mixture)  # (B, N, L)
        masks = self.separator(enc_feat)  # (B, S, N, L)
        masked_feat = masks * enc_feat.unsqueeze(1)  # (B, S, N, L)
        out = self.decoder(masked_feat)  # (B, S, T_est)
        return out
