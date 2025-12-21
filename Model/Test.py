# 1. 辅助函数：处理 Padding



# 2. 修正后的 UpBlock：使用 Upsample + Conv 代替 TransposeConv



# 3. 在最外层 Model 的 forward 里这样写
class CompleteModel(nn.Module):
    # ... init ...

    def forward(self, x):
        # x: (B, 1, T)

        # Step 1: Pad
        x_padded, pad_len = pad_to_divisor(x, divisor=8)

        # Step 2: Encoder -> Masking -> Decoder
        # 此时 encoder 里的下采样不会丢失任何余数信息
        # decoder 里的上采样会严格还原回 padded 的长度
        feats = self.encoder(x_padded)
        # ... processing ...
        out = self.decoder(feats)

        # Step 3: Unpad
        # 裁剪掉刚才补的 0，保证输入输出长度完全一致
        out = unpad(out, pad_len)

        return out
# import torch
# from Model.Baseline import EncoderDecoder, TasNet_TCN as TCNModel
#
# # import TCN as TCNModel
#
# # encoder = MultiscaleEncoderDecoder.MultiScaleEncoder(out_channels=256, kernel_sizes=[16,32,64], stride=8)
# # decoder = MultiscaleEncoderDecoder.SharedDecoder(in_channels=256, kernel_size=16, stride=8)
# encoder = EncoderDecoder.Encoder(enc_dim=256)
# decoder = EncoderDecoder.Decoder(enc_dim=256)
# separation = TCNModel.SeparatorTCN(enc_channels=256, bottleneck_channels=128)
# # separation = TCNModel.SeparatorTCN(256, 2)
#
# mixture = torch.randn(4, 1, 32000)  # example input mixture (4 batches)
#
# enc_feat = encoder(mixture)
# masks = separation(enc_feat)
# masked_feat = masks * enc_feat.unsqueeze(1)   # [B,S,N,K]
# separated = decoder(masked_feat)
#
# print(separated.shape)  # (4, 2, T)
#
