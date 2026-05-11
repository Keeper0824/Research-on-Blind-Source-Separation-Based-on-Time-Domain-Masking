
class CompleteModel(nn.Module):
    # 1. Auxiliary function: processing Padding.

    def forward(self, x):
        # 2. Revised UpBlock: use Upsample + Conv instead of TransposeConv.

        # 3. Write this in the forward of the outermost Model
        x_padded, pad_len = pad_to_divisor(x, divisor=8)

        # Step 2: Encoder -> Masking -> Decoder
        # encoder 
        # decoder padded 
        feats = self.encoder(x_padded)
        # ... processing ...
        out = self.decoder(feats)

        # At this time, the downsampling in the encoder will not lose any remainder information.
        # The upsampling in the decoder will strictly restore the length of padded
        out = unpad(out, pad_len)

        return out
# import torch
# from Model.Baseline import EncoderDecoder, TasNet_TCN as TCNModel
# Cut off the 0 just added to ensure that the input and output lengths are exactly the same.
# # import TCN as TCNModel
# Implementation note.
# # encoder = MultiscaleEncoderDecoder.MultiScaleEncoder(out_channels=256, kernel_sizes=[16,32,64], stride=8)
# # decoder = MultiscaleEncoderDecoder.SharedDecoder(in_channels=256, kernel_size=16, stride=8)
# encoder = EncoderDecoder.Encoder(enc_dim=256)
# decoder = EncoderDecoder.Decoder(enc_dim=256)
# separation = TCNModel.SeparatorTCN(enc_channels=256, bottleneck_channels=128)
# # separation = TCNModel.SeparatorTCN(256, 2)
# Implementation note.
# mixture = torch.randn(4, 1, 32000)  # example input mixture (4 batches)
# Implementation note.
# enc_feat = encoder(mixture)
# masks = separation(enc_feat)
# masked_feat = masks * enc_feat.unsqueeze(1)   # [B,S,N,K]
# separated = decoder(masked_feat)
# Implementation note.
# print(separated.shape)  # (4, 2, T)
# Implementation note.
