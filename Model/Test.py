import torch
import EncoderDecoder
# import TCN as TCNModel
import TasNet_TCN as TCNModel

encoder = EncoderDecoder.Encoder(enc_dim=256)
separation = TCNModel.SeparatorTCN(enc_channels=256, bottleneck_channels=128)
# separation = TCNModel.SeparatorTCN(256, 2)
decoder = EncoderDecoder.Decoder(enc_dim=256)

mixture = torch.randn(4, 1, 32000)  # example input mixture (4 batches)

enc_feat = encoder(mixture)

# masks, masked_feat = separation(enc_feat)
masks = separation(enc_feat)

separated = decoder(masks)

print(separated.shape)  # (4, 2, T)
