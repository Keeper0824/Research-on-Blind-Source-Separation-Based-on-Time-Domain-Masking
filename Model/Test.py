import torch
from Model.Baseline import EncoderDecoder, TasNet_TCN as TCNModel

# import TCN as TCNModel

# encoder = MultiscaleEncoderDecoder.MultiScaleEncoder(out_channels=256, kernel_sizes=[16,32,64], stride=8)
# decoder = MultiscaleEncoderDecoder.SharedDecoder(in_channels=256, kernel_size=16, stride=8)
encoder = EncoderDecoder.Encoder(enc_dim=256)
decoder = EncoderDecoder.Decoder(enc_dim=256)
separation = TCNModel.SeparatorTCN(enc_channels=256, bottleneck_channels=128)
# separation = TCNModel.SeparatorTCN(256, 2)

mixture = torch.randn(4, 1, 32000)  # example input mixture (4 batches)

enc_feat = encoder(mixture)
masks = separation(enc_feat)
masked_feat = masks * enc_feat.unsqueeze(1)   # [B,S,N,K]
separated = decoder(masked_feat)

print(separated.shape)  # (4, 2, T)

