import torch
import BSSLoss

# 1) 只用 SI-SNR
loss_fn = BSSLoss.BSSLoss(mode="si_snr")

# 2) SI-SNR + MAE
loss_fn = BSSLoss.BSSLoss(mode="si_snr_mae", lambda_mae=0.1)

# 3) SI-SNR + MAE + STOI
loss_fn = BSSLoss.BSSLoss(
    mode="si_snr_mae_stoi",
    lambda_mae=0.05,
    lambda_stoi=0.1,
    sample_rate=16000,
    extended_stoi=False,
)

B = 4       # batch size
T = 16000   # 1 秒的音频 (16kHz)

est = torch.randn(B, T, requires_grad=True)
ref = torch.randn(B, T)

loss = loss_fn(est, ref)
print("Loss value:", loss.item())
loss.backward()
print("Backward OK, est.grad shape:", est.grad.shape)
