import torch

def snr_metric(x, s, eps=1e-13):
   """
   Positive SNR in dB (higher is better).
   """
   x_zm = x - torch.mean(x, dim=-1, keepdim=True)
   s_zm = s - torch.mean(s, dim=-1, keepdim=True)
   tmp = torch.mean(s_zm ** 2, dim=-1, keepdim=True) / (torch.mean((x_zm - s_zm) ** 2, dim=-1, keepdim=True) + eps)
   snr_value = torch.squeeze(10 * torch.log10(tmp + eps))
   return torch.mean(snr_value)
