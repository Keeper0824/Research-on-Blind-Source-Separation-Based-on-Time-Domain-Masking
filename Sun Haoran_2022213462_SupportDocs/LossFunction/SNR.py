import torch

def negative_snr_loss(x, s, eps=1e-13):
   """
   Args:
       x: Enhanced fo shape [B, T]
       s: Reference of shape [B, T]
   Returns:
       snr: [B] negative SNR loss
   """
   x_zm = x - torch.mean(x, dim=-1, keepdim=True)
   s_zm = s - torch.mean(s, dim=-1, keepdim=True)
   tmp = torch.mean(s_zm ** 2, dim=-1, keepdim=True) / (torch.mean((x_zm - s_zm) ** 2, dim=-1, keepdim=True) + eps)
   loss = torch.squeeze(-10 * torch.log10(tmp + eps))
   loss = torch.mean(loss)
   return loss