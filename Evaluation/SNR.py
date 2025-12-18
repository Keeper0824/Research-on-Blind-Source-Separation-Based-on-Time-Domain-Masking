import torch


def snr(est, ref, eps=1e-8):
    alpha = (est * ref).sum(dim=-1, keepdim=True) / ((est * est).sum(dim=-1, keepdim=True) + eps)
    est_s = alpha * est
    num = (ref * ref).sum(dim=-1) + eps
    den = ((ref - est_s) ** 2).sum(dim=-1) + eps
    return 10 * torch.log10(num / den)
