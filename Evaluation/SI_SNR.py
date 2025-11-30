import torch

def l2norm(mat, keep_dim=False):
    return torch.norm(mat, dim=-1, keepdim=keep_dim)
def si_snr_metric(x, s, eps=1e-13):
    """
    Returns positive SI-SNR in dB (higher is better).
    """
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    proj = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (l2norm(s_zm, keep_dim=True) ** 2 + eps)
    si_snr_value = 20 * torch.log10((l2norm(proj) + eps) / (l2norm(x_zm - proj) + eps))
    return torch.mean(si_snr_value)
