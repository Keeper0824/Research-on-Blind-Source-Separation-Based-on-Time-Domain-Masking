import torch
from itertools import permutations

def l2norm(mat, keep_dim=False):
    return torch.norm(mat, dim=-1, keepdim=keep_dim)


def si_snr(x, s, eps=1e-8):
    """
    Args:
        x: estimated signal, shape [B, T]
        s: reference signal, shape [B, T]
    Returns:
        si-snr: [B] SI-SNR loss
    """
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    proj = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (l2norm(s_zm, keep_dim=True) ** 2 + eps)
    si_snr_value = 20 * torch.log10((l2norm(proj) + eps) / (l2norm(x_zm - proj) + eps))
    return si_snr_value


def negative_si_snr(x, s):
    """
    Args:
        x: estimated signal, shape [B, T]
        s: reference signal, shape [B, T]
    Returns:
        si-snr: [B] negative SI-SNR loss
    """
    return -torch.mean(si_snr(x, s))


def pit_si_snr_loss(
    ests: torch.Tensor,
    refs: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Permutation Invariant Training (PIT) 的 SI-SNR loss.

    ests: [B, S, T]
    refs: [B, S, T]
    """
    B, S, T = ests.shape
    perms = list(permutations(range(S)))
    sisnr_set = []
    for perm in perms:
        s_perm = refs[:, perm, :]
        sisnr_per_src = []
        for s_idx in range(S):
            sisnr_per_src.append(
                si_snr(ests[:, s_idx, :], s_perm[:, s_idx, :], eps=eps)
            )   # [B]

        sisnr_avg = sum(sisnr_per_src) / S      # [B]
        sisnr_set.append(sisnr_avg)

    sisnr_stack = torch.stack(sisnr_set, dim=1)
    max_sisnr, _ = torch.max(sisnr_stack, dim=1)  # [B]
    loss = -torch.mean(max_sisnr)
    return loss
