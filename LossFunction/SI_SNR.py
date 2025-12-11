import torch
from itertools import permutations

def l2norm(mat, keep_dim=False):
    return torch.norm(mat, dim=-1, keepdim=keep_dim)


def si_snr(x, s, eps=1e-13):
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


def negative_si_snr(x, s, eps=1e-13):
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

    ests: [B, S, T]  模型输出的估计源
    refs: [B, S, T]  对应的参考源（顺序未知）
    return: 标量，负的 SI-SNR（越小越好）
    """
    assert ests.ndim == 3 and refs.ndim == 3, "ests/refs 必须是 [B, S, T]"
    B, S, T = ests.shape
    assert refs.shape == (B, S, T), "refs 形状必须和 ests 一致"

    # 所有说话人的排列
    perms = list(permutations(range(S)))   # S=2 时就是 (0,1) 和 (1,0)
    num_perms = len(perms)

    # 对每一种排列，算一遍这个排列下的平均 SI-SNR
    sisnr_set = []
    for perm in perms:
        # permute 参考信号的说话人顺序: [B, S, T]
        s_perm = refs[:, perm, :]
        # 对每一个说话人算 SI-SNR，然后对 S 取平均
        sisnr_per_src = []
        for s_idx in range(S):
            sisnr_per_src.append(
                si_snr(ests[:, s_idx, :], s_perm[:, s_idx, :], eps=eps)
            )   # [B]

        sisnr_avg = sum(sisnr_per_src) / S      # [B]
        sisnr_set.append(sisnr_avg)

    # [B, num_perms]
    sisnr_stack = torch.stack(sisnr_set, dim=1)
    # 每个样本选择 SI-SNR 最大的那一种排列
    max_sisnr, _ = torch.max(sisnr_stack, dim=1)  # [B]

    # 训练用负号，作为 loss
    loss = -torch.mean(max_sisnr)
    return loss
