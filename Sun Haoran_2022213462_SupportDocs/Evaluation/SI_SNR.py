import torch


def si_snr(est, ref, eps=1e-8):
    """
    SI-SNR in dB
    returns: scalar tensor
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)

    proj = (torch.sum(est * ref, dim=-1, keepdim=True) * ref) / (torch.sum(ref * ref, dim=-1, keepdim=True) + eps)
    noise = est - proj

    ratio = (torch.sum(proj * proj, dim=-1) + eps) / (torch.sum(noise * noise, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def pit_align_by_sisnr(est_b2t, ref_b2t):
    """
    PIT alignment for 2 speakers by SI-SNR.
    est_b2t/ref_b2t: [B,2,T]
    returns:
      est_aligned: [B,2,T]
      best_perm: list of length B, each is tuple like (0,1) or (1,0)
      sisnr_pair: [B] best (mean over 2 speakers) SI-SNR
    """
    B = est_b2t.size(0)

    # perm1: (0,1)
    s00 = si_snr(est_b2t[:, 0, :], ref_b2t[:, 0, :])
    s11 = si_snr(est_b2t[:, 1, :], ref_b2t[:, 1, :])
    score_p01 = 0.5 * (s00 + s11)

    # perm2: (1,0)
    s10 = si_snr(est_b2t[:, 1, :], ref_b2t[:, 0, :])
    s01 = si_snr(est_b2t[:, 0, :], ref_b2t[:, 1, :])
    score_p10 = 0.5 * (s10 + s01)

    choose_p01 = score_p01 >= score_p10  # [B]
    est_aligned = est_b2t.clone()
    est_aligned[~choose_p01] = est_b2t[~choose_p01][:, [1, 0], :]

    best_score = torch.where(choose_p01, score_p01, score_p10)

    best_perm = [(0, 1) if bool(c.item()) else (1, 0) for c in choose_p01]
    return est_aligned, best_perm, best_score


def si_snr_improvement(est_b2t, ref_b2t, mix_bt):
    """
    SI-SNRi after PIT alignment. All tensors on same device.
    est_b2t/ref_b2t: [B,2,T], mix_bt: [B,T]
    returns: [B] (mean over 2 speakers improvement)
    """
    # baseline: mix vs each ref
    mix_rep = mix_bt.unsqueeze(1).expand_as(ref_b2t)
    base = 0.5 * (si_snr(mix_rep[:, 0, :], ref_b2t[:, 0, :]) + si_snr(mix_rep[:, 1, :], ref_b2t[:, 1, :]))

    # aligned est
    est_aligned, _, est_score = pit_align_by_sisnr(est_b2t, ref_b2t)
    # est_score already mean SI-SNR over 2 speakers
    return est_score - base
