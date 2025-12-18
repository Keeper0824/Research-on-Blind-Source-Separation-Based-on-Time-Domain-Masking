import os
import math
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

from Model.Baseline.Model_TasNet import Model
from DataLoader.WSJDataLoader import MakeDataLoader


def ensure_mix_bt(mix):
    """
    Ensure mixture is [B, T] (float tensor).
    Accepts common shapes:
      [T] -> [1, T]
      [B, T] -> [B, T]
      [B, 1, T] -> [B, T]
      [1, B, T] -> [B, T]   (你之前那个坑)
    """
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)
    elif mix.dim() == 3:
        # [B,1,T]
        if mix.size(1) == 1:
            mix = mix.squeeze(1)
        # [1,B,T] -> [B,T]
        elif mix.size(0) == 1 and mix.size(2) > mix.size(1):
            mix = mix.squeeze(0)
        else:
            # 其他奇怪情况：尽量把最后一维当 T
            # 优先 squeeze 掉为 1 的维度
            mix = mix.squeeze()
            if mix.dim() == 1:
                mix = mix.unsqueeze(0)
    elif mix.dim() != 2:
        raise ValueError(f"Unsupported mix shape: {tuple(mix.shape)}")
    return mix.contiguous()


def ensure_ref_b2t(ref):
    """
    Ensure reference sources is [B, 2, T]
    Accepts:
      [2, T] -> [1,2,T]
      [B,2,T] -> ok
      [B,T,2] -> permute
      [2,B,T] -> permute
    """
    if ref.dim() == 2:
        # [2,T]
        if ref.size(0) == 2:
            ref = ref.unsqueeze(0)
        # [T,2]
        elif ref.size(1) == 2:
            ref = ref.transpose(0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported ref shape: {tuple(ref.shape)}")

    elif ref.dim() == 3:
        if ref.size(1) == 2:
            pass  # [B,2,T]
        elif ref.size(2) == 2:
            ref = ref.permute(0, 2, 1)  # [B,2,T]
        elif ref.size(0) == 2:
            ref = ref.permute(1, 0, 2)  # [B,2,T]
        else:
            raise ValueError(f"Unsupported ref shape: {tuple(ref.shape)}")
    else:
        raise ValueError(f"Unsupported ref shape: {tuple(ref.shape)}")

    return ref.contiguous()


def si_snr(est, ref, eps=1e-8):
    """
    SI-SNR in dB. est/ref: [T] or [..., T]
    returns: scalar tensor
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)

    proj = (torch.sum(est * ref, dim=-1, keepdim=True) * ref) / (torch.sum(ref * ref, dim=-1, keepdim=True) + eps)
    noise = est - proj

    ratio = (torch.sum(proj * proj, dim=-1) + eps) / (torch.sum(noise * noise, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def snr(est, ref, eps=1e-8):
    alpha = (est * ref).sum(dim=-1, keepdim=True) / ((est * est).sum(dim=-1, keepdim=True) + eps)
    est_s = alpha * est
    num = (ref * ref).sum(dim=-1) + eps
    den = ((ref - est_s) ** 2).sum(dim=-1) + eps
    return 10 * torch.log10(num / den)


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


def try_pesq(est, ref, fs):
    """
    est/ref: [T] numpy float32
    """
    try:
        from pesq import pesq
        mode = "wb" if fs >= 16000 else "nb"
        return float(pesq(fs, ref, est, mode))
    except Exception:
        return float("nan")


def try_stoi(est: np.ndarray, ref: np.ndarray, fs: int) -> float:
    try:
        from pystoi.stoi import stoi
        return float(stoi(ref, est, fs, extended=False))
    except Exception:
        return float("nan")


def eval_test_full_metrics(model, test_loader, sample_rate, device, output_dir, max_save_samples, min_len_needed=16):
    """
    - 对 test set 跑完整个 epoch
    - 保存前 max_save_samples 个样本（mix/ref/est，对齐后）
    - 计算 SI-SNR / SI-SNRi / SNR / PESQ / STOI（均做 PIT 对齐）
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = {
        "SI-SNR": [],
        "SI-SNRi": [],
        "SNR": [],
        "PESQ": [],
        "STOI": [],
    }

    saved = 0

    with torch.no_grad():
        for mix, targets in tqdm(test_loader, desc="Evaluating"):
            mix = ensure_mix_bt(mix)
            targets = ensure_ref_b2t(targets)

            # pad to min_len_needed (Conv kernel safety)
            if mix.shape[-1] < min_len_needed:
                pad_len = min_len_needed - mix.shape[-1]
                mix = torch.nn.functional.pad(mix, (0, pad_len))
                targets = torch.nn.functional.pad(targets, (0, pad_len))

            mix = mix.to(device)  # [B,T]
            targets = targets.to(device)  # [B,2,T]

            # forward: your model expects [B,1,T]
            est = model(mix.unsqueeze(1))  # likely [B,2,T’] or [B,2,T]
            if est.dim() == 3 and est.size(1) != 2:
                raise RuntimeError(f"Model output unexpected: {tuple(est.shape)}")

            # crop to same length
            min_len = min(est.shape[-1], targets.shape[-1], mix.shape[-1])
            est = est[..., :min_len]
            targets = targets[..., :min_len]
            mix = mix[..., :min_len]

            # PIT align
            est_aligned, perm, sisnr_best = pit_align_by_sisnr(est, targets)  # sisnr_best: [B]

            # SI-SNR (mean over 2 spk) = sisnr_best
            si_snr_vals = sisnr_best  # [B]

            # SI-SNRi
            si_snri_vals = si_snr_improvement(est_aligned, targets, mix)  # [B]

            # SNR (mean over 2 spk)
            snr_vals = 0.5 * (snr(est_aligned[:, 0, :], targets[:, 0, :]) + snr(est_aligned[:, 1, :], targets[:, 1, :]))

            # PESQ / STOI：逐条算（CPU + numpy）
            pesq_vals = []
            stoi_vals = []
            est_cpu = est_aligned.detach().cpu().float().numpy()
            ref_cpu = targets.detach().cpu().float().numpy()

            for b in range(est_cpu.shape[0]):
                # each sample: average over 2 speakers
                p_list, s_list = [], []
                for spk in range(2):
                    e = est_cpu[b, spk]
                    r = ref_cpu[b, spk]
                    p_list.append(try_pesq(e.astype(np.float32), r.astype(np.float32), sample_rate))
                    s_list.append(try_stoi(e.astype(np.float32), r.astype(np.float32), sample_rate))
                pesq_vals.append(float(np.nanmean(p_list)))
                stoi_vals.append(float(np.nanmean(s_list)))

            # collect
            all_metrics["SI-SNR"].extend(si_snr_vals.detach().cpu().tolist())
            all_metrics["SI-SNRi"].extend(si_snri_vals.detach().cpu().tolist())
            all_metrics["SNR"].extend(snr_vals.detach().cpu().tolist())
            all_metrics["PESQ"].extend(pesq_vals)
            all_metrics["STOI"].extend(stoi_vals)

            # save wav (first N)
            B = mix.size(0)
            for i in range(B):
                if saved >= max_save_samples:
                    break

                base = f"test_{saved:04d}_perm{perm[i][0]}{perm[i][1]}"
                mix_i = mix[i].detach().cpu()
                s1_ref = targets[i, 0].detach().cpu()
                s2_ref = targets[i, 1].detach().cpu()
                s1_est = est_aligned[i, 0].detach().cpu()
                s2_est = est_aligned[i, 1].detach().cpu()

                torchaudio.save(os.path.join(output_dir, f"{base}_mix.wav"), mix_i.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_ref.wav"), s1_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_ref.wav"), s2_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_est.wav"), s1_est.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_est.wav"), s2_est.unsqueeze(0), sample_rate)

                saved += 1

    # summarize (nan-safe)
    def nanmean(x):
        x = np.array(x, dtype=np.float64)
        return float(np.nanmean(x)) if x.size else float("nan")

    results = {k: nanmean(v) for k, v in all_metrics.items()}

    print("\n" + "=" * 40)
    print(f"[Test] Saved {saved} samples to: {output_dir}")
    print("[Test] Final Metrics (PIT-aligned, mean over utterances):")
    for k, v in results.items():
        print(f"  {k:8s}: {v:.4f}")
    print("=" * 40 + "\n")

    # extra hint if PESQ/STOI missing
    if math.isnan(results["PESQ"]):
        print("PESQ 是 NaN：你环境可能没装 pesq。可尝试：pip install pesq")
    if math.isnan(results["STOI"]):
        print("STOI 是 NaN：你环境可能没装 pystoi。可尝试：pip install pystoi")

    return results


def load_ckpt_into_model(model, model_path, device):
    print(f"Loading weights: {model_path}")
    ckpt = torch.load(model_path, map_location=device)

    # pick state dict
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    # remove "module." prefix if needed
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)
    print("Model weights loaded.")


def run_test(model_path, test_root, batch_size, sample_rate, device,
             output_dir="./test_results/train_convtasnet_20251216-225848", max_save_samples=20, ):
    model = Model(enc_dim=512, enc_kernel=16, enc_stride=8, bottleneck_channels=128, hidden_channels=256, num_blocks=8,
                  num_repeats=3, num_speakers=2).to(device)

    load_ckpt_into_model(model, model_path, device)

    _, _, test_loader = MakeDataLoader(root=test_root, batch_size=batch_size, num_workers=4, sample_rate=sample_rate,
                                       segment=4.0)

    return eval_test_full_metrics(model=model, test_loader=test_loader, sample_rate=sample_rate, device=device,
                                  output_dir=output_dir, max_save_samples=max_save_samples)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_test(model_path="../../checkpoints/train_convtasnet_20251216-225848/conv_tasnet_final.pt",
             test_root="Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min", batch_size=16, sample_rate=8000,
             device=device, output_dir="../../test_results/train_convtasnet_20251216-225848", max_save_samples=20)

# import torch
# import torchaudio
# import os
# from LossFunction.SI_SNR import pit_si_snr_loss
# from Model.Model_TasNet import Model
# from DataLoader import MakeDataLoader
#
#
# def eval_test_full(model, test_loader, loss_fn, sample_rate, device, output_dir, max_samples=20):
#     """
#     Run through the entire test set (or first N samples) after training finishes.
#     Save separated wav files for each test sample.
#     Also compute average test loss (e.g. negative SI-SNR).
#     """
#     model.eval()
#     os.makedirs(output_dir, exist_ok=True)
#
#     count = 0
#     total_loss = 0.0
#     num_batches = 0
#
#     with torch.no_grad():
#         for mix, targets in test_loader:
#             min_len_needed = 16
#             if mix.shape[-1] < min_len_needed:
#                 pad_len = min_len_needed - mix.shape[-1]
#                 mix = torch.nn.functional.pad(mix, (0, pad_len), mode="constant", value=0.0)
#
#                 # targets 同样 pad（否则对应不上）
#                 targets = torch.nn.functional.pad(targets, (0, pad_len), mode="constant", value=0.0)
#
#             mix = mix.to(device)  # [B, T]
#             targets = targets.to(device)  # [B, 2, T]
#
#             est_sources = model(mix.unsqueeze(1))  # [B, 2, T’]
#
#             min_len = min(est_sources.shape[-1], targets.shape[-1])
#             est_sources = est_sources[..., :min_len]  # [B, 2, L]
#             targets = targets[..., :min_len]  # [B, 2, L]
#             mix = mix[..., :min_len]  # [B, L]
#
#             batch_loss = loss_fn(est_sources, targets)
#             total_loss += batch_loss.item()
#             num_batches += 1
#
#             B = mix.size(0)
#             for i in range(B):
#                 if count >= max_samples:
#                     print("[Test] Reached max_samples limit.")
#                     avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
#                     print(f"[Test] Average test loss (−SI-SNR): {avg_loss:.4f} dB")
#                     return
#
#                 mix_i = mix[i].cpu()
#                 s1_ref = targets[i, 0].cpu()
#                 s2_ref = targets[i, 1].cpu()
#                 s1_est = est_sources[i, 0].cpu()
#                 s2_est = est_sources[i, 1].cpu()
#
#                 base = f"test_{count}"
#                 torchaudio.save(os.path.join(output_dir, f"{base}_mix.wav"),
#                                 mix_i.unsqueeze(0), sample_rate)
#                 torchaudio.save(os.path.join(output_dir, f"{base}_s1_ref.wav"),
#                                 s1_ref.unsqueeze(0), sample_rate)
#                 torchaudio.save(os.path.join(output_dir, f"{base}_s2_ref.wav"),
#                                 s2_ref.unsqueeze(0), sample_rate)
#                 torchaudio.save(os.path.join(output_dir, f"{base}_s1_est.wav"),
#                                 s1_est.unsqueeze(0), sample_rate)
#                 torchaudio.save(os.path.join(output_dir, f"{base}_s2_est.wav"),
#                                 s2_est.unsqueeze(0), sample_rate)
#                 count += 1
#
#     if num_batches > 0:
#         avg_loss = total_loss / num_batches
#     else:
#         avg_loss = float('nan')
#     print(f"[Test] Saved {count} test samples to: {output_dir}")
#     print(f"[Test] Average test loss (−SI-SNR): {avg_loss:.4f} dB")
#
#
# def run_test(model_path, test_root, batch_size, sample_rate, device):
#     model = Model(
#         enc_dim=512,
#         enc_kernel=16,
#         enc_stride=8,
#         bottleneck_channels=128,
#         hidden_channels=512,
#         num_blocks=8,
#         num_repeats=3,
#         num_speakers=2
#     ).to(device)
#
#     # 2. 加载训练好的权重
#     print(f"Loading weights: {model_path}")
#     ckpt = torch.load(model_path, map_location=device)
#     model.load_state_dict(ckpt["model_state_dict"])
#
#     # 3. 建立测试集 DataLoader
#     _,_, test_loader = MakeDataLoader(
#         root=test_root,
#         batch_size=batch_size,
#         num_workers=4,
#         sample_rate=sample_rate,
#         segment=4
#     )
#
#     # 4. 输出目录
#     output_dir = "./test_results"
#
#     # 5. 调用你给的 eval_test_full()
#     eval_test_full(
#         model=model,
#         test_loader=test_loader,
#         loss_fn=pit_si_snr_loss,  # ★ 重要：PIT loss
#         sample_rate=sample_rate,
#         device=device,
#         output_dir=output_dir,
#         max_samples=20
#     )
#
#
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     run_test(
#         model_path="checkpoints/train_convtasnet_20251209-193249/conv_tasnet_final.pt",  # 修改为你实际的 pt 文件
#         test_root="Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min",  # 你的数据集路径
#         batch_size=1,  # test 可以 batch=1
#         sample_rate=8000,  # 按你的数据集来的
#         device=device
#     )
