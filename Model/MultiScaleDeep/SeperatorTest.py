import os
import math
import time
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from Model.MultiScaleDeep.Model_TasNet_MultiScale import Model
from DataLoader.WSJDataLoader import MakeDataLoader
from Evaluation.SI_SNR import si_snr
from Evaluation.SI_SNR import si_snr_improvement
from Evaluation.SI_SNR import pit_align_by_sisnr
from Evaluation.SNR import snr
from Evaluation.PESQ import pesqTest
from Evaluation.STOI import stoiTest


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
            est_aligned = est_aligned - est_aligned.mean(dim=-1, keepdim=True)

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
                    p_list.append(pesqTest(e.astype(np.float32), r.astype(np.float32), sample_rate))
                    s_list.append(stoiTest(e.astype(np.float32), r.astype(np.float32), sample_rate))
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
    # ========= 写入 outdir 的 txt =========
    ts = time.strftime("%Y%m%d-%H%M%S")
    txt_path = os.path.join(output_dir, f"final_metrics_{ts}.txt")

    num_utts = len(all_metrics["SI-SNR"])  # 评估了多少条 utterance（逐条 append 的）
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Final Metrics (PIT-aligned, mean over utterances)\n")
        f.write(f"Saved wav samples: {saved}\n")
        f.write(f"Evaluated utterances: {num_utts}\n")
        f.write(f"Sample rate: {sample_rate}\n")
        f.write("-" * 40 + "\n")
        for k, v in results.items():
            # NaN 也能写进去，别让它报错
            if isinstance(v, float) and math.isnan(v):
                f.write(f"{k:8s}: NaN\n")
            else:
                f.write(f"{k:8s}: {v:.4f}\n")
        f.write("-" * 40 + "\n")

    print(f"Metrics saved to: {txt_path}")

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
             output_dir, max_save_samples=20):
    model = Model(enc_dim=256, enc_kernel=(11, 21, 31), enc_stride=8, bottleneck_channels=128, hidden_channels=256,
                  num_blocks=8, num_repeats=3, num_speakers=2).to(device)

    load_ckpt_into_model(model, model_path, device)

    _, _, test_loader = MakeDataLoader(root=test_root, batch_size=batch_size, num_workers=4, sample_rate=sample_rate,
                                       segment=4.0)

    return eval_test_full_metrics(model=model, test_loader=test_loader, sample_rate=sample_rate, device=device,
                                  output_dir=output_dir, max_save_samples=max_save_samples)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_test(model_path="../../checkpoints/train_MultiDeep_20260204-142829/conv_tasnet_final.pt",
             test_root="../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min", batch_size=32, sample_rate=8000,
             device=device, output_dir="../../Results/train_MultiDeep_20260204-142829", max_save_samples=20)

