import os
import math
import time
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from Model.MultiScale.Model_TasNet_MultiScale import Model

# 数据加载
from DataLoader.CommDataLoader import MakeDataLoader as MakeCommLoader

try:
    from DataLoader.WSJDataLoader import MakeDataLoader as MakeWSJLoader
except ImportError:
    pass

# 指标计算
from Evaluation.SI_SNR import si_snr
from Evaluation.SI_SNR import si_snr_improvement
from Evaluation.SI_SNR import pit_align_by_sisnr
from Evaluation.SNR import snr

try:
    from Evaluation.PESQ import pesqTest
    from Evaluation.STOI import stoiTest
except ImportError:
    pass


# ==========================================
# 新增：通信信号误码率 (BER) 计算函数
# ==========================================
def compute_ber(est_waveform, ref_waveform):
    """
    计算误码率的简易占位函数 (基于零点交叉的极性判决，适用于 BPSK)。
    如果你使用的是更复杂的调制(QPSK, QAM)，请替换为你对应的解调判决逻辑。
    """
    est_bits = (est_waveform > 0).float()
    ref_bits = (ref_waveform > 0).float()
    errors = torch.sum(torch.abs(est_bits - ref_bits))
    total_bits = torch.numel(ref_bits)
    return (errors / total_bits).item()


def ensure_mix_bt(mix):
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)
    elif mix.dim() == 3:
        if mix.size(1) == 1:
            mix = mix.squeeze(1)
        elif mix.size(0) == 1 and mix.size(2) > mix.size(1):
            mix = mix.squeeze(0)
        else:
            mix = mix.squeeze()
            if mix.dim() == 1:
                mix = mix.unsqueeze(0)
    elif mix.dim() != 2:
        raise ValueError(f"Unsupported mix shape: {tuple(mix.shape)}")
    return mix.contiguous()


def ensure_ref_b2t(ref):
    if ref.dim() == 2:
        if ref.size(0) == 2:
            ref = ref.unsqueeze(0)
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


def eval_test_full_metrics(model, test_loader, sample_rate, device, output_dir, max_save_samples, min_len_needed=16,
                           dataset_type="comm"):
    """
    - 对 test set 跑完整个 epoch
    - dataset_type: "comm" (算BER) 或 "speech" (算PESQ, STOI)
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # 1. 动态初始化指标字典
    all_metrics = {
        "SI-SNR": [],
        "SI-SNRi": [],
        "SNR": [],
    }
    if dataset_type == "speech":
        all_metrics["PESQ"] = []
        all_metrics["STOI"] = []
    elif dataset_type == "comm":
        all_metrics["BER"] = []

    saved = 0

    with torch.no_grad():
        for mix, targets in tqdm(test_loader, desc=f"Evaluating ({dataset_type.upper()})"):
            mix = ensure_mix_bt(mix)
            targets = ensure_ref_b2t(targets)

            # pad to min_len_needed (Conv kernel safety)
            if mix.shape[-1] < min_len_needed:
                pad_len = min_len_needed - mix.shape[-1]
                mix = torch.nn.functional.pad(mix, (0, pad_len))
                targets = torch.nn.functional.pad(targets, (0, pad_len))

            mix = mix.to(device)  # [B,T]
            targets = targets.to(device)  # [B,2,T]

            # forward
            est = model(mix.unsqueeze(1))
            if est.dim() == 3 and est.size(1) != 2:
                raise RuntimeError(f"Model output unexpected: {tuple(est.shape)}")

            # crop to same length
            min_len = min(est.shape[-1], targets.shape[-1], mix.shape[-1])
            est = est[..., :min_len]
            targets = targets[..., :min_len]
            mix = mix[..., :min_len]

            # PIT align
            est_aligned, perm, sisnr_best = pit_align_by_sisnr(est, targets)  # sisnr_best: [B]

            # 公共指标：SI-SNR, SI-SNRi, SNR
            si_snr_vals = sisnr_best
            si_snri_vals = si_snr_improvement(est_aligned, targets, mix)
            snr_vals = 0.5 * (snr(est_aligned[:, 0, :], targets[:, 0, :]) + snr(est_aligned[:, 1, :], targets[:, 1, :]))

            all_metrics["SI-SNR"].extend(si_snr_vals.detach().cpu().tolist())
            all_metrics["SI-SNRi"].extend(si_snri_vals.detach().cpu().tolist())
            all_metrics["SNR"].extend(snr_vals.detach().cpu().tolist())

            # 2. 差异化指标计算
            if dataset_type == "speech":
                # PESQ / STOI：逐条算（CPU + numpy）
                pesq_vals = []
                stoi_vals = []
                est_cpu = est_aligned.detach().cpu().float().numpy()
                ref_cpu = targets.detach().cpu().float().numpy()

                for b in range(est_cpu.shape[0]):
                    p_list, s_list = [], []
                    for spk in range(2):
                        e = est_cpu[b, spk]
                        r = ref_cpu[b, spk]
                        p_list.append(pesqTest(e.astype(np.float32), r.astype(np.float32), sample_rate))
                        s_list.append(stoiTest(e.astype(np.float32), r.astype(np.float32), sample_rate))
                    pesq_vals.append(float(np.nanmean(p_list)))
                    stoi_vals.append(float(np.nanmean(s_list)))

                all_metrics["PESQ"].extend(pesq_vals)
                all_metrics["STOI"].extend(stoi_vals)

            elif dataset_type == "comm":
                # BER
                ber_vals = []
                for b in range(est_aligned.shape[0]):
                    b_err = 0.5 * (
                            compute_ber(est_aligned[b, 0], targets[b, 0]) +
                            compute_ber(est_aligned[b, 1], targets[b, 1])
                    )
                    ber_vals.append(b_err)
                all_metrics["BER"].extend(ber_vals)

            # 3. Save wav (first N)
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
    print(f"[Test] Final Metrics ({dataset_type.upper()} Mode, PIT-aligned):")
    for k, v in results.items():
        if k == "BER":
            print(f"  {k:8s}: {v:.6f} ({v * 100:.4f}%)")
        else:
            print(f"  {k:8s}: {v:.4f}")
    print("=" * 40 + "\n")

    # extra hint if PESQ/STOI missing (仅语音模式提示)
    if dataset_type == "speech":
        if math.isnan(results.get("PESQ", 0)):
            print("PESQ 是 NaN：你环境可能没装 pesq。可尝试：pip install pesq")
        if math.isnan(results.get("STOI", 0)):
            print("STOI 是 NaN：你环境可能没装 pystoi。可尝试：pip install pystoi")

    # ========= 写入 outdir 的 txt =========
    ts = time.strftime("%Y%m%d-%H%M%S")
    txt_path = os.path.join(output_dir, f"final_metrics_{dataset_type}_{ts}.txt")

    num_utts = len(all_metrics["SI-SNR"])
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Final Metrics (Mode: {dataset_type}, PIT-aligned, mean over utterances)\n")
        f.write(f"Saved wav samples: {saved}\n")
        f.write(f"Evaluated utterances: {num_utts}\n")
        f.write(f"Sample rate: {sample_rate}\n")
        f.write("-" * 40 + "\n")
        for k, v in results.items():
            if isinstance(v, float) and math.isnan(v):
                f.write(f"{k:8s}: NaN\n")
            elif k == "BER":
                f.write(f"{k:8s}: {v:.6f} ({v * 100:.4f}%)\n")
            else:
                f.write(f"{k:8s}: {v:.4f}\n")
        f.write("-" * 40 + "\n")

    print(f"Metrics saved to: {txt_path}")

    return results


def load_ckpt_into_model(model, model_path, device):
    print(f"Loading weights: {model_path}")
    ckpt = torch.load(model_path, map_location=device)

    # pick state dict (优先加载 ema_state，同时兼容各类键名)
    if isinstance(ckpt, dict):
        if "ema_state" in ckpt:
            print("Found 'ema_state'. Loading EMA weights...")
            sd = ckpt["ema_state"]
        elif "model_state" in ckpt:
            print("Found 'model_state'. Loading best model weights...")
            sd = ckpt["model_state"]
        elif "model_state_dict" in ckpt:
            print("Found 'model_state_dict'...")
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            print("Found 'state_dict'...")
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    # remove "module." prefix if needed
    if isinstance(sd, dict):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        # 剔除 EMA 专有的计数器变量
        if "n_averaged" in sd:
            sd.pop("n_averaged")

    model.load_state_dict(sd, strict=True)
    print("Model weights loaded successfully.")


def run_test(model_path, test_root, batch_size, sample_rate, device,
             output_dir, max_save_samples=20, dataset_type="comm"):
    # 你的多尺度模型定义
    model = Model(enc_dim=516, enc_kernel=[8, 16, 24], enc_stride=8, bottleneck_channels=128, hidden_channels=256,
                  num_blocks=8, num_repeats=3, num_speakers=2).to(device)

    load_ckpt_into_model(model, model_path, device)

    # 动态切换 Dataloader
    if dataset_type == "comm":
        _, _, test_loader = MakeCommLoader(root=test_root, batch_size=batch_size, num_workers=8, segment=32000)
    elif dataset_type == "speech":
        # 语音用秒为单位 4.0，并且需要传 sample_rate
        _, _, test_loader = MakeWSJLoader(root=test_root, batch_size=batch_size, num_workers=8, segment=4.0,
                                          sample_rate=sample_rate)
    else:
        raise ValueError("dataset_type 必须是 'comm' 或 'speech'")

    return eval_test_full_metrics(model=model, test_loader=test_loader, sample_rate=sample_rate, device=device,
                                  output_dir=output_dir, max_save_samples=max_save_samples, dataset_type=dataset_type)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==== 当你需要跑 Comm 数据集时，按照如下配置 ====
    run_test(
        model_path="../../checkpoints/train_multiscale_comm_20260214-130236/best_model.pt",
        test_root="../../Dataset/CommDataset",
        batch_size=128,
        sample_rate=100000,
        device=device,
        output_dir="../../Results/train_multiscale_comm_20260214-130236",
        max_save_samples=20,
        dataset_type="comm"  # <-- 关键：切换为通信模式，只计算 BER，不计算 PESQ/STOI
    )

    # ==== 当你需要跑 语音(WSJ) 数据集时，取消下方注释并修改路径 ====
    """
    run_test(
        model_path="../../checkpoints/your_speech_model.pt",
        test_root="../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min", 
        batch_size=16, 
        sample_rate=8000,
        device=device, 
        output_dir="../../Results/train_multiscale_speech", 
        max_save_samples=20,
        dataset_type="speech"  # <-- 关键：切换为语音模式，计算 PESQ/STOI，不计算 BER
    )
    """