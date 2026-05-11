import os
import time
import math
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# ========================================================
# PyTorch 2.6 torch.load
# ========================================================
_original_load = torch.load
def _patched_load(*args, **kwargs):
    # weights_only False
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = _patched_load
# ========================================================

# ConvTasNet DPRNNTasNet
from asteroid.models import DPRNNTasNet

# DataLoader
from DataLoader.WSJDataLoader import MakeDataLoader as MakeWSJLoader
from Evaluation.SI_SNR import si_snr, si_snr_improvement, pit_align_by_sisnr
from Evaluation.SNR import snr

try:
    from Evaluation.PESQ import pesqTest
    from Evaluation.STOI import stoiTest
except ImportError:
    pass



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
            pass
        elif ref.size(2) == 2:
            ref = ref.permute(0, 2, 1)
        elif ref.size(0) == 2:
            ref = ref.permute(1, 0, 2)
        else:
            raise ValueError(f"Unsupported ref shape: {tuple(ref.shape)}")
    else:
        raise ValueError(f"Unsupported ref shape: {tuple(ref.shape)}")
    return ref.contiguous()


def eval_asteroid_model(model_name, model, loader, device, sample_rate, output_dir, max_save_samples=20):
    """Asteroid"""
    print(f"\n==============================================")
    print(f" 正在运行 Asteroid [{model_name}] 基准测试")
    print(f"==============================================")

    model.eval()
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    all_metrics = {
        "SI-SNR": [],
        "SI-SNRi": [],
        "SNR": [],
        "PESQ": [],
        "STOI": [],
    }

    saved = 0

    with torch.no_grad():
        for mix, targets in tqdm(loader, desc=f"Evaluating {model_name}"):
            mix = ensure_mix_bt(mix)
            targets = ensure_ref_b2t(targets)

            mix = mix.to(device)
            targets = targets.to(device)

            # Asteroid (B, T)
            est = model(mix)

            # (B, 1, Spk, T)
            if est.dim() == 4 and est.size(1) == 1:
                est = est.squeeze(1)

            # ( Asteroid )
            min_len = min(est.shape[-1], targets.shape[-1], mix.shape[-1])
            est = est[..., :min_len]
            targets = targets[..., :min_len]
            mix = mix[..., :min_len]

            # 1. PIT alignment.
            est_aligned, perm, sisnr_best = pit_align_by_sisnr(est, targets)

            # Implementation note.
            est_aligned = est_aligned - est_aligned.mean(dim=-1, keepdim=True)

            # Step 2.
            si_snr_vals = sisnr_best
            si_snri_vals = si_snr_improvement(est_aligned, targets, mix)
            snr_vals = 0.5 * (snr(est_aligned[:, 0, :], targets[:, 0, :]) + snr(est_aligned[:, 1, :], targets[:, 1, :]))

            all_metrics["SI-SNR"].extend(si_snr_vals.detach().cpu().tolist())
            all_metrics["SI-SNRi"].extend(si_snri_vals.detach().cpu().tolist())
            all_metrics["SNR"].extend(snr_vals.detach().cpu().tolist())

            # 3. CPU PESQ STOI
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

            # Step 4.
            B = mix.size(0)
            for i in range(B):
                if saved >= max_save_samples:
                    break

                base = f"{model_name}_{saved:04d}_perm{perm[i][0]}{perm[i][1]}"
                mix_i = mix[i].detach().cpu()
                s1_ref = targets[i, 0].detach().cpu()
                s2_ref = targets[i, 1].detach().cpu()
                s1_est = est_aligned[i, 0].detach().cpu()
                s2_est = est_aligned[i, 1].detach().cpu()

                torchaudio.save(os.path.join(model_output_dir, f"{base}_mix.wav"), mix_i.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(model_output_dir, f"{base}_s1_ref.wav"), s1_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(model_output_dir, f"{base}_s2_ref.wav"), s2_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(model_output_dir, f"{base}_s1_est.wav"), s1_est.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(model_output_dir, f"{base}_s2_est.wav"), s2_est.unsqueeze(0), sample_rate)

                saved += 1

    # Summary results
    def nanmean(x):
        x = np.array(x, dtype=np.float64)
        return float(np.nanmean(x)) if x.size else float("nan")

    results = {k: nanmean(v) for k, v in all_metrics.items()}

    print(f"\n[{model_name}] 最终平均成绩 (PIT-aligned):")
    for k, v in results.items():
        print(f"  {k:8s}: {v:.4f}")

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = "../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"
    sample_rate = 8000

    # Implementation note.
    output_dir = "../../Results/Asteroid_Baselines_WSJ"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading WSJ0-2Mix Test Set...")
    _, _, test_loader = MakeWSJLoader(root=root, batch_size=2, num_workers=4, segment=4.0, sample_rate=sample_rate)

    all_results = {}

    # ===============================================
    # Asteroid DPRNNTasNet
    # ===============================================
    print("\nLoading DPRNNTasNet from Asteroid Hub...")
    # HuggingFace
    model_dprnn = DPRNNTasNet.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean").to(device)
    res_dprnn = eval_asteroid_model("DPRNNTasNet", model_dprnn, test_loader, device, sample_rate, output_dir)
    all_results["DPRNNTasNet"] = res_dprnn

    # ===============================================
    # Implementation note.
    # ===============================================
    ts = time.strftime("%Y%m%d-%H%M%S")
    txt_path = os.path.join(output_dir, f"asteroid_dprnn_baseline_wsj_{ts}.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Asteroid Pre-trained DPRNN Baseline on WSJ0-2Mix\n")
        f.write(f"Sample rate: {sample_rate} Hz\n")
        f.write("-" * 40 + "\n")

        for model_name, metrics in all_results.items():
            f.write(f"Model: {model_name}\n")
            for k, v in metrics.items():
                if isinstance(v, float) and math.isnan(v):
                    f.write(f"  {k:8s}: NaN\n")
                else:
                    f.write(f"  {k:8s}: {v:.4f}\n")
            f.write("-" * 40 + "\n")

    print(f"\n评估结果已保存至: {txt_path}")


if __name__ == "__main__":
    main()