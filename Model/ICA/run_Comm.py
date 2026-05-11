import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import FastICA, NMF

# Implementation note.
from DataLoader.CommDataLoader import MakeDataLoader

# Implementation note.
from Evaluation.SI_SNR import si_snr
from Evaluation.SNR import snr


# ==========================================
# BER calculation.
# ==========================================
def compute_ber(est_waveform, ref_waveform):
    """BPSK"""
    est_bits = (est_waveform > 0).float()
    ref_bits = (ref_waveform > 0).float()
    errors = torch.sum(torch.abs(est_bits - ref_bits))
    total_bits = torch.numel(ref_bits)
    return (errors / total_bits).item()


# ==========================================
# ( PIT )
# ==========================================
def evaluate_metrics_pit(est_s1, est_s2, s1, s2, mix):
    """SI-SNR, SI-SNRi, SNR, BER ICA/NMF (PIT)"""
    # Tensor ( 1D)
    e1 = torch.from_numpy(est_s1).float()
    e2 = torch.from_numpy(est_s2).float()
    t1 = torch.from_numpy(s1).float()
    t2 = torch.from_numpy(s2).float()
    m = torch.from_numpy(mix).float()

    # 1. PIT ( SI-SNR)
    sisnr_11 = si_snr(e1, t1)
    sisnr_22 = si_snr(e2, t2)
    sisnr_12 = si_snr(e1, t2)
    sisnr_21 = si_snr(e2, t1)

    mean_sisnr_A = (sisnr_11 + sisnr_22) / 2
    mean_sisnr_B = (sisnr_12 + sisnr_21) / 2

    # Implementation note.
    if mean_sisnr_A > mean_sisnr_B:
        best_e1, best_e2 = e1, e2
        best_sisnr = mean_sisnr_A
    else:
        best_e1, best_e2 = e2, e1
        best_sisnr = mean_sisnr_B

    # 2. SNR
    snr_val = (snr(best_e1, t1) + snr(best_e2, t2)) / 2

    # 3. SI-SNRi (Improvement)
    base_sisnr_1 = si_snr(m, t1)
    base_sisnr_2 = si_snr(m, t2)
    mean_base_sisnr = (base_sisnr_1 + base_sisnr_2) / 2
    sisnri = best_sisnr - mean_base_sisnr

    # 4. BER calculation.
    ber_val = (compute_ber(best_e1, t1) + compute_ber(best_e2, t2)) / 2

    return best_sisnr.item(), sisnri.item(), snr_val.item(), ber_val


# ==========================================
# Implementation note.
# ==========================================
def run_statistical_separation(mix_wav, method='nmf', n_fft=512, hop=256):
    """(ICA / NMF)"""
    # 1. STFT
    spec_complex = torch.stft(
        torch.from_numpy(mix_wav).unsqueeze(0),
        n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=torch.hann_window(n_fft),
        return_complex=True
    ).squeeze(0).numpy()

    mag = np.abs(spec_complex)
    phase = np.angle(spec_complex)
    X = mag.T  # (Samples, Features)

    # Step 2.
    if method == 'ica':
        model = FastICA(n_components=2, max_iter=800, tol=1e-2, random_state=42)
    elif method == 'nmf':
        model = NMF(n_components=2, init='random', random_state=42, max_iter=500)
    else:
        raise ValueError("Unknown method")

    try:
        W = model.fit_transform(X)
        H = model.components_

        # 3. + Soft Masking
        est_mag_1 = np.outer(W[:, 0], H[0, :]).T
        est_mag_2 = np.outer(W[:, 1], H[1, :]).T

        sum_mag = est_mag_1 + est_mag_2 + 1e-8
        mask_1 = est_mag_1 / sum_mag
        mask_2 = est_mag_2 / sum_mag

        Y1_mag = mag * mask_1
        Y2_mag = mag * mask_2

        # 4. iSTFT
        Y1_complex = Y1_mag * np.exp(1j * phase)
        Y2_complex = Y2_mag * np.exp(1j * phase)

        y1_wav = torch.istft(
            torch.from_numpy(Y1_complex).unsqueeze(0),
            n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=torch.hann_window(n_fft),
            length=len(mix_wav)
        ).squeeze(0).numpy()

        y2_wav = torch.istft(
            torch.from_numpy(Y2_complex).unsqueeze(0),
            n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=torch.hann_window(n_fft),
            length=len(mix_wav)
        ).squeeze(0).numpy()

        return y1_wav, y2_wav

    except Exception as e:
        # Implementation note.
        return mix_wav, mix_wav


def main():
    root = "../../Dataset/Comm0-2Dataset"
    segment_points = 32000
    max_samples = 50  # CPU

    # Implementation note.
    output_dir = "../../Results/Statistical_Baselines_Comm"
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    txt_path = os.path.join(output_dir, f"metrics_{ts}.txt")

    print(f"加载 Comm0-2Dataset 测试集 (Segment={segment_points})...")
    _, _, test_loader = MakeDataLoader(root=root, batch_size=1, num_workers=4, segment=segment_points)

    methods = ['ica', 'nmf']

    # Implementation note.
    all_results = {}

    for method in methods:
        print(f"\n==============================================")
        print(f" 正在运行 {method.upper()} 基准测试 (Max samples: {max_samples})")
        print(f"==============================================")

        metrics = {"SI-SNR": [], "SI-SNRi": [], "SNR": [], "BER": []}
        count = 0
        pbar = tqdm(total=max_samples)

        for mix, targets in test_loader:
            if count >= max_samples:
                break

            mix_np = mix.squeeze().numpy()
            s1_np = targets[:, 0].squeeze().numpy()
            s2_np = targets[:, 1].squeeze().numpy()

            # Implementation note.
            est_s1, est_s2 = run_statistical_separation(mix_np, method=method)

            # ( PIT )
            sisnr_val, sisnri_val, snr_val, ber_val = evaluate_metrics_pit(est_s1, est_s2, s1_np, s2_np, mix_np)

            metrics["SI-SNR"].append(sisnr_val)
            metrics["SI-SNRi"].append(sisnri_val)
            metrics["SNR"].append(snr_val)
            metrics["BER"].append(ber_val)

            count += 1
            pbar.update(1)
            pbar.set_postfix({
                "SI-SNR": f"{np.mean(metrics['SI-SNR']):.2f}",
                "BER": f"{np.mean(metrics['BER']):.4f}"
            })

        pbar.close()

        # Implementation note.
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        all_results[method] = avg_metrics

        print(f"\n[{method.upper()}] 最终平均成绩:")
        for k, v in avg_metrics.items():
            if k == "BER":
                print(f"  {k:8s}: {v:.6f} ({v * 100:.4f}%) - 误比特率")
            else:
                print(f"  {k:8s}: {v:.4f} dB")

    # ==========================================
    # Write the TXT report.
    # ==========================================
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Statistical Baselines on Comm0-2Dataset\n")
        f.write(f"Evaluated samples: {max_samples}\n")
        f.write(f"Segment points: {segment_points}\n")
        f.write("-" * 40 + "\n")

        for method in methods:
            f.write(f"Method: {method.upper()}\n")
            avg_metrics = all_results[method]
            for k, v in avg_metrics.items():
                if k == "BER":
                    f.write(f"  {k:8s}: {v:.6f} ({v * 100:.4f}%)\n")
                else:
                    f.write(f"  {k:8s}: {v:.4f} dB\n")
            f.write("-" * 40 + "\n")

    print(f"\n所有评估结果已保存至: {txt_path}")


if __name__ == "__main__":
    main()