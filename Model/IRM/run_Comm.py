import os
import time
import torch
import numpy as np
from tqdm import tqdm

# Implementation note.
from DataLoader.CommDataLoader import MakeDataLoader as MakeCommLoader

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
# Ideal ratio masking (IRM) separation
# ==========================================
def compute_irm_score(mix, s1, s2, n_fft=512, hop=128):
    """Calculate the upper bound score for ideal ratio masking (IRM) (Oracle Performance)"""
    eps = 1e-8
    # Make sure window is on the correct device
    window = torch.hann_window(n_fft).to(mix.device)

    # 1. STFT
    X_complex = torch.stft(mix, n_fft, hop, window=window, return_complex=True)
    X_mag = X_complex.abs()

    S1_mag = torch.stft(s1, n_fft, hop, window=window, return_complex=True).abs()
    S2_mag = torch.stft(s2, n_fft, hop, window=window, return_complex=True).abs()

    # 2. Construct IRM Mask
    mask1 = S1_mag / (S1_mag + S2_mag + eps)
    mask2 = S2_mag / (S1_mag + S2_mag + eps)

    # 3. Apply Mask (Oracle Separation)
    S1_est_mag = X_mag * mask1
    S2_est_mag = X_mag * mask2

    # 4. iSTFT (Back to Time Domain) using original mixture phase
    phase = torch.angle(X_complex)

    S1_est_complex = S1_est_mag * torch.exp(1j * phase)
    S2_est_complex = S2_est_mag * torch.exp(1j * phase)

    # Recover time domain signals
    s1_recon = torch.istft(S1_est_complex, n_fft, hop, window=window, length=mix.shape[-1])
    s2_recon = torch.istft(S2_est_complex, n_fft, hop, window=window, length=mix.shape[-1])

    return s1_recon, s2_recon


# ==========================================
# Assessment process
# ==========================================
def evaluate_baselines(loader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "Mix_SI-SNR": [], "Mix_SNR": [], "Mix_BER": [],
        "IRM_SI-SNR": [], "IRM_SI-SNRi": [], "IRM_SNR": [], "IRM_BER": []
    }

    print("Running Oracle (IRM) and Mixture Baselines on Comm0-2Dataset...")

    with torch.no_grad():
        for mix, targets in tqdm(loader, desc="Evaluating Baselines"):
            mix = mix.to(device)
            s1 = targets[:, 0].to(device)
            s2 = targets[:, 1].to(device)

            # ===================================================
            # Baseline 1: Input Mixture (directly treat the mixed signal as output, test the lower limit)
            # ===================================================
            mix_sisnr = (si_snr(mix, s1) + si_snr(mix, s2)) / 2
            mix_snr = (snr(mix, s1) + snr(mix, s2)) / 2
            mix_ber = (compute_ber(mix, s1) + compute_ber(mix, s2)) / 2

            metrics["Mix_SI-SNR"].extend(mix_sisnr.detach().cpu().tolist())
            metrics["Mix_SNR"].extend(mix_snr.detach().cpu().tolist())
            metrics["Mix_BER"].append(mix_ber)

            # ===================================================
            # Baseline 2: IRM Oracle (absolute theoretical upper limit of frequency domain mask)
            # ===================================================
            s1_irm, s2_irm = compute_irm_score(mix, s1, s2)

            # Remove DC operation (consistent with the processing during the local neural network testing)
            s1_irm = s1_irm - s1_irm.mean(dim=-1, keepdim=True)
            s2_irm = s2_irm - s2_irm.mean(dim=-1, keepdim=True)

            irm_sisnr = (si_snr(s1_irm, s1) + si_snr(s2_irm, s2)) / 2
            irm_snr = (snr(s1_irm, s1) + snr(s2_irm, s2)) / 2

            # SI-SNRi (improvement) = restored SI-SNR - mixed-signal SI-SNR.
            irm_sisnri = irm_sisnr - mix_sisnr

            irm_ber = (compute_ber(s1_irm, s1) + compute_ber(s2_irm, s2)) / 2

            metrics["IRM_SI-SNR"].extend(irm_sisnr.detach().cpu().tolist())
            metrics["IRM_SI-SNRi"].extend(irm_sisnri.detach().cpu().tolist())
            metrics["IRM_SNR"].extend(irm_snr.detach().cpu().tolist())
            metrics["IRM_BER"].append(irm_ber)

    # Summary calculation mean
    def nanmean(x):
        return float(np.nanmean(x)) if len(x) > 0 else float("nan")

    results = {k: nanmean(v) for k, v in metrics.items()}

    # Print to console
    print("\n" + "=" * 50)
    print(f"[Comm0-2Dataset Baselines Final Results]")
    print(f"--- 1. Input Mixture (理论下限 / 不做处理) ---")
    print(f"  SI-SNR  : {results['Mix_SI-SNR']:.4f} dB")
    print(f"  SNR     : {results['Mix_SNR']:.4f} dB")
    print(f"  BER     : {results['Mix_BER']:.6f} ({results['Mix_BER'] * 100:.4f}%) - 误比特率")

    print(f"\n--- 2. IRM Oracle (STFT 频域分离理论上限) ---")
    print(f"  SI-SNR  : {results['IRM_SI-SNR']:.4f} dB")
    print(f"  SI-SNRi : {results['IRM_SI-SNRi']:.4f} dB")
    print(f"  SNR     : {results['IRM_SNR']:.4f} dB")
    print(f"  BER     : {results['IRM_BER']:.6f} ({results['IRM_BER'] * 100:.4f}%) - 误比特率")
    print("=" * 50 + "\n")

    # Write to TXT file
    ts = time.strftime("%Y%m%d-%H%M%S")
    txt_path = os.path.join(output_dir, f"oracle_baselines_comm_{ts}.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Oracle (IRM) & Mixture Baselines on Comm0-2Dataset\n")
        f.write(f"Evaluated utterances: {len(metrics['Mix_SI-SNR'])}\n")
        f.write("-" * 50 + "\n")
        f.write("1. Input Mixture (Lower Bound - No Processing)\n")
        f.write(f"  SI-SNR  : {results['Mix_SI-SNR']:.4f} dB\n")
        f.write(f"  SNR     : {results['Mix_SNR']:.4f} dB\n")
        f.write(f"  BER     : {results['Mix_BER']:.6f} ({results['Mix_BER'] * 100:.4f}%)\n")

        f.write("\n2. IRM Oracle (Upper Bound for STFT based Masking)\n")
        f.write(f"  SI-SNR  : {results['IRM_SI-SNR']:.4f} dB\n")
        f.write(f"  SI-SNRi : {results['IRM_SI-SNRi']:.4f} dB\n")
        f.write(f"  SNR     : {results['IRM_SNR']:.4f} dB\n")
        f.write(f"  BER     : {results['IRM_BER']:.6f} ({results['IRM_BER'] * 100:.4f}%)\n")
        f.write("-" * 50 + "\n")

    print(f"Metrics saved to: {txt_path}")
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset path
    root = "../../Dataset/Comm0-2Dataset"

    # Results save path
    output_dir = "../../Results/Oracle_Baselines_Comm"

    print(f"Device: {device}")

    # Load test set
    _, _, test_loader = MakeCommLoader(root=root, batch_size=32, num_workers=4, segment=32000)

    # Run tests
    evaluate_baselines(test_loader, device, output_dir)