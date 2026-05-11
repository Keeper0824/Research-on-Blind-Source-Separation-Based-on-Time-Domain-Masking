import os
import time
import math
import torch
import numpy as np
from tqdm import tqdm

# Introduce the local data loader
from DataLoader.WSJDataLoader import MakeDataLoader as MakeWSJLoader

# Introducing indicator calculations
from Evaluation.SI_SNR import si_snr, si_snr_improvement
from Evaluation.SNR import snr
from Evaluation.PESQ import pesqTest
from Evaluation.STOI import stoiTest


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
def evaluate_baselines(loader, device, output_dir, sample_rate=8000):
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "Mix_SI-SNR": [], "Mix_SNR": [], "Mix_PESQ": [], "Mix_STOI": [],
        "IRM_SI-SNR": [], "IRM_SI-SNRi": [], "IRM_SNR": [], "IRM_PESQ": [], "IRM_STOI": []
    }

    print("Running Oracle (IRM) and Mixture Baselines on WSJ0-2Mix Dataset...")

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

            metrics["Mix_SI-SNR"].extend(mix_sisnr.detach().cpu().tolist())
            metrics["Mix_SNR"].extend(mix_snr.detach().cpu().tolist())

            # ===================================================
            # Baseline 2: IRM Oracle (absolute theoretical upper limit of frequency domain mask)
            # ===================================================
            s1_irm, s2_irm = compute_irm_score(mix, s1, s2)

            # Implementation note.
            s1_irm = s1_irm - s1_irm.mean(dim=-1, keepdim=True)
            s2_irm = s2_irm - s2_irm.mean(dim=-1, keepdim=True)

            irm_sisnr = (si_snr(s1_irm, s1) + si_snr(s2_irm, s2)) / 2
            irm_snr = (snr(s1_irm, s1) + snr(s2_irm, s2)) / 2

            # SI-SNRi (improvement) = restored SI-SNR - mixed-signal SI-SNR.
            irm_sisnri = irm_sisnr - mix_sisnr

            metrics["IRM_SI-SNR"].extend(irm_sisnr.detach().cpu().tolist())
            metrics["IRM_SI-SNRi"].extend(irm_sisnri.detach().cpu().tolist())
            metrics["IRM_SNR"].extend(irm_snr.detach().cpu().tolist())

            # ===================================================
            # CPU side calculation (PESQ / STOI)
            # ===================================================
            mix_np = mix.detach().cpu().numpy()
            s1_np = s1.detach().cpu().numpy()
            s2_np = s2.detach().cpu().numpy()
            s1_irm_np = s1_irm.detach().cpu().numpy()
            s2_irm_np = s2_irm.detach().cpu().numpy()

            B = mix_np.shape[0]
            for b in range(B):
                # 1. Calculate PESQ and STOI of Mixture
                p_mix_1 = pesqTest(mix_np[b], s1_np[b], sample_rate)
                p_mix_2 = pesqTest(mix_np[b], s2_np[b], sample_rate)
                metrics["Mix_PESQ"].append(float(np.nanmean([p_mix_1, p_mix_2])))

                s_mix_1 = stoiTest(mix_np[b], s1_np[b], sample_rate)
                s_mix_2 = stoiTest(mix_np[b], s2_np[b], sample_rate)
                metrics["Mix_STOI"].append(float(np.nanmean([s_mix_1, s_mix_2])))

                # 2. Calculate PESQ and STOI of IRM
                p_irm_1 = pesqTest(s1_irm_np[b], s1_np[b], sample_rate)
                p_irm_2 = pesqTest(s2_irm_np[b], s2_np[b], sample_rate)
                metrics["IRM_PESQ"].append(float(np.nanmean([p_irm_1, p_irm_2])))

                s_irm_1 = stoiTest(s1_irm_np[b], s1_np[b], sample_rate)
                s_irm_2 = stoiTest(s2_irm_np[b], s2_np[b], sample_rate)
                metrics["IRM_STOI"].append(float(np.nanmean([s_irm_1, s_irm_2])))

    # Summarize and calculate the mean (to prevent errors when nan is included)
    def nanmean(x):
        return float(np.nanmean(x)) if len(x) > 0 else float("nan")

    results = {k: nanmean(v) for k, v in metrics.items()}

    # Print to console
    print("\n" + "=" * 50)
    print(f"[WSJ0-2Mix Baselines Final Results]")
    print(f"--- 1. Input Mixture (理论下限 / 不做处理) ---")
    print(f"  SI-SNR  : {results['Mix_SI-SNR']:.4f} dB")
    print(f"  SNR     : {results['Mix_SNR']:.4f} dB")
    print(f"  PESQ    : {results['Mix_PESQ']:.4f}")
    print(f"  STOI    : {results['Mix_STOI']:.4f}")

    print(f"\n--- 2. IRM Oracle (STFT 频域分离理论上限) ---")
    print(f"  SI-SNR  : {results['IRM_SI-SNR']:.4f} dB")
    print(f"  SI-SNRi : {results['IRM_SI-SNRi']:.4f} dB")
    print(f"  SNR     : {results['IRM_SNR']:.4f} dB")
    print(f"  PESQ    : {results['IRM_PESQ']:.4f}")
    print(f"  STOI    : {results['IRM_STOI']:.4f}")
    print("=" * 50 + "\n")

    # Write to TXT file
    ts = time.strftime("%Y%m%d-%H%M%S")
    txt_path = os.path.join(output_dir, f"oracle_baselines_wsj_{ts}.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Oracle (IRM) & Mixture Baselines on WSJ0-2Mix Dataset\n")
        f.write(f"Evaluated utterances: {len(metrics['Mix_SI-SNR'])}\n")
        f.write(f"Sample Rate: {sample_rate} Hz\n")
        f.write("-" * 50 + "\n")

        f.write("1. Input Mixture (Lower Bound - No Processing)\n")
        f.write(f"  SI-SNR  : {results['Mix_SI-SNR']:.4f} dB\n")
        f.write(f"  SNR     : {results['Mix_SNR']:.4f} dB\n")
        f.write(f"  PESQ    : {results['Mix_PESQ']:.4f}\n")
        f.write(f"  STOI    : {results['Mix_STOI']:.4f}\n")

        f.write("\n2. IRM Oracle (Upper Bound for STFT based Masking)\n")
        f.write(f"  SI-SNR  : {results['IRM_SI-SNR']:.4f} dB\n")
        f.write(f"  SI-SNRi : {results['IRM_SI-SNRi']:.4f} dB\n")
        f.write(f"  SNR     : {results['IRM_SNR']:.4f} dB\n")
        f.write(f"  PESQ    : {results['IRM_PESQ']:.4f}\n")
        f.write(f"  STOI    : {results['IRM_STOI']:.4f}\n")
        f.write("-" * 50 + "\n")

    print(f"Metrics saved to: {txt_path}")
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset path
    root = "../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"
    sample_rate = 8000

    # Results save path
    output_dir = "../../Results/Oracle_Baselines_WSJ"

    print(f"Device: {device}")

    # Load test set
    _, _, test_loader = MakeWSJLoader(root=root, batch_size=32, num_workers=4, segment=4.0, sample_rate=sample_rate)

    # Run tests
    evaluate_baselines(test_loader, device, output_dir, sample_rate=sample_rate)