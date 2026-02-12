import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from LossFunction.SI_SNR import si_snr
from DataLoader.WSJDataLoader import MakeDataLoader as MakeWSJLoader


def compute_irm_score(mix, s1, s2, n_fft=512, hop=128):
    """
    计算理想比值掩蔽 (IRM) 的上限分数 (Oracle Performance)
    """
    eps = 1e-8
    window = torch.hann_window(n_fft).to(mix.device)
    X = torch.stft(mix, n_fft, hop, window=window, return_complex=True).abs()
    S1 = torch.stft(s1, n_fft, hop, window=window, return_complex=True).abs()
    S2 = torch.stft(s2, n_fft, hop, window=window, return_complex=True).abs()

    mask1 = S1 / (S1 + S2 + eps)
    mask2 = S2 / (S1 + S2 + eps)

    S1_est = X * mask1
    S2_est = X * mask2

    X_complex = torch.stft(mix, n_fft, hop, window=window, return_complex=True)
    phase = torch.angle(X_complex)

    S1_est_complex = S1_est * torch.exp(1j * phase)
    S2_est_complex = S2_est * torch.exp(1j * phase)

    s1_recon = torch.istft(S1_est_complex, n_fft, hop, window=window, length=mix.shape[-1])
    s2_recon = torch.istft(S2_est_complex, n_fft, hop, window=window, length=mix.shape[-1])

    return s1_recon, s2_recon


def evaluate_baselines(loader, device):
    results = {"mix_snr": [], "irm_snr": []}

    print("Running Oracle (IRM) and Mixture Baselines...")
    for mix, targets in tqdm(loader):
        mix = mix.to(device)
        s1 = targets[:, 0].to(device)
        s2 = targets[:, 1].to(device)

        # Baseline 1: Input Mixture SI-SNR
        # 计算整个 Batch 的结果，形状是 [8]
        batch_snr_mix = (si_snr(mix, s1) + si_snr(mix, s2)) / 2

        # 【修正点】先 .mean() 取平均，再 .item() 转标量
        results["mix_snr"].append(batch_snr_mix.mean().item())

        # Baseline 2: IRM
        s1_irm, s2_irm = compute_irm_score(mix, s1, s2)
        batch_snr_irm = (si_snr(s1_irm, s1) + si_snr(s2_irm, s2)) / 2

        # 【修正点】同上，先取平均
        results["irm_snr"].append(batch_snr_irm.mean().item())

    print(f"\n[Baseline Results]")
    print(f"Input Mixture SI-SNR (Lower Bound): {np.mean(results['mix_snr']):.2f} dB")
    print(f"IRM Oracle SI-SNR (Upper Bound for STFT): {np.mean(results['irm_snr']):.2f} dB")
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = "../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"

    # 加载测试集
    _, _, test_loader = MakeWSJLoader(root=root, batch_size=8, num_workers=4, segment=4.0, sample_rate=8000)

    evaluate_baselines(test_loader, device)