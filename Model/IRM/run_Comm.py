import torch
import numpy as np
from tqdm import tqdm
from LossFunction.SI_SNR import si_snr
# 【修改点1】引入 CommDataLoader
from DataLoader.CommDataLoader import MakeDataLoader as MakeCommLoader


def compute_irm_score(mix, s1, s2, n_fft=512, hop=128):
    """
    计算理想比值掩蔽 (IRM) 的上限分数 (Oracle Performance)
    """
    eps = 1e-8
    # 确保 window 在正确的设备上
    window = torch.hann_window(n_fft).to(mix.device)

    # 1. STFT
    X = torch.stft(mix, n_fft, hop, window=window, return_complex=True).abs()
    S1 = torch.stft(s1, n_fft, hop, window=window, return_complex=True).abs()
    S2 = torch.stft(s2, n_fft, hop, window=window, return_complex=True).abs()

    # 2. Construct IRM Mask
    mask1 = S1 / (S1 + S2 + eps)
    mask2 = S2 / (S1 + S2 + eps)

    # 3. Apply Mask (Oracle Separation)
    S1_est = X * mask1
    S2_est = X * mask2

    # 4. iSTFT (Back to Time Domain)
    X_complex = torch.stft(mix, n_fft, hop, window=window, return_complex=True)
    phase = torch.angle(X_complex)

    S1_est_complex = S1_est * torch.exp(1j * phase)
    S2_est_complex = S2_est * torch.exp(1j * phase)

    # 恢复时域信号
    s1_recon = torch.istft(S1_est_complex, n_fft, hop, window=window, length=mix.shape[-1])
    s2_recon = torch.istft(S2_est_complex, n_fft, hop, window=window, length=mix.shape[-1])

    return s1_recon, s2_recon


def evaluate_baselines(loader, device):
    results = {"mix_snr": [], "irm_snr": []}

    print("Running Oracle (IRM) and Mixture Baselines on CommDataset...")

    for mix, targets in tqdm(loader):
        mix = mix.to(device)
        s1 = targets[:, 0].to(device)
        s2 = targets[:, 1].to(device)

        # Baseline 1: Input Mixture SI-SNR (不做任何处理的分数)
        # 计算整个 Batch 的结果
        batch_snr_mix = (si_snr(mix, s1) + si_snr(mix, s2)) / 2

        # 【关键修复】先取平均再转标量
        results["mix_snr"].append(batch_snr_mix.mean().item())

        # Baseline 2: IRM (频域理论上限)
        s1_irm, s2_irm = compute_irm_score(mix, s1, s2)
        batch_snr_irm = (si_snr(s1_irm, s1) + si_snr(s2_irm, s2)) / 2

        results["irm_snr"].append(batch_snr_irm.mean().item())

    print(f"\n[CommDataset Baseline Results]")
    print(f"Input Mixture SI-SNR (Lower Bound): {np.mean(results['mix_snr']):.2f} dB")
    print(f"IRM Oracle SI-SNR (Upper Bound for STFT): {np.mean(results['irm_snr']):.2f} dB")
    return results


if __name__ == "__main__":
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 【修改点2】 CommDataset 路径
    root = "../../Dataset/CommDataset"

    # 【修改点3】 使用点数作为 segment (32000点), 不需要 sample_rate 参数
    # 注意：CommDataset 通常没有 split="tt"，如果报错，请检查 CommDataLoader 里是否写了 split="tt"
    # 如果只有 "tr" 和 "cv"，这里临时改成 split="cv" 也可以，或者确保你有 test set
    train_loader, valid_loader, test_loader = MakeCommLoader(root=root, batch_size=8, num_workers=4, segment=32000)

    # 运行测试 (使用 test_loader)
    evaluate_baselines(test_loader, device)