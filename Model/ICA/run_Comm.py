import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import FastICA, NMF
from itertools import permutations

# 【修改点1】引入 CommDataLoader
from DataLoader.CommDataLoader import MakeDataLoader

# 假设你的 SI_SNR 在这个位置
try:
    from LossFunction.SI_SNR import si_snr
except ImportError:
    def si_snr(preds, target, eps=1e-8):
        def _dot(x, y): return torch.sum(x * y, dim=-1, keepdim=True)

        target_energy = _dot(target, target) + eps
        optimal_dot = _dot(preds, target)
        s_target = (optimal_dot * target) / target_energy
        e_noise = preds - s_target
        snr = 10 * torch.log10(_dot(s_target, s_target) / (_dot(e_noise, e_noise) + eps))
        return snr.mean()


def compute_metrics(est_s1, est_s2, s1, s2):
    """ 计算 SI-SNR (PIT) """
    if isinstance(est_s1, np.ndarray): est_s1 = torch.from_numpy(est_s1).float()
    if isinstance(est_s2, np.ndarray): est_s2 = torch.from_numpy(est_s2).float()
    if isinstance(s1, np.ndarray): s1 = torch.from_numpy(s1).float()
    if isinstance(s2, np.ndarray): s2 = torch.from_numpy(s2).float()

    snr_1 = (si_snr(est_s1, s1) + si_snr(est_s2, s2)) / 2
    snr_2 = (si_snr(est_s1, s2) + si_snr(est_s2, s1)) / 2
    return max(snr_1, snr_2).item()


def run_statistical_separation(mix_wav, method='nmf', n_fft=512, hop=256):
    """
    对单条信号执行统计学分离
    注意：对于通讯信号，频谱特征可能不像语音那样有明显的谐波结构，
    但这正好能测试传统方法是否具备泛化能力。
    """
    # 1. STFT 变换
    spec_complex = torch.stft(
        torch.from_numpy(mix_wav).unsqueeze(0),
        n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=torch.hann_window(n_fft),
        return_complex=True
    ).squeeze(0).numpy()

    mag = np.abs(spec_complex)
    phase = np.angle(spec_complex)
    X = mag.T  # (Samples, Features)

    # 2. 运行算法
    if method == 'ica':
        model = FastICA(n_components=2, max_iter=800, tol=1e-2, random_state=42)
    elif method == 'nmf':
        model = NMF(n_components=2, init='random', random_state=42, max_iter=500)
    else:
        raise ValueError("Unknown method")

    try:
        W = model.fit_transform(X)
        H = model.components_

        # 3. 重构幅度谱 + Soft Masking
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
        return mix_wav, mix_wav


def main():
    # 【修改点2】 配置 CommDataset 路径
    root = "../../Dataset/CommDataset"

    # 【修改点3】 CommDataLoader 也不需要 sample_rate，segment 用点数
    # 根据你之前的 trainTasNet2.py，segment=32000 是比较合理的点数
    segment_points = 32000

    print(f"加载 CommDataset 测试集 (Segment={segment_points})...")

    # 注意：这里调用的是 CommDataLoader 的 MakeDataLoader
    # 它的签名通常是 (root, batch_size, num_workers, segment)
    _, _, test_loader = MakeDataLoader(root=root, batch_size=1, num_workers=4, segment=segment_points)

    methods = ['ica', 'nmf']
    results = {m: [] for m in methods}
    max_samples = 50

    for method in methods:
        print(f"\n正在运行 {method.upper()} 基准测试 (Max samples: {max_samples})...")
        count = 0
        pbar = tqdm(total=max_samples)

        for mix, targets in test_loader:
            if count >= max_samples:
                break

            # CommDataset 返回的数据是 float tensor
            mix_np = mix.squeeze().numpy()
            s1_np = targets[:, 0].squeeze().numpy()
            s2_np = targets[:, 1].squeeze().numpy()

            # 运行分离算法
            est_s1, est_s2 = run_statistical_separation(mix_np, method=method)

            # 计算分数
            score = compute_metrics(est_s1, est_s2, s1_np, s2_np)
            results[method].append(score)

            count += 1
            pbar.update(1)
            pbar.set_postfix({"Avg SI-SNR": f"{np.mean(results[method]):.2f}"})

        pbar.close()
        avg_score = np.mean(results[method])
        print(f"CommDataset Statistical Baseline [{method.upper()}] Average SI-SNR: {avg_score:.2f} dB")


if __name__ == "__main__":
    main()