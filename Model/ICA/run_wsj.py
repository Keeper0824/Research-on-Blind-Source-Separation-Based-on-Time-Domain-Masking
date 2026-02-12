import os
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from sklearn.decomposition import FastICA, NMF
from itertools import permutations

# 引入你的数据加载器和Loss
from DataLoader.WSJDataLoader import MakeDataLoader

# 假设你的 SI_SNR 在这个位置，如果不在请调整 import
try:
    from LossFunction.SI_SNR import si_snr
except ImportError:
    # 如果找不到，手写一个简单的 SI-SNR 以防万一
    def si_snr(preds, target, eps=1e-8):
        # preds: [T], target: [T]
        def _dot(x, y): return torch.sum(x * y, dim=-1, keepdim=True)

        target_energy = _dot(target, target) + eps
        optimal_dot = _dot(preds, target)
        s_target = (optimal_dot * target) / target_energy
        e_noise = preds - s_target
        snr = 10 * torch.log10(_dot(s_target, s_target) / (_dot(e_noise, e_noise) + eps))
        return snr.mean()


def compute_metrics(est_s1, est_s2, s1, s2):
    """
    计算 SI-SNR，因为统计学方法不知道哪个输出对应哪个说话人，
    所以我们需要做一个简单的排列组合 (Permutation Invariant Training 思想) 找最大值。
    """
    # 转换为 Tensor
    if isinstance(est_s1, np.ndarray): est_s1 = torch.from_numpy(est_s1).float()
    if isinstance(est_s2, np.ndarray): est_s2 = torch.from_numpy(est_s2).float()
    if isinstance(s1, np.ndarray): s1 = torch.from_numpy(s1).float()
    if isinstance(s2, np.ndarray): s2 = torch.from_numpy(s2).float()

    # 路径 1: est1->s1, est2->s2
    snr_1 = (si_snr(est_s1, s1) + si_snr(est_s2, s2)) / 2
    # 路径 2: est1->s2, est2->s1
    snr_2 = (si_snr(est_s1, s2) + si_snr(est_s2, s1)) / 2

    return max(snr_1, snr_2).item()


def run_statistical_separation(mix_wav, method='nmf', n_fft=512, hop=256):
    """
    对单条音频执行统计学分离
    mix_wav: numpy array [T]
    method: 'ica' 或 'nmf'
    """
    # 1. STFT 变换 (时域 -> 频域)
    # mix_wav shape: (T,)
    # stft shape: (F, Time) complex
    # sklearn 需要 (n_samples, n_features)，通常我们把时间帧看作 samples

    # 这里的 window 等参数尽量简单，只为了提取幅度谱
    spec_complex = torch.stft(
        torch.from_numpy(mix_wav).unsqueeze(0),
        n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=torch.hann_window(n_fft),
        return_complex=True
    ).squeeze(0).numpy()  # (F, T)

    # 提取幅度谱 (Magnitude) 和 相位 (Phase)
    mag = np.abs(spec_complex)  # (F, T)
    phase = np.angle(spec_complex)

    # 输入给 sklearn 的数据形状： (Samples=T, Features=F)
    X = mag.T

    # 2. 运行算法
    if method == 'ica':
        # FastICA 容易不收敛，增加 max_iter
        model = FastICA(n_components=2, max_iter=800, tol=1e-2, random_state=42)
    elif method == 'nmf':
        # NMF 是非负矩阵分解，更适合幅度谱
        model = NMF(n_components=2, init='random', random_state=42, max_iter=500)
    else:
        raise ValueError("Unknown method")

    try:
        # transform 得到的是随时间变化的激活 (Time activations)
        # components_ 得到的是频谱基向量 (Frequency basis)
        W = model.fit_transform(X)  # (T, 2)
        H = model.components_  # (2, F)

        # 3. 重构幅度谱 (Reconstruction)
        # Component 1
        est_mag_1 = np.outer(W[:, 0], H[0, :]).T  # (F, T)
        # Component 2
        est_mag_2 = np.outer(W[:, 1], H[1, :]).T  # (F, T)

        # 4. 维纳滤波 / Soft Masking (关键步骤：利用比例掩蔽恢复信号)
        # 纯粹的矩阵乘积丢失了细节，用 Mask 效果更好
        sum_mag = est_mag_1 + est_mag_2 + 1e-8
        mask_1 = est_mag_1 / sum_mag
        mask_2 = est_mag_2 / sum_mag

        # 应用原始幅度谱
        Y1_mag = mag * mask_1
        Y2_mag = mag * mask_2

        # 5. iSTFT (频域 -> 时域)
        # 使用原始混合语音的相位 (Phase-Aware)
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
        # 如果算法不收敛 (ICA 常见)，返回混合语音作为失败的结果
        # print(f"Algorithm failed: {e}")
        return mix_wav, mix_wav


def main():
    # 配置路径
    root = "../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"
    sample_rate = 8000

    print("加载测试集...")
    _, _, test_loader = MakeDataLoader(root=root, batch_size=1, num_workers=4, segment=4.0, sample_rate=sample_rate)

    methods = ['ica', 'nmf']  # 你可以两个都跑，或者只跑 ica
    results = {m: [] for m in methods}

    # 为了节省时间，只跑前 100 个样本做评估 (足够发论文对比了，统计学方法很慢)
    max_samples = 50

    for method in methods:
        print(f"\n正在运行 {method.upper()} 基准测试 (Max samples: {max_samples})...")
        count = 0
        pbar = tqdm(total=max_samples)

        for mix, targets in test_loader:
            if count >= max_samples:
                break

            mix_np = mix.squeeze().numpy()  # (T,)
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
        print(f"Independent Statistical Baseline [{method.upper()}] Average SI-SNR: {avg_score:.2f} dB")


if __name__ == "__main__":
    main()