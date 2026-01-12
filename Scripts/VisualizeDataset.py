import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy.signal import welch


def visualize_generated_data(root_dir='./CommDataset', split='tr', sample_idx=None):
    """
    可视化生成的通信信号数据集
    :param root_dir: 数据集根目录
    :param split: 'tr' (训练集), 'cv' (验证集), 'tt' (测试集)
    :param sample_idx: 指定查看第几个样本，None则随机选择
    """
    mix_dir = os.path.join(root_dir, split, 'mix')
    s1_dir = os.path.join(root_dir, split, 's1')
    s2_dir = os.path.join(root_dir, split, 's2')

    # 检查目录是否存在
    if not os.path.exists(mix_dir):
        print(f"Error: Directory {mix_dir} not found. Please run the generation script first.")
        return

    # 获取文件列表
    files = sorted(os.listdir(mix_dir))
    if len(files) == 0:
        print("Error: No data found.")
        return

    # 选择样本
    if sample_idx is None:
        filename = random.choice(files)
    else:
        filename = f"{sample_idx:05d}.npy"
        if filename not in files:
            print(f"Error: File {filename} not found.")
            return

    print(f"Visualizing sample: {filename} from {split} set")

    # 加载数据 (.npy)
    mix = np.load(os.path.join(mix_dir, filename))
    s1 = np.load(os.path.join(s1_dir, filename))
    s2 = np.load(os.path.join(s2_dir, filename))

    # --- 开始绘图 ---
    fig, axes = plt.subplots(3, 2, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.4)

    fs = 16000  # 采样率
    n_plot = 400  # 仅绘制前 400 个点，以便看清波形细节

    # 1. 时域波形 (Time Domain) - Zoomed In
    t = np.arange(n_plot) / fs

    # Mix
    axes[0, 0].plot(t, mix[:n_plot], 'k', linewidth=1.5)
    axes[0, 0].set_title(f'Mixed Signal (Time Domain) - First {n_plot} samples')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    # Source 1
    axes[1, 0].plot(t, s1[:n_plot], 'b', linewidth=1.5)
    axes[1, 0].set_title('Source 1 (Time Domain)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)

    # Source 2
    axes[2, 0].plot(t, s2[:n_plot], 'r', linewidth=1.5)
    axes[2, 0].set_title('Source 2 (Time Domain)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].grid(True, alpha=0.3)

    # 2. 频域功率谱密度 (PSD) - 全长信号
    # 使用 Welch 方法计算功率谱
    def plot_psd(ax, sig, color, label):
        f, Pxx = welch(sig, fs, nperseg=1024)
        ax.semilogy(f, Pxx, color=color)
        ax.set_title(label)
        ax.set_ylabel('PSD (V**2/Hz)')
        ax.grid(True, which='both', alpha=0.3)
        # 标记载波频率 2000Hz
        ax.axvline(2000, color='g', linestyle='--', alpha=0.5, label='Carrier (2kHz)')
        ax.legend(loc='upper right')

    plot_psd(axes[0, 1], mix, 'k', 'Mixed Signal PSD')
    plot_psd(axes[1, 1], s1, 'b', 'Source 1 PSD')
    plot_psd(axes[2, 1], s2, 'r', 'Source 2 PSD')
    axes[2, 1].set_xlabel('Frequency (Hz)')

    plt.suptitle(f'Dataset Visualization: {filename}\nSampling Rate={fs}Hz, Carrier~2000Hz', fontsize=16)
    plt.show()


if __name__ == "__main__":
    # 随机查看一个训练集样本
    visualize_generated_data()