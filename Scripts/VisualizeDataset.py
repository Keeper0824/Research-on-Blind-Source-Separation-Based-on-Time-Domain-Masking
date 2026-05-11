import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy.signal import welch


def visualize_generated_data(root_dir='./Comm0-2Dataset', split='tr', sample_idx=None):
    """Visually generated communication signal data set :param root_dir: Data set root directory :param split: 'tr' (training set), 'cv' (validation set), 'tt' (test set) :param sample_idx: specifies which sample to view, None means random selection"""
    mix_dir = os.path.join(root_dir, split, 'mix')
    s1_dir = os.path.join(root_dir, split, 's1')
    s2_dir = os.path.join(root_dir, split, 's2')

    # Check if directory exists
    if not os.path.exists(mix_dir):
        print(f"Error: Directory {mix_dir} not found. Please run the generation script first.")
        return

    # Get file list
    files = sorted(os.listdir(mix_dir))
    if len(files) == 0:
        print("Error: No data found.")
        return

    # Select sample
    if sample_idx is None:
        filename = random.choice(files)
    else:
        filename = f"{sample_idx:05d}.npy"
        if filename not in files:
            print(f"Error: File {filename} not found.")
            return

    print(f"Visualizing sample: {filename} from {split} set")

    # Load data (.npy)
    mix = np.load(os.path.join(mix_dir, filename))
    s1 = np.load(os.path.join(s1_dir, filename))
    s2 = np.load(os.path.join(s2_dir, filename))

    # --- Start drawing ---
    fig, axes = plt.subplots(3, 2, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.4)

    fs = 16000  # Sampling rate
    n_plot = 400  # Plot only the first 400 points to see waveform details

    # 1. Time Domain - Zoomed In
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

    # 2. Frequency Domain Power Spectral Density (PSD) - Full Length Signal
    # Calculate the power spectrum using Welch's method.
    def plot_psd(ax, sig, color, label):
        f, Pxx = welch(sig, fs, nperseg=1024)
        ax.semilogy(f, Pxx, color=color)
        ax.set_title(label)
        ax.set_ylabel('PSD (V**2/Hz)')
        ax.grid(True, which='both', alpha=0.3)
        # Mark carrier frequency 2000Hz
        ax.axvline(2000, color='g', linestyle='--', alpha=0.5, label='Carrier (2kHz)')
        ax.legend(loc='upper right')

    plot_psd(axes[0, 1], mix, 'k', 'Mixed Signal PSD')
    plot_psd(axes[1, 1], s1, 'b', 'Source 1 PSD')
    plot_psd(axes[2, 1], s2, 'r', 'Source 2 PSD')
    axes[2, 1].set_xlabel('Frequency (Hz)')

    plt.suptitle(f'Dataset Visualization: {filename}\nSampling Rate={fs}Hz, Carrier~2000Hz', fontsize=16)
    plt.show()


if __name__ == "__main__":
    # View a random training set sample
    visualize_generated_data()