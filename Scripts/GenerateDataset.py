import numpy as np
import os
import shutil


# ==========================================
# 1. 核心信号生成函数 (保持之前的逻辑)
# ==========================================
def generate_bit_sequence(num_bits):
    return np.random.randint(0, 2, num_bits)


def qam_modulate(bits, m_order):
    if m_order == 2:  # BPSK
        return 2 * bits - 1
    elif m_order == 4:  # QPSK
        if len(bits) % 2 != 0: bits = bits[:-1]
        return ((2 * bits[0::2] - 1) + 1j * (2 * bits[1::2] - 1)) / np.sqrt(2)
    elif m_order == 16:  # 16QAM
        def map_2bits(b):
            return 2 * (2 * b[0::2] + b[1::2]) - 3

        if len(bits) % 4 != 0: bits = bits[:-(len(bits) % 4)]
        return (map_2bits(bits[0::2]) + 1j * map_2bits(bits[1::2])) / np.sqrt(10)
    return np.zeros(len(bits) // int(np.log2(m_order)))


def rrcosfilter(N, alpha, Ts, Fs):
    # 简化版 RRC
    T_delta = 1 / float(Fs)
    h_rrc = np.zeros(N, dtype=float)
    for x in range(N):
        t = (x - N / 2) * T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and abs(t) == Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (
                        (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            n = np.sin(np.pi * t * (1 - alpha) / Ts) + 4 * alpha * t / Ts * np.cos(np.pi * t * (1 + alpha) / Ts)
            d = np.pi * t / Ts * (1 - (4 * alpha * t / Ts) ** 2)
            h_rrc[x] = n / d
    return h_rrc


def generate_single_source(length_limit, mod_type='QPSK', sps=8, fs=16000, fc=2000):
    num_symbols = int(length_limit / sps * 1.5)
    m_order_map = {'BPSK': 2, 'QPSK': 4, '16QAM': 16}
    m_order = m_order_map.get(mod_type, 4)

    bits = generate_bit_sequence(num_symbols * int(np.log2(m_order)))
    symbols = qam_modulate(bits, m_order)

    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    h_rrc = rrcosfilter(sps * 10 + 1, 0.35, 1.0, sps)
    baseband = np.convolve(upsampled, h_rrc, mode='same')

    freq_offset = np.random.uniform(-50, 50)
    phase_offset = np.random.uniform(0, 2 * np.pi)
    t = np.arange(len(baseband)) / fs

    carrier = np.exp(1j * (2 * np.pi * (fc + freq_offset) * t + phase_offset))
    signal_real = np.real(baseband * carrier)

    if len(signal_real) > length_limit:
        start = np.random.randint(0, len(signal_real) - length_limit)
        return signal_real[start: start + length_limit]
    else:
        return np.pad(signal_real, (0, length_limit - len(signal_real)))


# ==========================================
# 2. 数据集生成器 (适配 WSJ0 目录结构)
# ==========================================
def create_comm_dataset(root_dir='./Comm0-2Dataset', num_train=20000, num_val=5000, num_test=5000, sample_len=32000):
    splits = {
        'tr': num_train,  # 训练集
        'cv': num_val,  # 验证集
        'tt': num_test  # 测试集
    }

    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)  # 清空旧数据

    for split_name, num_files in splits.items():
        print(f"Generating {split_name} set ({num_files} files)...")

        # 创建目录结构：root/tr/mix, root/tr/s1, root/tr/s2
        os.makedirs(os.path.join(root_dir, split_name, 'mix'), exist_ok=True)
        os.makedirs(os.path.join(root_dir, split_name, 's1'), exist_ok=True)
        os.makedirs(os.path.join(root_dir, split_name, 's2'), exist_ok=True)

        for i in range(num_files):
            # 随机参数
            mod1 = np.random.choice(['BPSK', 'QPSK', '16QAM'])
            mod2 = np.random.choice(['BPSK', 'QPSK', '16QAM'])

            s1 = generate_single_source(sample_len, mod_type=mod1)
            s2 = generate_single_source(sample_len, mod_type=mod2)

            # 能量归一化
            s1 = s1 / (np.std(s1) + 1e-8)
            s2 = s2 / (np.std(s2) + 1e-8)

            # 混合 SIR (0dB - 5dB)
            sir_db = np.random.uniform(0, 5)
            scale = 10 ** (sir_db / 20)
            # 随机决定谁强谁弱
            if np.random.rand() > 0.5:
                s1 *= scale
            else:
                s2 *= scale

            mix_clean = s1 + s2

            # 加噪 SNR (10dB - 30dB)
            snr_db = np.random.uniform(10, 30)
            sig_power = np.mean(mix_clean ** 2)
            noise_power = sig_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), sample_len)

            mix_final = mix_clean + noise

            # 最终幅值归一化 (防止梯度爆炸)
            max_val = np.max(np.abs(mix_final)) + 1e-8
            mix_final /= max_val
            s1 /= max_val
            s2 /= max_val

            # 保存为 .npy 格式 (文件名用索引)
            # 使用 float32 节省空间
            file_name = f"{i:05d}.npy"
            np.save(os.path.join(root_dir, split_name, 'mix', file_name), mix_final.astype(np.float32))
            np.save(os.path.join(root_dir, split_name, 's1', file_name), s1.astype(np.float32))
            np.save(os.path.join(root_dir, split_name, 's2', file_name), s2.astype(np.float32))

    print(f"Dataset generation complete! Saved to {root_dir}")


if __name__ == "__main__":
    # 生成数据
    create_comm_dataset()