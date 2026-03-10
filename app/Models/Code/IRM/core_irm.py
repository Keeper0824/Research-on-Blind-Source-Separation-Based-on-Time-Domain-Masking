import torch

class IRMInferenceCore:
    def __init__(self, n_fft=512, hop=128, device=None):
        """
        IRM (Ideal Ratio Mask) 理论上限推理核心类。
        不需要加载权重文件，仅用于作为极限对比基线。
        :param n_fft: STFT 窗口大小 (语音常用 512)
        :param hop: 步长 (常用 128 或 256)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_fft = n_fft
        self.hop = hop

    @torch.no_grad()
    def separate(self, mix_waveform, s1_waveform, s2_waveform):
        """
        计算 IRM 并分离信号。注意：与盲分离模型不同，这里必须提供 ground truth！
        :param mix_waveform: 混合信号 Tensor, (1, T) 或 (T,)
        :param s1_waveform: 干净源 1 Tensor, (1, T) 或 (T,)
        :param s2_waveform: 干净源 2 Tensor, (1, T) 或 (T,)
        :return: est_sources, shape (2, T), 位于 CPU 上
        """
        # 1. 维度与设备整理，统一转为一维 (T,) 以适配 stft 批量操作
        mix = mix_waveform.squeeze().to(self.device)
        s1 = s1_waveform.squeeze().to(self.device)
        s2 = s2_waveform.squeeze().to(self.device)

        # 检查长度对齐，取最短长度
        min_len = min(mix.shape[-1], s1.shape[-1], s2.shape[-1])
        mix = mix[:min_len]
        s1 = s1[:min_len]
        s2 = s2[:min_len]

        eps = 1e-8
        window = torch.hann_window(self.n_fft).to(self.device)

        # 2. 计算 STFT 频谱
        X_complex = torch.stft(mix, self.n_fft, self.hop, window=window, return_complex=True)
        X_mag = X_complex.abs()

        S1_mag = torch.stft(s1, self.n_fft, self.hop, window=window, return_complex=True).abs()
        S2_mag = torch.stft(s2, self.n_fft, self.hop, window=window, return_complex=True).abs()

        # 3. 构造 IRM Mask (理想比值掩蔽)
        mask1 = S1_mag / (S1_mag + S2_mag + eps)
        mask2 = S2_mag / (S1_mag + S2_mag + eps)

        # 4. 应用 Mask (Oracle 分离)
        S1_est_mag = X_mag * mask1
        S2_est_mag = X_mag * mask2

        # 5. iSTFT (使用原混合信号的相位还原到时域)
        phase = torch.angle(X_complex)

        S1_est_complex = S1_est_mag * torch.exp(1j * phase)
        S2_est_complex = S2_est_mag * torch.exp(1j * phase)

        s1_recon = torch.istft(S1_est_complex, self.n_fft, self.hop, window=window, length=mix.shape[-1])
        s2_recon = torch.istft(S2_est_complex, self.n_fft, self.hop, window=window, length=mix.shape[-1])

        # 6. 去除直流分量 (与你的深度学习模型输出保持相同后处理)
        s1_recon = s1_recon - s1_recon.mean(dim=-1, keepdim=True)
        s2_recon = s2_recon - s2_recon.mean(dim=-1, keepdim=True)

        # 7. 拼接并返回 (2, T)，转移至 CPU 以便 UI 绘图
        est_sources = torch.stack((s1_recon, s2_recon), dim=0)
        return est_sources.cpu()