import os
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from itertools import permutations
from DataLoader.WSJDataLoader import MakeDataLoader as MakeWSJLoader

# 引入你的 DataLoader
# 默认使用 CommDataset，如果跑 WSJ 请切换注释
from DataLoader.CommDataLoader import MakeDataLoader as MakeCommLoader


# from DataLoader.WSJDataLoader import MakeDataLoader as MakeWSJLoader

class STFTMaskNet(nn.Module):
    """
    标准的时频域基准模型 (PIT-BLSTM)
    流程: Wave -> STFT -> LogMag -> BiLSTM -> Mask -> Apply Mask -> iSTFT
    """

    def __init__(self, n_fft=512, hop_length=128, hidden_size=300, num_layers=3, num_spks=2):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_spks = num_spks

        # 输入维度是频点数 n_fft // 2 + 1
        input_dim = n_fft // 2 + 1

        # 核心：双向 LSTM 提取频谱特征
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 输出层：映射回 mask 维度 (双向所以 *2)
        self.linear = nn.Linear(hidden_size * 2, input_dim * num_spks)

        # Mask 范围限制在 [0, 1]
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, T_wave]
        return: [B, num_spks, T_wave]
        """
        B = x.shape[0]
        device = x.device

        # 1. STFT (Time -> Freq)
        # return: [B, F, T, 2] -> [B, F, T] (complex magnitude)
        window = torch.hann_window(self.n_fft).to(device)
        stft_res = torch.stft(x, self.n_fft, self.hop_length, window=window, return_complex=True)
        mag = stft_res.abs()  # [B, F, T]
        phase = stft_res.angle()

        # 2. Log-Spectrogram (Feature Extraction)
        # 转置为 [B, T, F] 喂给 LSTM
        log_mag = torch.log1p(mag).permute(0, 2, 1)

        # 3. LSTM Mask Estimation
        rnn_out, _ = self.lstm(log_mag)  # [B, T, H*2]
        mask_out = self.linear(rnn_out)  # [B, T, F*Spk]
        mask_out = self.activation(mask_out)

        # Reshape to [B, Spk, F, T]
        mask_out = mask_out.view(B, -1, self.num_spks, self.n_fft // 2 + 1)
        mask_out = mask_out.permute(0, 2, 3, 1)  # [B, Spk, F, T]

        # 4. Apply Mask (Freq Domain Separation)
        # 扩展 mag 维度以匹配 mask: [B, 1, F, T] * [B, Spk, F, T]
        est_mags = mag.unsqueeze(1) * mask_out

        # 5. iSTFT (Freq -> Time)
        # 使用原始混合语音的相位 (Phase-Aware)
        # est_mags: [B, Spk, F, T]
        est_wavs = []
        for i in range(self.num_spks):
            m = est_mags[:, i]
            # Reconstruct complex: Mag * e^(j*Phase)
            comp = m * torch.exp(1j * phase)
            wav = torch.istft(comp, self.n_fft, self.hop_length, window=window, length=x.shape[-1])
            est_wavs.append(wav)

        return torch.stack(est_wavs, dim=1)  # [B, Spk, T]


# 简单的 PIT Loss (Si-SNR)
def pit_snr_loss(preds, targets):
    """
    preds: [B, Spk, T]
    targets: [B, Spk, T]
    """

    def si_snr(s1, s2, eps=1e-8):
        s1_s2_dot = torch.sum(s1 * s2, dim=-1, keepdim=True)
        s2_s2_dot = torch.sum(s2 * s2, dim=-1, keepdim=True)
        s_target = (s1_s2_dot * s2) / (s2_s2_dot + eps)
        e_noise = s1 - s_target
        snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + eps))
        return -torch.mean(snr)

    # 简化的 PIT：只处理 2 个说话人
    # Permutation 1: 0->0, 1->1
    loss1 = (si_snr(preds[:, 0], targets[:, 0]) + si_snr(preds[:, 1], targets[:, 1])) / 2
    # Permutation 2: 0->1, 1->0
    loss2 = (si_snr(preds[:, 0], targets[:, 1]) + si_snr(preds[:, 1], targets[:, 0])) / 2

    return torch.min(loss1, loss2)


def main():
    # === 配置 ===
    # 你的 CommDataset 路径
    # root = "../../Dataset/CommDataset"
    root = "../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"
    # 如果要跑 WSJ，把这里改成 WSJ 路径，下面 loader 也改一下

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256  # 频域模型显存占用小，可以大一点
    epochs = 100  # 基准模型不需要跑太久，收敛就行

    print(f"Running TF-Domain Baseline (PIT-BLSTM) on {device}...")

    # === 数据加载 ===
    # 使用 CommLoader
    # train_loader, valid_loader, _ = MakeCommLoader(root=root, batch_size=batch_size, num_workers=4, segment=32000)
    train_loader, valid_loader, _ = MakeWSJLoader(
        root=root,
        batch_size=batch_size,
        num_workers=4,
        segment=4,
        sample_rate=8000
    )
    # === 模型 ===
    model = STFTMaskNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # === 训练循环 ===
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for mix, targets in pbar:
            mix = mix.to(device)  # [B, T]
            targets = targets.to(device)  # [B, 2, T]

            optimizer.zero_grad()
            est = model(mix)

            loss = pit_snr_loss(est, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.2f}"})

        avg_train_loss = train_loss / len(train_loader)

        # === 验证 ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mix, targets in valid_loader:
                mix = mix.to(device)
                targets = targets.to(device)
                est = model(mix)
                loss = pit_snr_loss(est, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.2f}, Val Loss={avg_val_loss:.2f}")

        # 保存最好的
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "baseline_tf_best.pt")
            print("Saved Best Model.")

    print(f"Training Finished. Best Val Loss (neg SI-SNR): {best_loss:.2f}")


if __name__ == "__main__":
    main()