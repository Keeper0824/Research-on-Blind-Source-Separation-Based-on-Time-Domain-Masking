import os
import sys
import time
import torch
import torch.nn as nn
import torchaudio
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# ==========================================
# 1. 导入数据加载器和评估指标
# ==========================================
from DataLoader.CommDataLoader import MakeDataLoader
from Evaluation.SI_SNR import si_snr_improvement, pit_align_by_sisnr
from LossFunction.SI_SNR import pit_si_snr_loss
from Evaluation.SNR import snr


# ==========================================
# 2. 辅助工具：日志与 BER 计算
# ==========================================
class Logger(object):
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def compute_ber(est_waveform, ref_waveform):
    """
    计算误比特率 (BER)，基于 BPSK 零点判决
    """
    est_bits = (est_waveform > 0).float()
    ref_bits = (ref_waveform > 0).float()
    errors = torch.sum(torch.abs(est_bits - ref_bits))
    total_bits = torch.numel(ref_bits)
    return (errors / total_bits).item()


def ensure_mix_bt(mix):
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)
    elif mix.dim() == 3:
        mix = mix.squeeze()
    return mix


def ensure_ref_b2t(ref):
    if ref.dim() == 2:
        ref = ref.unsqueeze(0)
    elif ref.dim() == 3 and ref.size(2) == 2:
        ref = ref.permute(0, 2, 1)
    return ref


def plot_metrics(output_dir, epochs, train_loss, val_loss):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(epochs, [-x for x in train_loss], label="Train SI-SNR (dB)")
    plt.plot(epochs, [-x for x in val_loss], label="Val SI-SNR (dB)")
    plt.xlabel("epoch");
    plt.ylabel("SI-SNR (dB)")
    plt.legend();
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curve_sisnr.png"), dpi=200)
    plt.close()


# ==========================================
# 3. 时频域模型定义 (PIT-BLSTM)
# ==========================================
class STFTMaskNet(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, hidden_size=300, num_layers=3, num_spks=2):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_spks = num_spks
        input_dim = n_fft // 2 + 1

        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.linear = nn.Linear(hidden_size * 2, input_dim * num_spks)
        self.activation = nn.Sigmoid()

    # 强制关闭 AMP，防止 STFT 在半精度下数值溢出
    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        x = x.float()
        B = x.shape[0]
        device = x.device

        # 1. STFT
        window = torch.hann_window(self.n_fft).to(device)
        stft_res = torch.stft(x, self.n_fft, self.hop_length, window=window, return_complex=True)
        mag = stft_res.abs()
        phase = stft_res.angle()

        # 2. Log-Mag Features
        log_mag = torch.log1p(mag).permute(0, 2, 1)  # [B, T, F]

        # 3. LSTM Masking
        rnn_out, _ = self.lstm(log_mag)
        mask_out = self.linear(rnn_out)
        mask_out = self.activation(mask_out)

        mask_out = mask_out.view(B, -1, self.num_spks, self.n_fft // 2 + 1)
        mask_out = mask_out.permute(0, 2, 3, 1)  # [B, Spk, F, T]

        # 4. Apply Mask & iSTFT
        est_mags = mag.unsqueeze(1) * mask_out
        est_wavs = []
        for i in range(self.num_spks):
            m = est_mags[:, i]
            comp = m * torch.exp(1j * phase)
            wav = torch.istft(comp, self.n_fft, self.hop_length, window=window, length=x.shape[-1])
            est_wavs.append(wav)

        return torch.stack(est_wavs, dim=1)


# ==========================================
# 4. 验证与测试流程
# ==========================================
def eval_valid(model, valid_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for mix, targets in valid_loader:
            mix, targets = mix.to(device), targets.to(device)
            est = model(mix)
            min_len = min(est.shape[-1], targets.shape[-1])
            loss = loss_fn(est[..., :min_len], targets[..., :min_len])
            total_loss += loss.item()
    model.train()
    return total_loss / len(valid_loader)


def run_final_test(model, test_loader, sample_rate, device, output_dir, max_save_samples=20):
    print(f"\n>>> 开始最终测试 (Comm0-2Dataset 标准) <<<")
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    metrics = {"SI-SNR": [], "SI-SNRi": [], "SNR": [], "BER": []}
    saved = 0

    with torch.no_grad():
        for mix, targets in tqdm(test_loader, desc="Testing"):
            mix = ensure_mix_bt(mix).to(device)
            targets = ensure_ref_b2t(targets).to(device)

            est = model(mix)

            # 对齐长度
            min_len = min(est.shape[-1], targets.shape[-1], mix.shape[-1])
            est = est[..., :min_len]
            targets = targets[..., :min_len]
            mix = mix[..., :min_len]

            # 1. PIT 对齐
            est_aligned, perm, sisnr_best = pit_align_by_sisnr(est, targets)

            # 去直流 (通信信号通常是零均值)
            est_aligned = est_aligned - est_aligned.mean(dim=-1, keepdim=True)

            # 2. 计算指标
            metrics["SI-SNR"].extend(sisnr_best.cpu().tolist())
            metrics["SI-SNRi"].extend(si_snr_improvement(est_aligned, targets, mix).cpu().tolist())

            batch_snr = 0.5 * (snr(est_aligned[:, 0], targets[:, 0]) + snr(est_aligned[:, 1], targets[:, 1]))
            metrics["SNR"].extend(batch_snr.cpu().tolist())

            # 3. 计算 BER
            ber_batch = []
            for b in range(est_aligned.shape[0]):
                ber = 0.5 * (compute_ber(est_aligned[b, 0], targets[b, 0]) +
                             compute_ber(est_aligned[b, 1], targets[b, 1]))
                ber_batch.append(ber)
            metrics["BER"].extend(ber_batch)

            # 4. 保存 wav
            for i in range(mix.size(0)):
                if saved >= max_save_samples: break
                base = f"tf_baseline_{saved:04d}"
                mix_cpu = mix[i].cpu()
                s1_ref = targets[i, 0].cpu()
                s2_ref = targets[i, 1].cpu()
                s1_est = est_aligned[i, 0].cpu()
                s2_est = est_aligned[i, 1].cpu()

                torchaudio.save(os.path.join(output_dir, f"{base}_mix.wav"), mix_cpu.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_est.wav"), s1_est.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_est.wav"), s2_est.unsqueeze(0), sample_rate)
                saved += 1

    # 结果汇总写入 TXT
    def nanmean(x):
        return float(np.nanmean(x)) if len(x) > 0 else float("nan")

    results = {k: nanmean(v) for k, v in metrics.items()}

    print("\n" + "=" * 40)
    print(f"[TF-Baseline Comm] 最终成绩:")
    for k, v in results.items():
        if k == "BER":
            print(f"  {k:8s}: {v:.6f} ({v * 100:.4f}%)")
        else:
            print(f"  {k:8s}: {v:.4f} dB")
    print("=" * 40 + "\n")

    ts = time.strftime("%Y%m%d-%H%M%S")
    txt_path = os.path.join(output_dir, f"final_metrics_comm_{ts}.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("TF-Domain Baseline (PIT-BLSTM) on Comm0-2Dataset\n")
        f.write(f"Sample Rate: {sample_rate}\n")
        f.write("-" * 40 + "\n")
        for k, v in results.items():
            if k == "BER":
                f.write(f"  {k:8s}: {v:.6f} ({v * 100:.4f}%)\n")
            else:
                f.write(f"  {k:8s}: {v:.4f} dB\n")
        f.write("-" * 40 + "\n")

    print(f"Metrics saved to: {txt_path}")


# ==========================================
# 5. 主程序
# ==========================================
def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_name = f"train_tf_baseline_comm_{timestamp}"
    log_file = os.path.join("../../logs", log_name + ".log")
    sys.stdout = Logger(log_file)
    print(f"Log file: {log_file} (TF-BLSTM on Comm)")

    ckpt_dir = os.path.join("../../checkpoints", log_name)
    output_dir = os.path.join("../../outputs", log_name)
    result_dir = os.path.join("../../Results/TF_Baseline_Comm", log_name)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = "../../Dataset/Comm0-2Dataset"

    # Comm0-2Dataset: segment 32000 points, batch_size 256 (显存允许的情况下尽量大)
    train_loader, valid_loader, test_loader = MakeDataLoader(
        root=root, batch_size=256, num_workers=8, segment=32000
    )

    # 针对通信信号优化 STFT 参数
    model = STFTMaskNet(n_fft=1024, hop_length=256).to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=100, steps_per_epoch=len(train_loader), pct_start=0.1
    )
    loss_fn = pit_si_snr_loss

    best_val_loss = float("inf")
    patience, no_improve = 15, 0
    epochs_hist, train_loss_hist, val_loss_hist = [], [], []

    # --- 训练循环 ---
    model.train()
    for epoch in range(100):
        total_loss = 0.0
        print(f"\n===== Epoch {epoch + 1}/100 =====")

        for mix, targets in train_loader:
            mix, targets = mix.to(device), targets.to(device)
            optimizer.zero_grad()

            est = model(mix)
            min_len = min(est.shape[-1], targets.shape[-1])

            loss = loss_fn(est[..., :min_len], targets[..., :min_len])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            ema_model.update_parameters(model)
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = eval_valid(ema_model, valid_loader, loss_fn, device)
        print(f"Epoch {epoch + 1} | Train: {-avg_train:.2f} dB | Val: {-avg_val:.2f} dB")

        epochs_hist.append(epoch + 1)
        train_loss_hist.append(avg_train)
        val_loss_hist.append(avg_val)
        plot_metrics(output_dir, epochs_hist, train_loss_hist, val_loss_hist)

        if avg_val < best_val_loss - 1e-3:
            best_val_loss = avg_val
            no_improve = 0
            best_state = copy.deepcopy(
                ema_model.module.state_dict() if hasattr(ema_model, "module") else ema_model.state_dict())
            torch.save(best_state, os.path.join(ckpt_dir, "best_model.pt"))
            print("-> Saved Best Model.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    # --- 最终测试 ---
    print("\n>>> Training Finished. Starting Final Test... <<<")
    model.load_state_dict(best_state)
    run_final_test(model, test_loader, 100000, device, result_dir)


if __name__ == "__main__":
    main()