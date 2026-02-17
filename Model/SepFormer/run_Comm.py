import os
import sys
import time
import torch
import torchaudio
import copy
import numpy as np
from tqdm import tqdm
from torch.amp import autocast
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from LossFunction.SI_SNR import pit_si_snr_loss


# ========================================================
# 引入 Asteroid 中的 Dual-Path Transformer (与 SepFormer 原理一致)
# ========================================================
from asteroid.models import DPTNet

# 导入你的数据加载器和评估指标
from DataLoader.CommDataLoader import MakeDataLoader
from Evaluation.SI_SNR import si_snr, si_snr_improvement, pit_align_by_sisnr
from Evaluation.SNR import snr


# ==========================================
# 1. 通信专属：误比特率 (BER) 计算
# ==========================================
def compute_ber(est_waveform, ref_waveform):
    """
    计算误比特率 (BER)，基于 BPSK 零点判决
    """
    est_bits = (est_waveform > 0).float()
    ref_bits = (ref_waveform > 0).float()
    errors = torch.sum(torch.abs(est_bits - ref_bits))
    total_bits = torch.numel(ref_bits)
    return (errors / total_bits).item()


# ==========================================
# 2. 辅助工具：日志与数据形状对齐
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


def ensure_mix_bt(mix):
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)
    elif mix.dim() == 3:
        if mix.size(1) == 1:
            mix = mix.squeeze(1)
        elif mix.size(0) == 1 and mix.size(2) > mix.size(1):
            mix = mix.squeeze(0)
        else:
            mix = mix.squeeze()
            if mix.dim() == 1: mix = mix.unsqueeze(0)
    return mix.contiguous()


def ensure_ref_b2t(ref):
    if ref.dim() == 2:
        if ref.size(0) == 2:
            ref = ref.unsqueeze(0)
        elif ref.size(1) == 2:
            ref = ref.transpose(0, 1).unsqueeze(0)
    elif ref.dim() == 3:
        if ref.size(2) == 2:
            ref = ref.permute(0, 2, 1)
        elif ref.size(0) == 2:
            ref = ref.permute(1, 0, 2)
    return ref.contiguous()


# ==========================================
# 3. 验证与终极测试循环
# ==========================================
def eval_valid(model, valid_loader, loss_fn, device):
    """训练过程中的验证循环"""
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for mix, targets in valid_loader:
            mix, targets = ensure_mix_bt(mix).to(device), ensure_ref_b2t(targets).to(device)
            # 使用混合精度降低 Transformer 显存消耗
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                est_sources = model(mix)
            min_len = min(est_sources.shape[-1], targets.shape[-1])
            loss = loss_fn(est_sources[..., :min_len], targets[..., :min_len])
            total_val_loss += loss.item()
    model.train()
    return total_val_loss / len(valid_loader)


def run_final_test(model, test_loader, sample_rate, device, output_dir, max_save_samples=20):
    """终极测试：计算 BER, SI-SNR, SI-SNRi, SNR"""
    print(f"\n==============================================")
    print(f" 开始执行 Transformer 终极测试 (CommDataset 标准)")
    print(f"==============================================")

    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = {"SI-SNR": [], "SI-SNRi": [], "SNR": [], "BER": []}
    saved = 0

    with torch.no_grad():
        for mix, targets in tqdm(test_loader, desc="Testing"):
            mix = ensure_mix_bt(mix).to(device)
            targets = ensure_ref_b2t(targets).to(device)

            est = model(mix)

            if est.dim() == 4 and est.size(1) == 1:
                est = est.squeeze(1)

            min_len = min(est.shape[-1], targets.shape[-1], mix.shape[-1])
            est = est[..., :min_len]
            targets = targets[..., :min_len]
            mix = mix[..., :min_len]

            # 1. PIT 对齐
            est_aligned, perm, sisnr_best = pit_align_by_sisnr(est, targets)
            est_aligned = est_aligned - est_aligned.mean(dim=-1, keepdim=True)

            # 2. 计算指标
            si_snr_vals = sisnr_best
            si_snri_vals = si_snr_improvement(est_aligned, targets, mix)
            snr_vals = 0.5 * (snr(est_aligned[:, 0, :], targets[:, 0, :]) + snr(est_aligned[:, 1, :], targets[:, 1, :]))

            all_metrics["SI-SNR"].extend(si_snr_vals.detach().cpu().tolist())
            all_metrics["SI-SNRi"].extend(si_snri_vals.detach().cpu().tolist())
            all_metrics["SNR"].extend(snr_vals.detach().cpu().tolist())

            # 3. 计算 BER
            ber_vals = []
            for b in range(est_aligned.shape[0]):
                b_err = 0.5 * (
                        compute_ber(est_aligned[b, 0], targets[b, 0]) +
                        compute_ber(est_aligned[b, 1], targets[b, 1])
                )
                ber_vals.append(b_err)
            all_metrics["BER"].extend(ber_vals)

            # 4. 保存音频样本
            for i in range(mix.size(0)):
                if saved >= max_save_samples: break
                base = f"transformer_test_{saved:04d}_perm{perm[i][0]}{perm[i][1]}"
                mix_i = mix[i].detach().cpu()
                s1_ref, s2_ref = targets[i, 0].detach().cpu(), targets[i, 1].detach().cpu()
                s1_est, s2_est = est_aligned[i, 0].detach().cpu(), est_aligned[i, 1].detach().cpu()

                torchaudio.save(os.path.join(output_dir, f"{base}_mix.wav"), mix_i.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_ref.wav"), s1_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_ref.wav"), s2_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_est.wav"), s1_est.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_est.wav"), s2_est.unsqueeze(0), sample_rate)
                saved += 1

    # 汇总写入 TXT
    results = {k: float(np.nanmean(v)) for k, v in all_metrics.items()}

    print("\n" + "=" * 40)
    print(f"[Transformer CommDataset] 最终成绩单:")
    for k, v in results.items():
        if k == "BER":
            print(f"  {k:8s}: {v:.6f} ({v * 100:.4f}%) - 误比特率")
        else:
            print(f"  {k:8s}: {v:.4f} dB")
    print("=" * 40 + "\n")

    ts = time.strftime("%Y%m%d-%H%M%S")
    txt_path = os.path.join(output_dir, f"final_metrics_transformer_comm_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Dual-Path Transformer on CommDataset\n")
        f.write(f"Sample rate: {sample_rate} Hz\n")
        f.write("-" * 40 + "\n")
        for k, v in results.items():
            if k == "BER":
                f.write(f"  {k:8s}: {v:.6f} ({v * 100:.4f}%)\n")
            else:
                f.write(f"  {k:8s}: {v:.4f} dB\n")
        f.write("-" * 40 + "\n")
    print(f"最终结果已保存至: {txt_path}")


# ==========================================
# 4. 主程序：极限训练 + 测试
# ==========================================
def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_name = f"train_transformer_comm_{timestamp}"

    log_file = os.path.join("../../logs", log_name + ".log")
    ckpt_dir = os.path.join("../../checkpoints", log_name)
    test_out_dir = os.path.join("../../Results/Transformer_Baselines_Comm", log_name)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)

    sys.stdout = Logger(log_file)
    print(f"Log file: {log_file}")

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = "../../Dataset/CommDataset"
    sample_rate = 100000

    # 【警告！】Transformer 在长度 32000 的序列上计算自注意力，矩阵维度直接起飞。
    # 强烈建议先用 batch_size=1 试水。如果秒崩，那是真的没法跑。
    batch_size = 16
    train_loader, valid_loader, test_loader = MakeDataLoader(root=root, batch_size=batch_size, num_workers=4,
                                                             segment=32000)

    # --- 极度缩水的 Transformer 配置 ---
    print("\nInitializing Dual-Path Transformer (DPTNet) from scratch...")
    model = DPTNet(
        n_src=2,
        out_chan=64,
        in_chan=64,
        ff_hid=128,  # 前馈网络维度
        chunk_size=100,  # 切块大小
        hop_size=50,
        n_repeats=6,  # 原版是 6 层 Transformer，这里砍到 2 层保命
        norm_type="gLN",
        mask_act="relu"
    ).to(device)

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Transformer 学习率不能太大
    loss_fn = pit_si_snr_loss
    num_epochs = 100

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, epochs=num_epochs, steps_per_epoch=len(train_loader), pct_start=0.1
    )

    best_val_loss = float("inf")
    patience, no_improve = 10, 0
    best_state_dict = None

    print("\n>>> 开始极限训练 Transformer 模型 <<<")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        for batch_idx, (mix, targets) in enumerate(train_loader, start=1):
            mix, targets = ensure_mix_bt(mix).to(device), ensure_ref_b2t(targets).to(device)
            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                est = model(mix)
                min_len = min(est.shape[-1], targets.shape[-1])
                loss = loss_fn(est[..., :min_len], targets[..., :min_len])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            ema_model.update_parameters(model)

            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"  Step {batch_idx}/{len(train_loader)} | Train Loss (−SI-SNR): {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = eval_valid(ema_model, valid_loader, loss_fn, device)
        print(f"Epoch {epoch + 1} 结束 | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss - 1e-3:
            best_val_loss = avg_val_loss
            no_improve = 0
            best_state_dict = copy.deepcopy(
                ema_model.module.state_dict() if hasattr(ema_model, "module") else ema_model.state_dict())
            print(f"  => 发现更好的验证集结果: {best_val_loss:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"触发早停 (Early Stopping)!")
                break

    print("\n>>> 训练结束，加载最佳 EMA 权重进行最终测试 <<<")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        torch.save(best_state_dict, os.path.join(ckpt_dir, "best_transformer_model.pt"))

    run_final_test(model, test_loader, sample_rate, device, test_out_dir, max_save_samples=20)


if __name__ == "__main__":
    main()