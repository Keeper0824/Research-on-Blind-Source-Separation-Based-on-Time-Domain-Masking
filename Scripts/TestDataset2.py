import os
import sys
import time
import torch
import torchaudio
from torchaudio.models import ConvTasNet
from DataLoader import MakeDataLoader
from LossFunction.SI_SNR import negative_si_snr as si_snr_loss


class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        # 确保目录存在
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 兼容 Python 对 stdout.flush 的调用
        self.terminal.flush()
        self.log.flush()


def main():
    # ---------------- 日志设置：同时写入命令行和文件 ----------------
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join("../logs", f"train_convtasnet_{timestamp}.log")
    sys.stdout = Logger(log_file)
    print(f"Log file: {log_file}")

    # ---------------- 基本配置 ----------------
    root = "Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"
    sample_rate = 8000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainLoader, validLoader, testLoader = MakeDataLoader(
        root=root, batch_size=8, num_workers=8,
        segment=4.0, sample_rate=sample_rate
    )

    # torchaudio 自带 ConvTasNet_Test
    model = ConvTasNet(num_sources=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 用来保存模型参数的目录
    ckpt_dir = "../checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # --------- 训练 ---------
    num_epochs = 30
    global_step = 0

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = len(trainLoader)

        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        for batch_idx, (mix, targets) in enumerate(trainLoader, start=1):
            mix = mix.to(device)          # (B, T)
            targets = targets.to(device)  # (B, 2, T)
            est_sources = model(mix.unsqueeze(1))  # (B, 2, T)

            loss = si_snr_loss(est_sources, targets)

            optimizer.zero_grad()
            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1
            if global_step % 100 == 0:
                print(
                    f"Epoch {epoch + 1} | Step {global_step} "
                    f"({batch_idx}/{num_batches}) "
                    f"| Train loss (−SI-SNR): {loss.item():.4f}"
                )

        avg_train_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} finished. Avg train loss: {avg_train_loss:.4f}")

        # --------- 阶段性保存模型参数（每个 epoch 一个 checkpoint）---------
        ckpt_path = os.path.join(ckpt_dir, f"conv_tasnet_epoch{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_train_loss": avg_train_loss,
            },
            ckpt_path,
        )
        print(f"[Checkpoint] Saved: {ckpt_path}")

    # --------- 最终模型参数单独保存一份 ---------
    final_ckpt_path = os.path.join(ckpt_dir, "conv_tasnet_final.pt")
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_ckpt_path,
    )
    print(f"[Final] Model saved to: {final_ckpt_path}")

    # --------- 用测试集跑一个 batch，导出音频 ---------
    os.makedirs("../outputs", exist_ok=True)
    model.eval()
    with torch.no_grad():
        for mix, targets in testLoader:
            mix = mix.to(device)          # (B, T)
            targets = targets.to(device)  # (B, 2, T)

            est_sources = model(mix.unsqueeze(1))      # (B, 2, T)

            # 只导出 batch 里的第一个样本
            mix_0 = mix[0].cpu()                # (T,)
            s1_ref = targets[0, 0].cpu()
            s2_ref = targets[0, 1].cpu()
            s1_est = est_sources[0, 0].cpu()
            s2_est = est_sources[0, 1].cpu()

            # 保存 wav
            torchaudio.save("outputs/mix.wav",  mix_0.unsqueeze(0), sample_rate)
            torchaudio.save("outputs/s1_ref.wav", s1_ref.unsqueeze(0), sample_rate)
            torchaudio.save("outputs/s2_ref.wav", s2_ref.unsqueeze(0), sample_rate)
            torchaudio.save("outputs/s1_est.wav", s1_est.unsqueeze(0), sample_rate)
            torchaudio.save("outputs/s2_est.wav", s2_est.unsqueeze(0), sample_rate)

            print("已保存到 outputs/ 目录：mix / s1_ref / s2_ref / s1_est / s2_est")
            break


if __name__ == "__main__":
    main()
