print('Hello World')

# import os
# import sys
# import time
# import torch
# import torchaudio
# from torch.amp import GradScaler, autocast
# from Model.Model_TasNet import Model
# from LossFunction import BSSLoss
# from DataLoader import MakeDataLoader
#
#
# class Logger(object):
#     def __init__(self, log_path):
#         self.terminal = sys.stdout
#         # 确保目录存在
#         os.makedirs(os.path.dirname(log_path), exist_ok=True)
#         self.log = open(log_path, "a", encoding="utf-8", buffering=1)
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         # 兼容 Python 对 stdout.flush 的调用
#         self.terminal.flush()
#         self.log.flush()
#
#
# def eval_valid(model, validLoader, loss_fn, device):
#     model.eval()
#     total_val_loss = 0.0
#     num_batches = len(validLoader)
#
#     with torch.no_grad():
#         for mix, targets in validLoader:
#             mix = mix.to(device)  # (B, T)
#             targets = targets.to(device)  # (B, 2, T)
#
#             with autocast(device_type='cuda', dtype=torch.float16):
#                 est_sources = model(mix.unsqueeze(1))  # (B, 2, T_est)
#
#                 min_len = min(est_sources.shape[-1], targets.shape[-1])
#                 est_sources = est_sources[..., :min_len]
#                 targets = targets[..., :min_len]
#
#                 val_loss = loss_fn(est_sources, targets)
#
#             total_val_loss += val_loss.item()
#
#     avg_val_loss = total_val_loss / num_batches
#     model.train()
#     return avg_val_loss
#
#
# def eval_and_save(model, testLoader, sample_rate, device, epoch=None):
#     model.eval()
#     os.makedirs("outputs", exist_ok=True)
#
#     tag = f"_epoch{epoch}" if epoch is not None else ""
#
#     with torch.no_grad():
#         for mix, targets in testLoader:
#             mix = mix.to(device)  # (B, T)
#             targets = targets.to(device)  # (B, 2, T)
#
#             est_sources = model(mix.unsqueeze(1))  # (B, 2, T_est)
#
#             min_len = min(est_sources.shape[-1], targets.shape[-1])
#             est_sources = est_sources[..., :min_len]
#             targets = targets[..., :min_len]
#             mix = mix[..., :min_len]
#
#             mix_0 = mix[0].cpu()
#             s1_ref = targets[0, 0].cpu()
#             s2_ref = targets[0, 1].cpu()
#             s1_est = est_sources[0, 0].cpu()
#             s2_est = est_sources[0, 1].cpu()
#
#             torchaudio.save(f"outputs/mix{tag}.wav", mix_0.unsqueeze(0), sample_rate)
#             torchaudio.save(f"outputs/s1_ref{tag}.wav", s1_ref.unsqueeze(0), sample_rate)
#             torchaudio.save(f"outputs/s2_ref{tag}.wav", s2_ref.unsqueeze(0), sample_rate)
#             torchaudio.save(f"outputs/s1_est{tag}.wav", s1_est.unsqueeze(0), sample_rate)
#             torchaudio.save(f"outputs/s2_est{tag}.wav", s2_est.unsqueeze(0), sample_rate)
#
#             print(f"[Eval] Save to outputs/：*{tag}.wav")
#             break
#
#     model.train()
#
#
# def main():
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     log_file = os.path.join("logs", f"train_convtasnet_{timestamp}.log")
#     sys.stdout = Logger(log_file)
#     print(f"Log file: {log_file}")
#
#     root = "Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"
#     sample_rate = 8000
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     trainLoader, validLoader, testLoader = MakeDataLoader(
#         root=root, batch_size=24, num_workers=8,
#         segment=4.0, sample_rate=sample_rate
#     )
#
#     model = Model(enc_dim=256, enc_kernel=16, enc_stride=8, bottleneck_channels=128, hidden_channels=256, num_blocks=8,
#                   num_repeats=3, num_speakers=2).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     scaler = GradScaler()
#
#     ckpt_dir = "checkpoints"
#     os.makedirs(ckpt_dir, exist_ok=True)
#
#     num_epochs = 10
#     global_step = 0
#
#     loss_fn = BSSLoss.BSSLoss(mode="si_snr")
#
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0.0
#         num_batches = len(trainLoader)
#
#         print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")
#
#         for batch_idx, (mix, targets) in enumerate(trainLoader, start=1):
#             mix = mix.to(device)  # (B, T)
#             targets = targets.to(device)  # (B, 2, T)
#
#             optimizer.zero_grad()
#
#             with autocast(device_type='cuda', dtype=torch.float16):
#                 est_sources = model(mix.unsqueeze(1))  # (B, 2, T_est)
#
#                 min_len = min(est_sources.shape[-1], targets.shape[-1])
#                 est_sources = est_sources[..., :min_len]
#                 targets = targets[..., :min_len]
#
#                 loss = loss_fn(est_sources, targets)
#
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             scaler.step(optimizer)
#             scaler.update()
#
#             total_loss += loss.item()
#             global_step += 1
#             if global_step % 100 == 0:
#                 print(
#                     f"Epoch {epoch + 1} | Step {global_step} "
#                     f"({batch_idx}/{num_batches}) "
#                     f"| Train loss (−SI-SNR): {loss.item():.4f}"
#                 )
#
#         avg_train_loss = total_loss / num_batches
#         print(f"Epoch {epoch + 1} finished. Avg train loss: {avg_train_loss:.4f}")
#
#         avg_val_loss = eval_valid(model, validLoader, loss_fn, device)
#         print(f"Epoch {epoch + 1} finished. Avg valid loss: {avg_val_loss:.4f}")
#
#         ckpt_path = os.path.join(ckpt_dir, f"conv_tasnet_epoch{epoch + 1}.pt")
#         torch.save(
#             {
#                 "epoch": epoch + 1,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "avg_train_loss": avg_train_loss,
#                 "avg_val_loss": avg_val_loss,
#             },
#             ckpt_path,
#         )
#         print(f"[Checkpoint] Saved: {ckpt_path}")
#
#         eval_and_save(model, testLoader, sample_rate, device, epoch=epoch + 1)
#
#     final_ckpt_path = os.path.join(ckpt_dir, "conv_tasnet_final.pt")
#     torch.save(
#         {
#             "epoch": num_epochs,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#         },
#         final_ckpt_path,
#     )
#     print(f"[Final] Model saved to: {final_ckpt_path}")
#
#     eval_and_save(model, testLoader, sample_rate, device, epoch="final")
#
#
# if __name__ == "__main__":
#     main()