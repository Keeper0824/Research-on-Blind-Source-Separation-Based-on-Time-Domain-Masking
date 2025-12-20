import os
import sys
import time
import torch
import torchaudio
import copy
import matplotlib.pyplot as plt
from contextlib import nullcontext
from itertools import permutations
from torch.amp import GradScaler, autocast
from Model.Baseline.Model_TasNet import Model
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from LossFunction.SI_SNR import pit_si_snr_loss, si_snr
from DataLoader.WSJDataLoader import MakeDataLoader


class Logger(object):
    """
    Simple logger that writes both to terminal and a log file.
    """

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


def eval_valid(model, valid_loader, loss_fn, device):
    """
    Run one full pass on the validation set and return average loss.
    """
    model.eval()
    total_val_loss = 0.0
    num_batches = len(valid_loader)

    with torch.no_grad():
        for mix, targets in valid_loader:
            mix = mix.to(device)  # (B, T)
            targets = targets.to(device)  # (B, 2, T)

            # Use AMP for faster inference as well (optional but fine)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                est_sources = model(mix.unsqueeze(1))  # (B, 2, T_est)

            # Align time length in case of slight mismatch
            min_len = min(est_sources.shape[-1], targets.shape[-1])
            est_sources = est_sources[..., :min_len]
            targets = targets[..., :min_len]

            val_loss = loss_fn(est_sources, targets)

            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / num_batches
    model.train()
    return avg_val_loss


def eval_test_full(model, test_loader, loss_fn, sample_rate, device, output_dir, max_samples=20):
    """
    Run through the entire test set (or first N samples) after training finishes.
    Save separated wav files for each test sample.
    Also compute average test loss (e.g. negative SI-SNR).
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for mix, targets in test_loader:
            mix = mix.to(device)  # [B, T]
            targets = targets.to(device)  # [B, 2, T]  (假设 2 个源)

            est_sources = model(mix.unsqueeze(1))  # [B, 2, T’]

            min_len = min(est_sources.shape[-1], targets.shape[-1])
            est_sources = est_sources[..., :min_len]  # [B, 2, L]
            targets = targets[..., :min_len]  # [B, 2, L]
            mix = mix[..., :min_len]  # [B, L]

            batch_loss = loss_fn(est_sources, targets)
            total_loss += batch_loss.item()
            num_batches += 1

            B = mix.size(0)
            for i in range(B):
                if count >= max_samples:
                    print("[Test] Reached max_samples limit.")
                    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
                    print(f"[Test] Average test loss (−SI-SNR): {avg_loss:.4f} dB")
                    return

                mix_i = mix[i].cpu()
                s1_ref = targets[i, 0].cpu()
                s2_ref = targets[i, 1].cpu()
                s1_est = est_sources[i, 0].cpu()
                s2_est = est_sources[i, 1].cpu()

                base = f"test_{count}"
                torchaudio.save(os.path.join(output_dir, f"{base}_mix.wav"),
                                mix_i.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_ref.wav"),
                                s1_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_ref.wav"),
                                s2_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_est.wav"),
                                s1_est.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_est.wav"),
                                s2_est.unsqueeze(0), sample_rate)
                count += 1

    if num_batches > 0:
        avg_loss = total_loss / num_batches
    else:
        avg_loss = float('nan')
    print(f"[Test] Saved {count} test samples to: {output_dir}")
    print(f"[Test] Average test loss (−SI-SNR): {avg_loss:.4f} dB")


def _safe_autocast(device):
    return autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _log_spec(x, sample_rate, n_fft=256, hop=64):
    """
    x: Tensor [T] float
    return: Tensor [F, TT] log-magnitude spectrogram
    """
    x = x.unsqueeze(0)  # [1, T]
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=torch.hann_window(n_fft, device=x.device),
        return_complex=True
    )  # [1, F, TT]
    mag = spec.abs().clamp_min(1e-8)
    log_mag = mag.log10()  # log10
    return log_mag.squeeze(0)  # [F, TT]


def align_waveform(est, ref):
    num = torch.dot(est, ref)
    den = torch.dot(est, est) + 1e-8
    alpha = num / den
    return est * alpha


def pit_reorder(ests: torch.Tensor, refs: torch.Tensor, eps=1e-8):
    """
    ests: [B, S, T]
    refs: [B, S, T]
    return:
      ests_reordered: [B, S, T]
      best_perm: LongTensor [B, S]  (每个样本选到的 permutation)
    """
    B, S, T = ests.shape
    perms = list(permutations(range(S)))

    scores = []  # list of [B]
    for perm in perms:
        refp = refs[:, perm, :]
        # 平均每个源的 si-snr
        s = 0.0
        for k in range(S):
            s = s + si_snr(ests[:, k, :], refp[:, k, :], eps=eps)  # [B]
        scores.append(s / S)

    score_stack = torch.stack(scores, dim=1)  # [B, P]
    best_idx = torch.argmax(score_stack, dim=1)  # [B]
    best_perm = torch.tensor(perms, device=ests.device)[best_idx]  # [B, S]

    # 根据 best_perm 重排 ests
    ests_reordered = torch.stack(
        [ests[b, best_perm[b]] for b in range(B)], dim=0
    )
    return ests_reordered, best_perm


def visualize_separation_epoch(model, mix, targets, sample_rate, device, save_prefix):
    model.eval()
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

    with torch.no_grad():
        if mix.dim() == 1:
            mix = mix.unsqueeze(0)  # [1, T]
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)  # [1, 2, T]

        mix = mix.to(device)
        targets = targets.to(device)

        with _safe_autocast(device):
            est = model(mix.unsqueeze(1))  # [1, 2, T_est]

        # 1. 对齐长度
        min_len = min(est.shape[-1], targets.shape[-1], mix.shape[-1])
        est = est[..., :min_len]
        targets = targets[..., :min_len]
        mix = mix[..., :min_len]

        # 2. 【关键步骤】调用 PIT 重排，自动修正通道顺序
        # est: [B, S, T], targets: [B, S, T]
        est, _ = pit_reorder(est, targets)

        # 3. 提取单条数据
        mix_ = mix[0].detach().float().cpu()
        s1_ref = targets[0, 0].detach().float().cpu()
        s2_ref = targets[0, 1].detach().float().cpu()

        # 此时 est[0,0] 已经确保是对应 s1_ref 的了
        s1_est = est[0, 0].detach().float().cpu()
        s2_est = est[0, 1].detach().float().cpu()

    # 4. 波形图 (加入 align_waveform 修正幅度和相位)
    s1_est_aligned = align_waveform(s1_est, s1_ref)
    s2_est_aligned = align_waveform(s2_est, s2_ref)

    wavs = [mix_, s1_ref, s2_ref, s1_est_aligned, s2_est_aligned]
    names = ["mix", "s1_ref", "s2_ref", "s1_est", "s2_est"]

    plt.figure(figsize=(12, 10))
    for i, (w, n) in enumerate(zip(wavs, names), start=1):
        plt.subplot(5, 1, i)
        plt.plot(w.numpy())
        plt.title(n)
        plt.tight_layout()
    plt.savefig(save_prefix + "_wave.png", dpi=200)
    plt.close()

    # 5. 频谱图 (同理)
    def to_t(x):
        return x.to(torch.float32)

    # 使用对齐后的波形画谱，看起来更直观
    specs = [
        _log_spec(to_t(mix_), sample_rate),
        _log_spec(to_t(s1_ref), sample_rate),
        _log_spec(to_t(s2_ref), sample_rate),
        _log_spec(to_t(s1_est_aligned), sample_rate),
        _log_spec(to_t(s2_est_aligned), sample_rate)
    ]

    plt.figure(figsize=(12, 10))
    for i, (sp, n) in enumerate(zip(specs, names), start=1):
        plt.subplot(5, 1, i)
        plt.imshow(sp.numpy(), origin="lower", aspect="auto")
        plt.title(n + " (Spectrogram)")
        plt.tight_layout()
    plt.savefig(save_prefix + "_spec.png", dpi=200)
    plt.close()

    model.train()


def plot_metrics(output_dir, epochs, train_loss, val_loss, grad_norm_epoch):
    os.makedirs(output_dir, exist_ok=True)

    # 1) SI-SNR(dB) = -loss（你的 loss 是 -SI-SNR）
    train_sisnr = [-x for x in train_loss]
    val_sisnr = [-x for x in val_loss]
    plt.figure()
    plt.plot(epochs, train_sisnr, label="train SI-SNR (dB)")
    plt.plot(epochs, val_sisnr, label="val SI-SNR (dB)")
    plt.xlabel("epoch");
    plt.ylabel("SI-SNR (dB)")
    plt.legend();
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curve_sisnr.png"), dpi=200)
    plt.close()

    # 2) gap = val_loss - train_loss（看过拟合很直观）
    gap = [v - t for t, v in zip(train_loss, val_loss)]
    plt.figure()
    plt.plot(epochs, gap, label="gap = val_loss - train_loss")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("epoch");
    plt.ylabel("loss gap")
    plt.legend();
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curve_gap.png"), dpi=200)
    plt.close()

    # 3) grad norm（每个 epoch 平均值）
    plt.figure()
    plt.plot(epochs, grad_norm_epoch, label="avg grad norm / epoch")
    plt.xlabel("epoch");
    plt.ylabel("grad norm")
    plt.legend();
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curve_gradnorm.png"), dpi=200)
    plt.close()


def main():
    # Basic run naming & logging setup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_name = f"train_convtasnet_{timestamp}"  # base name without extension
    log_file = os.path.join("../../logs", log_name + ".log")

    # Redirect stdout to logger
    sys.stdout = Logger(log_file)
    print(f"Log file: {log_file} (Baseline)")

    # Create per-run directories for checkpoints and outputs
    ckpt_root = "../../checkpoints"
    output_root = "../../outputs"
    ckpt_dir = os.path.join(ckpt_root, log_name)
    output_dir = os.path.join(output_root, log_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Output directory: {output_dir}")
    epochs_hist = []
    train_loss_hist = []
    val_loss_hist = []
    grad_norm_hist = []

    torch.backends.cudnn.benchmark = True
    # Data & device setup
    root = "../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"
    sample_rate = 8000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader = MakeDataLoader(root=root, batch_size=16, num_workers=8, segment=4.0,
                                                             sample_rate=sample_rate)
    # 固定一个 valid 样例：每个 epoch 都用同一个，方便看训练进步
    fixed_mix, fixed_targets = next(iter(valid_loader))
    fixed_mix = fixed_mix[0]  # [T]
    fixed_targets = fixed_targets[0]  # [2, T]

    # Model, optimizer, loss, AMP scaler
    # MutiScaleEncoder
    model = Model(enc_dim=512, enc_kernel=16, enc_stride=8, bottleneck_channels=128, hidden_channels=256, num_blocks=8,
                  num_repeats=3, num_speakers=2).to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scaler = GradScaler()
    loss_fn = pit_si_snr_loss
    num_epochs = 100
    global_step = 0
    patience = 5
    best_val_loss = float("inf")
    no_improve_epochs = 0
    best_state_dict = None
    epoch = 0

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-3,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1
    )

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_gn = 0.0
        total_loss = 0.0
        num_batches = len(train_loader)

        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        for batch_idx, (mix, targets) in enumerate(train_loader, start=1):
            mix = mix.to(device)  # (B, T)
            targets = targets.to(device)  # (B, 2, T)

            optimizer.zero_grad()

            # Forward pass with AMP
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                est_sources = model(mix.unsqueeze(1))  # (B, 2, T_est)

                # Align length
                min_len = min(est_sources.shape[-1], targets.shape[-1])
                est_sources = est_sources[..., :min_len]
                targets = targets[..., :min_len]

                loss = loss_fn(est_sources, targets)

            # Backward with scaled gradients
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # total_gn += float(gn)
            # scaler.step(optimizer)
            # scaler.update()

            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            total_gn += float(gn)
            optimizer.step()
            scheduler.step()
            ema_model.update_parameters(model)

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

        # Validation
        avg_val_loss = eval_valid(ema_model, valid_loader, loss_fn, device)
        print(f"Epoch {epoch + 1} finished. Avg valid loss: {avg_val_loss:.4f}")

        avg_gn = total_gn / num_batches

        epochs_hist.append(epoch + 1)
        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)
        grad_norm_hist.append(avg_gn)

        plot_metrics(output_dir, epochs_hist, train_loss_hist, val_loss_hist, grad_norm_hist)
        print("[Plot] Saved: curve_sisnr.png, curve_gap.png, curve_gradnorm.png")

        if avg_val_loss < best_val_loss - 1e-4:  # 可以加个很小的 margin 防止浮点抖动
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            best_state_dict = copy.deepcopy(
                ema_model.module.state_dict() if hasattr(ema_model, "module") else ema_model.state_dict())
            print(f"Validation improved, new best val loss: {best_val_loss:.4f}")
        else:
            no_improve_epochs += 1
            print(
                f"No improvement on valid for {no_improve_epochs} epoch(s) "
                f"(best: {best_val_loss:.4f})"
            )
            if no_improve_epochs >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1} "
                    f"(patience={patience})"
                )
                break

        # Save checkpoint for this epoch into the per-run ckpt_dir
        ckpt_path = os.path.join(ckpt_dir, f"conv_tasnet_epoch{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
            },
            ckpt_path,
        )
        print(f"[Checkpoint] Saved: {ckpt_path}")
        # 每轮保存一次可视化（波形+频谱）
        viz_prefix = os.path.join(output_dir, f"viz_epoch{epoch + 1:03d}")
        visualize_separation_epoch(ema_model, fixed_mix, fixed_targets, sample_rate, device, viz_prefix)
        print(f"[Viz] Saved: {viz_prefix}_wave.png and {viz_prefix}_spec.png")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model with val loss = {best_val_loss:.4f}")
    else:
        print("Warning: best_state_dict is None, did training even run?")

    # Final checkpoint and final audio examples
    final_ckpt_path = os.path.join(ckpt_dir, "conv_tasnet_final.pt")
    if hasattr(ema_model, "module"):
        ema_weights = ema_model.module.state_dict()
    else:
        ema_weights = ema_model.state_dict()
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_weights,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_ckpt_path,
    )
    print(f"[Final] Model saved to: {final_ckpt_path}")
    eval_test_full(model, test_loader, loss_fn, sample_rate, device, output_dir)
    final_viz_prefix = os.path.join(output_dir, "viz_final_best")
    visualize_separation_epoch(model, fixed_mix, fixed_targets, sample_rate, device, final_viz_prefix)
    print(f"[Viz] Saved FINAL: {final_viz_prefix}_wave.png and {final_viz_prefix}_spec.png")


if __name__ == "__main__":
    main()

# import os
# import sys
# import time
# import torch
# import torchaudio
# import copy
# import matplotlib.pyplot as plt
# from contextlib import nullcontext
# from torch.amp import GradScaler, autocast
# from Model.Baseline.Model_TasNet import Model
# from LossFunction.SI_SNR import pit_si_snr_loss
# from DataLoader.WSJDataLoader import MakeDataLoader
#
#
# class Logger(object):
#     """
#     Simple logger that writes both to terminal and a log file.
#     """
#
#     def __init__(self, log_path: str):
#         self.terminal = sys.stdout
#         os.makedirs(os.path.dirname(log_path), exist_ok=True)
#         self.log = open(log_path, "a", encoding="utf-8", buffering=1)
#
#     def write(self, message: str):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()
#
#
# def eval_valid(model, valid_loader, loss_fn, device):
#     """
#     Run one full pass on the validation set and return average loss.
#     """
#     model.eval()
#     total_val_loss = 0.0
#     num_batches = len(valid_loader)
#
#     with torch.no_grad():
#         for mix, targets in valid_loader:
#             mix = mix.to(device)  # (B, T)
#             targets = targets.to(device)  # (B, 2, T)
#
#             # Use AMP for faster inference as well (optional but fine)
#             with autocast(device_type="cuda", dtype=torch.float16):
#                 est_sources = model(mix.unsqueeze(1))  # (B, 2, T_est)
#
#                 # Align time length in case of slight mismatch
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
# def eval_test_full(model, test_loader, loss_fn, sample_rate, device, output_dir, max_samples=20):
#     """
#     Run through the entire test set (or first N samples) after training finishes.
#     Save separated wav files for each test sample.
#     Also compute average test loss (e.g. negative SI-SNR).
#     """
#     model.eval()
#     os.makedirs(output_dir, exist_ok=True)
#
#     count = 0
#     total_loss = 0.0
#     num_batches = 0
#
#     with torch.no_grad():
#         for mix, targets in test_loader:
#             mix = mix.to(device)  # [B, T]
#             targets = targets.to(device)  # [B, 2, T]  (假设 2 个源)
#
#             est_sources = model(mix.unsqueeze(1))  # [B, 2, T’]
#
#             min_len = min(est_sources.shape[-1], targets.shape[-1])
#             est_sources = est_sources[..., :min_len]  # [B, 2, L]
#             targets = targets[..., :min_len]  # [B, 2, L]
#             mix = mix[..., :min_len]  # [B, L]
#
#             batch_loss = loss_fn(est_sources, targets)
#             total_loss += batch_loss.item()
#             num_batches += 1
#
#             B = mix.size(0)
#             for i in range(B):
#                 if count >= max_samples:
#                     print("[Test] Reached max_samples limit.")
#                     avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
#                     print(f"[Test] Average test loss (−SI-SNR): {avg_loss:.4f} dB")
#                     return
#
#                 mix_i = mix[i].cpu()
#                 s1_ref = targets[i, 0].cpu()
#                 s2_ref = targets[i, 1].cpu()
#                 s1_est = est_sources[i, 0].cpu()
#                 s2_est = est_sources[i, 1].cpu()
#
#                 base = f"test_{count}"
#                 torchaudio.save(os.path.join(output_dir, f"{base}_mix.wav"),
#                                 mix_i.unsqueeze(0), sample_rate)
#                 torchaudio.save(os.path.join(output_dir, f"{base}_s1_ref.wav"),
#                                 s1_ref.unsqueeze(0), sample_rate)
#                 torchaudio.save(os.path.join(output_dir, f"{base}_s2_ref.wav"),
#                                 s2_ref.unsqueeze(0), sample_rate)
#                 torchaudio.save(os.path.join(output_dir, f"{base}_s1_est.wav"),
#                                 s1_est.unsqueeze(0), sample_rate)
#                 torchaudio.save(os.path.join(output_dir, f"{base}_s2_est.wav"),
#                                 s2_est.unsqueeze(0), sample_rate)
#                 count += 1
#
#     if num_batches > 0:
#         avg_loss = total_loss / num_batches
#     else:
#         avg_loss = float('nan')
#     print(f"[Test] Saved {count} test samples to: {output_dir}")
#     print(f"[Test] Average test loss (−SI-SNR): {avg_loss:.4f} dB")
#
# def _safe_autocast(device):
#     return autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
#
# def _log_spec(x, sample_rate, n_fft=256, hop=64):
#     """
#     x: Tensor [T] float
#     return: Tensor [F, TT] log-magnitude spectrogram
#     """
#     x = x.unsqueeze(0)  # [1, T]
#     spec = torch.stft(
#         x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
#         window=torch.hann_window(n_fft, device=x.device),
#         return_complex=True
#     )  # [1, F, TT]
#     mag = spec.abs().clamp_min(1e-8)
#     log_mag = mag.log10()  # log10
#     return log_mag.squeeze(0)  # [F, TT]
#
# def visualize_separation_epoch(model, mix, targets, sample_rate, device, save_prefix):
#     """
#     固定样例可视化：波形图 + 频谱图
#     mix: [1, T] or [T]
#     targets: [1, 2, T] or [2, T]
#     """
#     model.eval()
#     os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
#
#     with torch.no_grad():
#         # 统一 shape
#         if mix.dim() == 1:
#             mix = mix.unsqueeze(0)          # [1, T]
#         if targets.dim() == 2:
#             targets = targets.unsqueeze(0)  # [1, 2, T]
#
#         mix = mix.to(device)
#         targets = targets.to(device)
#
#         with _safe_autocast(device):
#             est = model(mix.unsqueeze(1))  # [1, 2, T’]
#
#         # 对齐长度
#         min_len = min(est.shape[-1], targets.shape[-1], mix.shape[-1])
#         est = est[..., :min_len]
#         targets = targets[..., :min_len]
#         mix = mix[..., :min_len]
#
#         # 取出单条
#         mix_ = mix[0].detach().float().cpu()
#         s1_ref = targets[0, 0].detach().float().cpu()
#         s2_ref = targets[0, 1].detach().float().cpu()
#         s1_est = est[0, 0].detach().float().cpu()
#         s2_est = est[0, 1].detach().float().cpu()
#
#     # ============ 1) 波形图 ============
#     wavs = [mix_, s1_ref, s2_ref, s1_est, s2_est]
#     names = ["mix", "s1_ref", "s2_ref", "s1_est", "s2_est"]
#
#     plt.figure(figsize=(12, 10))
#     for i, (w, n) in enumerate(zip(wavs, names), start=1):
#         plt.subplot(5, 1, i)
#         plt.plot(w.numpy())
#         plt.title(n)
#         plt.tight_layout()
#     plt.savefig(save_prefix + "_wave.png", dpi=200)
#     plt.close()
#
#     # ============ 2) 频谱图 ============
#     # 用 CPU 也行（快且稳），转回 tensor 方便 stft
#     def to_t(x):
#         return x.to(torch.float32)
#
#     mix_t = to_t(mix_)
#     s1_ref_t = to_t(s1_ref)
#     s2_ref_t = to_t(s2_ref)
#     s1_est_t = to_t(s1_est)
#     s2_est_t = to_t(s2_est)
#
#     # stft 放 CPU 上跑就行（省 GPU）
#     mix_spec = _log_spec(mix_t, sample_rate)
#     s1_ref_spec = _log_spec(s1_ref_t, sample_rate)
#     s2_ref_spec = _log_spec(s2_ref_t, sample_rate)
#     s1_est_spec = _log_spec(s1_est_t, sample_rate)
#     s2_est_spec = _log_spec(s2_est_t, sample_rate)
#
#     specs = [mix_spec, s1_ref_spec, s2_ref_spec, s1_est_spec, s2_est_spec]
#
#     plt.figure(figsize=(12, 10))
#     for i, (sp, n) in enumerate(zip(specs, names), start=1):
#         plt.subplot(5, 1, i)
#         plt.imshow(sp.numpy(), origin="lower", aspect="auto")
#         plt.title(n + " (log10 |STFT|)")
#         plt.tight_layout()
#     plt.savefig(save_prefix + "_spec.png", dpi=200)
#     plt.close()
#
#     model.train()
#
# def plot_metrics(output_dir, epochs, train_loss, val_loss, grad_norm_epoch):
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 1) SI-SNR(dB) = -loss（你的 loss 是 -SI-SNR）
#     train_sisnr = [-x for x in train_loss]
#     val_sisnr = [-x for x in val_loss]
#     plt.figure()
#     plt.plot(epochs, train_sisnr, label="train SI-SNR (dB)")
#     plt.plot(epochs, val_sisnr, label="val SI-SNR (dB)")
#     plt.xlabel("epoch"); plt.ylabel("SI-SNR (dB)")
#     plt.legend(); plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "curve_sisnr.png"), dpi=200)
#     plt.close()
#
#     # 2) gap = val_loss - train_loss（看过拟合很直观）
#     gap = [v - t for t, v in zip(train_loss, val_loss)]
#     plt.figure()
#     plt.plot(epochs, gap, label="gap = val_loss - train_loss")
#     plt.axhline(0.0, linestyle="--")
#     plt.xlabel("epoch"); plt.ylabel("loss gap")
#     plt.legend(); plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "curve_gap.png"), dpi=200)
#     plt.close()
#
#     # 3) grad norm（每个 epoch 平均值）
#     plt.figure()
#     plt.plot(epochs, grad_norm_epoch, label="avg grad norm / epoch")
#     plt.xlabel("epoch"); plt.ylabel("grad norm")
#     plt.legend(); plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "curve_gradnorm.png"), dpi=200)
#     plt.close()
#
#
# def main():
#     # Basic run naming & logging setup
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     log_name = f"train_convtasnet_{timestamp}"  # base name without extension
#     log_file = os.path.join("../../logs", log_name + ".log")
#
#     # Redirect stdout to logger
#     sys.stdout = Logger(log_file)
#     print(f"Log file: {log_file}")
#
#     # Create per-run directories for checkpoints and outputs
#     ckpt_root = "../../checkpoints"
#     output_root = "../../outputs"
#     ckpt_dir = os.path.join(ckpt_root, log_name)
#     output_dir = os.path.join(output_root, log_name)
#     os.makedirs(ckpt_dir, exist_ok=True)
#     os.makedirs(output_dir, exist_ok=True)
#
#     print(f"Checkpoint directory: {ckpt_dir}")
#     print(f"Output directory: {output_dir}")
#     epochs_hist = []
#     train_loss_hist = []
#     val_loss_hist = []
#     grad_norm_hist = []
#
#     torch.backends.cudnn.benchmark = True
#     # Data & device setup
#     root = "../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"
#     sample_rate = 8000
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     train_loader, valid_loader, test_loader = MakeDataLoader(root=root, batch_size=16, num_workers=8, segment=4.0,
#                                                              sample_rate=sample_rate)
#     # 固定一个 valid 样例：每个 epoch 都用同一个，方便看训练进步
#     fixed_mix, fixed_targets = next(iter(valid_loader))
#     fixed_mix = fixed_mix[0]  # [T]
#     fixed_targets = fixed_targets[0]  # [2, T]
#
#     # Model, optimizer, loss, AMP scaler
#     # Baseline
#     model = Model(enc_dim=512, enc_kernel=16, enc_stride=8, bottleneck_channels=128, hidden_channels=256, num_blocks=8,
#                   num_repeats=3, num_speakers=2).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     scaler = GradScaler()
#     # loss_fn = BSSLoss.BSSLoss(mode="si_snr")
#     loss_fn = pit_si_snr_loss
#     num_epochs = 100
#     global_step = 0
#     patience = 5
#     best_val_loss = float("inf")
#     no_improve_epochs = 0
#     best_state_dict = None
#     epoch = 0
#
#     # Training loop
#     model.train()
#     for epoch in range(num_epochs):
#         total_gn = 0.0
#         total_loss = 0.0
#         num_batches = len(train_loader)
#
#         print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")
#
#         for batch_idx, (mix, targets) in enumerate(train_loader, start=1):
#             mix = mix.to(device)  # (B, T)
#             targets = targets.to(device)  # (B, 2, T)
#
#             optimizer.zero_grad()
#
#             # Forward pass with AMP
#             with autocast(device_type="cuda", dtype=torch.float16):
#                 est_sources = model(mix.unsqueeze(1))  # (B, 2, T_est)
#
#                 # Align length
#                 min_len = min(est_sources.shape[-1], targets.shape[-1])
#                 est_sources = est_sources[..., :min_len]
#                 targets = targets[..., :min_len]
#
#                 loss = loss_fn(est_sources, targets)
#
#             # Backward with scaled gradients
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             total_gn += float(gn)
#             scaler.step(optimizer)
#             scaler.update()
#
#             total_loss += loss.item()
#             global_step += 1
#
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
#         # Validation
#         avg_val_loss = eval_valid(model, valid_loader, loss_fn, device)
#         print(f"Epoch {epoch + 1} finished. Avg valid loss: {avg_val_loss:.4f}")
#
#         avg_gn = total_gn / num_batches
#
#         epochs_hist.append(epoch + 1)
#         train_loss_hist.append(avg_train_loss)
#         val_loss_hist.append(avg_val_loss)
#         grad_norm_hist.append(avg_gn)
#
#         plot_metrics(output_dir, epochs_hist, train_loss_hist, val_loss_hist, grad_norm_hist)
#         print("[Plot] Saved: curve_sisnr.png, curve_gap.png, curve_gradnorm.png")
#
#         if avg_val_loss < best_val_loss - 1e-4:  # 可以加个很小的 margin 防止浮点抖动
#             best_val_loss = avg_val_loss
#             no_improve_epochs = 0
#             best_state_dict = copy.deepcopy(model.state_dict())
#             print(f"Validation improved, new best val loss: {best_val_loss:.4f}")
#         else:
#             no_improve_epochs += 1
#             print(
#                 f"⚠ No improvement on valid for {no_improve_epochs} epoch(s) "
#                 f"(best: {best_val_loss:.4f})"
#             )
#             if no_improve_epochs >= patience:
#                 print(
#                     f"Early stopping triggered at epoch {epoch + 1} "
#                     f"(patience={patience})"
#                 )
#                 break
#
#         # Save checkpoint for this epoch into the per-run ckpt_dir
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
#         # 每轮保存一次可视化（波形+频谱）
#         viz_prefix = os.path.join(output_dir, f"viz_epoch{epoch + 1:03d}")
#         visualize_separation_epoch(model, fixed_mix, fixed_targets, sample_rate, device, viz_prefix)
#         print(f"[Viz] Saved: {viz_prefix}_wave.png and {viz_prefix}_spec.png")
#
#     if best_state_dict is not None:
#         model.load_state_dict(best_state_dict)
#         print(f"Loaded best model with val loss = {best_val_loss:.4f}")
#     else:
#         print("Warning: best_state_dict is None, did training even run?")
#
#     # Final checkpoint and final audio examples
#     final_ckpt_path = os.path.join(ckpt_dir, "conv_tasnet_final.pt")
#     torch.save(
#         {
#             "epoch": epoch + 1,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#         },
#         final_ckpt_path,
#     )
#     print(f"[Final] Model saved to: {final_ckpt_path}")
#     eval_test_full(model, test_loader, loss_fn, sample_rate, device, output_dir)
#     final_viz_prefix = os.path.join(output_dir, "viz_final_best")
#     visualize_separation_epoch(model, fixed_mix, fixed_targets, sample_rate, device, final_viz_prefix)
#     print(f"[Viz] Saved FINAL: {final_viz_prefix}_wave.png and {final_viz_prefix}_spec.png")
#
#
# if __name__ == "__main__":
#     main()
