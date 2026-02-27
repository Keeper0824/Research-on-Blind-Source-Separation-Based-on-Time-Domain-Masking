import os
import sys
import time
import torch
import torchaudio
import copy
import matplotlib.pyplot as plt
from contextlib import nullcontext
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# 引入你的多尺度模型
from Model.MultiScale.Model_TasNet_MultiScale import Model
# 引入 Loss
from LossFunction.SI_SNR import pit_si_snr_loss
# 引入 CommDataLoader
from DataLoader.CommDataLoader import MakeDataLoader


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


def eval_valid(model, valid_loader, loss_fn, device):
    """
    Validation loop using the EMA model for stability.
    """
    model.eval()
    total_val_loss = 0.0
    num_batches = len(valid_loader)

    with torch.no_grad():
        for mix, targets in valid_loader:
            mix = mix.to(device, non_blocking=True)  # (B, T)
            targets = targets.to(device, non_blocking=True)  # (B, 2, T)

            with autocast(device_type="cuda", dtype=torch.float16):
                est_sources = model(mix.unsqueeze(1))

                # Align length
                min_len = min(est_sources.shape[-1], targets.shape[-1])
                est_sources = est_sources[..., :min_len]
                targets = targets[..., :min_len]

                val_loss = loss_fn(est_sources, targets)

            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / num_batches
    return avg_val_loss


def eval_test_full(model, test_loader, loss_fn, sample_rate, device, output_dir, max_samples=20):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for mix, targets in test_loader:
            mix = mix.to(device)
            targets = targets.to(device)

            est_sources = model(mix.unsqueeze(1))

            min_len = min(est_sources.shape[-1], targets.shape[-1])
            est_sources = est_sources[..., :min_len]
            targets = targets[..., :min_len]
            mix = mix[..., :min_len]

            batch_loss = loss_fn(est_sources, targets)
            total_loss += batch_loss.item()
            num_batches += 1

            B = mix.size(0)
            for i in range(B):
                if count >= max_samples:
                    break

                mix_i = mix[i].cpu()
                s1_ref = targets[i, 0].cpu()
                s2_ref = targets[i, 1].cpu()
                s1_est = est_sources[i, 0].cpu()
                s2_est = est_sources[i, 1].cpu()

                base = f"test_{count}"
                # Save as WAV for inspection
                torchaudio.save(os.path.join(output_dir, f"{base}_mix.wav"), mix_i.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_ref.wav"), s1_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_ref.wav"), s2_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_est.wav"), s1_est.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_est.wav"), s2_est.unsqueeze(0), sample_rate)
                count += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
    print(f"[Test] Saved {count} samples. Avg Test Loss (−SI-SNR): {avg_loss:.4f} dB")


def _log_spec(x, n_fft=256, hop=64):
    x = x.unsqueeze(0)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=torch.hann_window(n_fft, device=x.device),
                      return_complex=True)
    mag = spec.abs().clamp_min(1e-8)
    return mag.log10().squeeze(0)


def visualize_separation_epoch(model, mix, targets, sample_rate, device, save_prefix):
    model.eval()
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

    with torch.no_grad():
        if mix.dim() == 1: mix = mix.unsqueeze(0)
        if targets.dim() == 2: targets = targets.unsqueeze(0)

        mix = mix.to(device)
        targets = targets.to(device)

        with autocast(device_type="cuda", dtype=torch.float16):
            est = model(mix.unsqueeze(1))

        min_len = min(est.shape[-1], targets.shape[-1], mix.shape[-1])
        est = est[..., :min_len]
        targets = targets[..., :min_len]
        mix = mix[..., :min_len]

        # Take first sample
        mix_ = mix[0].detach().float().cpu()
        s1_ref = targets[0, 0].detach().float().cpu()
        s2_ref = targets[0, 1].detach().float().cpu()
        s1_est = est[0, 0].detach().float().cpu()
        s2_est = est[0, 1].detach().float().cpu()

    # Plot Waveforms
    wavs = [mix_, s1_ref, s2_ref, s1_est, s2_est]
    names = ["mix", "s1_ref", "s2_ref", "s1_est", "s2_est"]
    plt.figure(figsize=(12, 10))
    for i, (w, n) in enumerate(zip(wavs, names), start=1):
        plt.subplot(5, 1, i)
        plt.plot(w.numpy())
        plt.title(n)
        plt.tight_layout()
    plt.savefig(save_prefix + "_wave.png", dpi=200)
    plt.close()

    # Plot Spectrograms
    specs = [_log_spec(w.to(device)) for w in wavs]  # Calculate on GPU then move back if needed
    plt.figure(figsize=(12, 10))
    for i, (sp, n) in enumerate(zip(specs, names), start=1):
        plt.subplot(5, 1, i)
        plt.imshow(sp.cpu().numpy(), origin="lower", aspect="auto")
        plt.title(n + " (Spectrogram)")
        plt.tight_layout()
    plt.savefig(save_prefix + "_spec.png", dpi=200)
    plt.close()


def plot_metrics(output_dir, epochs, train_loss, val_loss, grad_norm_epoch):
    os.makedirs(output_dir, exist_ok=True)
    # SI-SNR
    plt.figure()
    plt.plot(epochs, [-x for x in train_loss], label="Train SI-SNR")
    plt.plot(epochs, [-x for x in val_loss], label="Val SI-SNR")
    plt.xlabel("Epoch")
    plt.ylabel("SI-SNR (dB)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "curve_sisnr.png"))
    plt.close()

    # Gap
    plt.figure()
    plt.plot(epochs, [v - t for t, v in zip(train_loss, val_loss)], label="Gap (Val - Train)")
    plt.axhline(0, linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "curve_gap.png"))
    plt.close()


def main():
    # --- 1. Setup ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_name = f"train_multiscale_comm_{timestamp}"
    log_file = os.path.join("../../logs", log_name + ".log")

    # Logger setup
    sys.stdout = Logger(log_file)
    print(f"Log file: {log_file}")

    ckpt_root = "../../checkpoints"
    output_root = "../../outputs"
    ckpt_dir = os.path.join(ckpt_root, log_name)
    output_dir = os.path.join(output_root, log_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Data Loading (Comm0-2Dataset) ---
    root = "../../Dataset/Comm0-2Dataset"
    sample_rate = 100000  # Comm signal SR
    segment_points = 32000  # Input length in points
    batch_size = 16  # Adjust to 8 if OOM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Batch Size: {batch_size}")

    train_loader, valid_loader, test_loader = MakeDataLoader(
        root=root,
        batch_size=batch_size,
        num_workers=8,
        segment=segment_points
    )

    # Fixed sample for visualization
    fixed_mix, fixed_targets = next(iter(valid_loader))
    fixed_mix = fixed_mix[0]
    fixed_targets = fixed_targets[0]

    # --- 3. Model Init (Golden Config) ---
    print("Initializing Multi-Scale Model with Golden Config...")
    model = Model(
        enc_dim=516,  # Increased capacity
        enc_kernel=[8, 16, 24],  # Multi-scale kernels
        enc_stride=8,  # Large stride for receptive field
        bottleneck_channels=128,
        hidden_channels=256,
        num_blocks=8,
        num_repeats=3,
        num_speakers=2
    ).to(device)

    # --- 4. Optimizer, EMA, Scheduler (CRITICAL FIXES) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # [FIX 1] EMA Model for stable validation
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)).to(device)

    # [FIX 2] Learning Rate Scheduler
    num_epochs = 100
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1
    )

    scaler = GradScaler()
    loss_fn = pit_si_snr_loss

    # Tracking
    best_val_loss = float("inf")
    no_improve = 0
    patience = 12
    early_stop_warmup = 15
    epochs_hist, train_loss_hist, val_loss_hist, grad_norm_hist = [], [], [], []

    # --- 5. Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_gn = 0.0

        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        for batch_idx, (mix, targets) in enumerate(train_loader, start=1):
            mix = mix.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.float16):
                est = model(mix.unsqueeze(1))
                # Align
                min_len = min(est.shape[-1], targets.shape[-1])
                est = est[..., :min_len]
                targets = targets[..., :min_len]

                loss = loss_fn(est, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            # [FIX 3] Step Scheduler & Update EMA
            scheduler.step()
            ema_model.update_parameters(model)

            total_loss += loss.item()
            total_gn += float(gn)

            if batch_idx % 100 == 0:
                print(f"Step {batch_idx}/{steps_per_epoch} | Loss: {loss.item():.4f}")

        # Epoch Stats
        avg_train = total_loss / steps_per_epoch
        avg_gn = total_gn / steps_per_epoch

        # Validation (Using EMA Model!)
        avg_val = eval_valid(ema_model, valid_loader, loss_fn, device)

        print(f"Epoch {epoch + 1} Done. Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Grad: {avg_gn:.2f}")

        # History
        epochs_hist.append(epoch + 1)
        train_loss_hist.append(avg_train)
        val_loss_hist.append(avg_val)
        grad_norm_hist.append(avg_gn)
        plot_metrics(output_dir, epochs_hist, train_loss_hist, val_loss_hist, grad_norm_hist)

        # Checkpointing
        if avg_val < best_val_loss - 1e-4:
            best_val_loss = avg_val
            no_improve = 0
            # Save EMA weights as best
            best_state = copy.deepcopy(
                ema_model.module.state_dict() if hasattr(ema_model, "module") else ema_model.state_dict())
            print(f"-> New Best Model! Val Loss: {best_val_loss:.4f}")
        else:
            if epoch >= early_stop_warmup:
                no_improve += 1
                print(f"-> No improve for {no_improve} epochs.")

        if no_improve >= patience:
            print("Early Stopping Triggered.")
            break

        # Periodic Save
        if (epoch + 1) % 5 == 0:
            viz_path = os.path.join(output_dir, f"viz_epoch{epoch + 1:03d}")
            visualize_separation_epoch(ema_model, fixed_mix, fixed_targets, sample_rate, device, viz_path)

            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "ema_state": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }, ckpt_path)

    # --- 6. Finalization ---
    print("\nTraining Finished.")
    if best_state is not None:
        print(f"Loading best EMA model (Val Loss: {best_val_loss:.4f})...")
        model.load_state_dict(best_state)

        # Save Final Best
        final_path = os.path.join(ckpt_dir, "best_model.pt")
        torch.save({"model_state": best_state}, final_path)

        # Test Set Evaluation
        eval_test_full(model, test_loader, loss_fn, sample_rate, device, output_dir)

        # Final Viz
        visualize_separation_epoch(model, fixed_mix, fixed_targets, sample_rate, device,
                                   os.path.join(output_dir, "viz_final"))


if __name__ == "__main__":
    main()