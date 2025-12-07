import os
import sys
import time
import torch
import torchaudio
from torch.amp import GradScaler, autocast
from Model.Model_TasNet import Model
from LossFunction import BSSLoss
from DataLoader import MakeDataLoader


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
            with autocast(device_type="cuda", dtype=torch.float16):
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


def eval_test_full(model, test_loader, sample_rate, device, output_dir, max_samples=20):
    """
    Run through the entire test set (or first N samples) after training finishes.
    Save separated wav files for each test sample.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for mix, targets in test_loader:
            mix = mix.to(device)
            targets = targets.to(device)
            est_sources = model(mix.unsqueeze(1))
            # Align length
            min_len = min(est_sources.shape[-1], targets.shape[-1])
            est_sources = est_sources[..., :min_len]
            targets = targets[..., :min_len]
            mix = mix[..., :min_len]

            # Save each sample in the batch
            B = mix.size(0)
            for i in range(B):
                if count >= max_samples:
                    print("[Test] Reached max_samples limit.")
                    return

                mix_i = mix[i].cpu()
                s1_ref = targets[i, 0].cpu()
                s2_ref = targets[i, 1].cpu()
                s1_est = est_sources[i, 0].cpu()
                s2_est = est_sources[i, 1].cpu()
                base = f"test_{count}"
                torchaudio.save(os.path.join(output_dir, f"{base}_mix.wav"), mix_i.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_ref.wav"), s1_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_ref.wav"), s2_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_est.wav"), s1_est.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_est.wav"), s2_est.unsqueeze(0), sample_rate)
                count += 1
    print(f"[Test] Saved {count} test samples to: {output_dir}")



def main():
    # Basic run naming & logging setup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_name = f"train_convtasnet_{timestamp}"  # base name without extension
    log_file = os.path.join("logs", log_name + ".log")

    # Redirect stdout to logger
    sys.stdout = Logger(log_file)
    print(f"Log file: {log_file}")

    # Create per-run directories for checkpoints and outputs
    ckpt_root = "checkpoints"
    output_root = "outputs"
    ckpt_dir = os.path.join(ckpt_root, log_name)
    output_dir = os.path.join(output_root, log_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Output directory: {output_dir}")

    torch.backends.cudnn.benchmark = True
    # Data & device setup
    root = "Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"
    sample_rate = 8000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader = MakeDataLoader(root=root, batch_size=16, num_workers=8, segment=4.0,
                                                             sample_rate=sample_rate)

    # Model, optimizer, loss, AMP scaler
    model = Model(enc_dim=256, enc_kernel=16, enc_stride=8, bottleneck_channels=128, hidden_channels=256, num_blocks=8,
                  num_repeats=3, num_speakers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    loss_fn = BSSLoss.BSSLoss(mode="si_snr")
    num_epochs = 10
    global_step = 0

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = len(train_loader)

        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        for batch_idx, (mix, targets) in enumerate(train_loader, start=1):
            mix = mix.to(device)  # (B, T)
            targets = targets.to(device)  # (B, 2, T)

            optimizer.zero_grad()

            # Forward pass with AMP
            with autocast(device_type="cuda", dtype=torch.float16):
                est_sources = model(mix.unsqueeze(1))  # (B, 2, T_est)

                # Align length
                min_len = min(est_sources.shape[-1], targets.shape[-1])
                est_sources = est_sources[..., :min_len]
                targets = targets[..., :min_len]

                loss = loss_fn(est_sources, targets)

            # Backward with scaled gradients
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

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
        avg_val_loss = eval_valid(model, valid_loader, loss_fn, device)
        print(f"Epoch {epoch + 1} finished. Avg valid loss: {avg_val_loss:.4f}")

        # Save checkpoint for this epoch into the per-run ckpt_dir
        ckpt_path = os.path.join(ckpt_dir, f"conv_tasnet_epoch{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
            },
            ckpt_path,
        )
        print(f"[Checkpoint] Saved: {ckpt_path}")

    # Final checkpoint and final audio examples
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
    eval_test_full(model, test_loader, sample_rate, device, output_dir)


if __name__ == "__main__":
    main()

