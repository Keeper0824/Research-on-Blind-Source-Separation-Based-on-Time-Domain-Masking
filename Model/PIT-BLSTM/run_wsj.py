import os
import sys
import time
import torch
import torch.nn as nn
import torchaudio
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from contextlib import nullcontext
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# Introducing data loading and loss functions
from DataLoader.WSJDataLoader import MakeDataLoader
from LossFunction.SI_SNR import pit_si_snr_loss


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


class STFTMaskNet(nn.Module):
    """Standard time-frequency domain benchmark model (PIT-BLSTM)"""

    def __init__(self, n_fft=512, hop_length=128, hidden_size=300, num_layers=3, num_spks=2):
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

    # Force mixed precision to be turned off in this forward propagation to prevent stft complex number operations from crashing under float16
    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        x = x.float()  # Make sure the input is float32
        B = x.shape[0]
        device = x.device

        window = torch.hann_window(self.n_fft).to(device)
        stft_res = torch.stft(x, self.n_fft, self.hop_length, window=window, return_complex=True)
        mag = stft_res.abs()
        phase = stft_res.angle()

        log_mag = torch.log1p(mag).permute(0, 2, 1)

        rnn_out, _ = self.lstm(log_mag)
        mask_out = self.linear(rnn_out)
        mask_out = self.activation(mask_out)

        mask_out = mask_out.view(B, -1, self.num_spks, self.n_fft // 2 + 1)
        mask_out = mask_out.permute(0, 2, 3, 1)

        est_mags = mag.unsqueeze(1) * mask_out

        est_wavs = []
        for i in range(self.num_spks):
            m = est_mags[:, i]
            comp = m * torch.exp(1j * phase)
            wav = torch.istft(comp, self.n_fft, self.hop_length, window=window, length=x.shape[-1])
            est_wavs.append(wav)

        return torch.stack(est_wavs, dim=1)


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


def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_name = f"train_tf_baseline_wsj_{timestamp}"
    log_file = os.path.join("../../logs", log_name + ".log")
    sys.stdout = Logger(log_file)
    print(f"Log file: {log_file} (TF-BLSTM on WSJ)")

    ckpt_dir = os.path.join("../../checkpoints", log_name)
    output_dir = os.path.join("../../outputs", log_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = "../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"

    train_loader, valid_loader, test_loader = MakeDataLoader(
        root=root, batch_size=256, num_workers=4, segment=4.0, sample_rate=8000
    )

    model = STFTMaskNet(n_fft=512, hop_length=128).to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=100, steps_per_epoch=len(train_loader), pct_start=0.1
    )
    loss_fn = pit_si_snr_loss

    best_val_loss = float("inf")
    patience, no_improve = 12, 0
    epochs_hist, train_loss_hist, val_loss_hist = [], [], []

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
        print(f"Epoch {epoch + 1} | Train SI-SNR: {-avg_train:.2f} | Val SI-SNR: {-avg_val:.2f}")

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


if __name__ == "__main__":
    main()