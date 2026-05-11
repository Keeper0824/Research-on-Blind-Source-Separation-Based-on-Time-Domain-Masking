import os
import time
import math
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from tqdm import tqdm

# Introduce DataLoader
from DataLoader.WSJDataLoader import MakeDataLoader as MakeWSJLoader

# Introduce evaluation indicators
from Evaluation.SI_SNR import si_snr, si_snr_improvement, pit_align_by_sisnr
from Evaluation.SNR import snr

try:
    from Evaluation.PESQ import pesqTest
    from Evaluation.STOI import stoiTest
except ImportError:
    pass


# ========================================================
# 1. Redefine the model structure (must be exactly the same as during training)
# ========================================================
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

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        x = x.float()
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


# ========================================================
# 2. Shape alignment auxiliary function
# ========================================================
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
        if ref.size(1) == 2:
            pass
        elif ref.size(2) == 2:
            ref = ref.permute(0, 2, 1)
        elif ref.size(0) == 2:
            ref = ref.permute(1, 0, 2)
    return ref.contiguous()


# ========================================================
# 3. Core testing and saving logic
# ========================================================
def evaluate_tf_baseline(model, loader, device, sample_rate, output_dir, max_save_samples=20):
    print(f"\n==============================================")
    print(f" 开始执行 TF-BLSTM 基准模型 终极测试")
    print(f"==============================================")

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = {"SI-SNR": [], "SI-SNRi": [], "SNR": [], "PESQ": [], "STOI": []}
    saved = 0

    with torch.no_grad():
        for mix, targets in tqdm(loader, desc="Evaluating TF-BLSTM"):
            mix = ensure_mix_bt(mix).to(device)
            targets = ensure_ref_b2t(targets).to(device)

            # Model forward inference
            est = model(mix)

            # Length alignment
            min_len = min(est.shape[-1], targets.shape[-1], mix.shape[-1])
            est = est[..., :min_len]
            targets = targets[..., :min_len]
            mix = mix[..., :min_len]

            # 1. PIT
            est_aligned, perm, sisnr_best = pit_align_by_sisnr(est, targets)

            # Implementation note.
            est_aligned = est_aligned - est_aligned.mean(dim=-1, keepdim=True)

            # 2. Calculate public indicators (Tensor calculation)
            si_snr_vals = sisnr_best
            si_snri_vals = si_snr_improvement(est_aligned, targets, mix)
            snr_vals = 0.5 * (snr(est_aligned[:, 0, :], targets[:, 0, :]) + snr(est_aligned[:, 1, :], targets[:, 1, :]))

            all_metrics["SI-SNR"].extend(si_snr_vals.detach().cpu().tolist())
            all_metrics["SI-SNRi"].extend(si_snri_vals.detach().cpu().tolist())
            all_metrics["SNR"].extend(snr_vals.detach().cpu().tolist())

            # 3. Calculate speech-specific indicators: PESQ and STOI (NumPy CPU calculation)
            est_cpu = est_aligned.detach().cpu().float().numpy()
            ref_cpu = targets.detach().cpu().float().numpy()

            for b in range(est_cpu.shape[0]):
                p_list, s_list = [], []
                for spk in range(2):
                    e = est_cpu[b, spk]
                    r = ref_cpu[b, spk]
                    p_list.append(pesqTest(e.astype(np.float32), r.astype(np.float32), sample_rate))
                    s_list.append(stoiTest(e.astype(np.float32), r.astype(np.float32), sample_rate))
                all_metrics["PESQ"].append(float(np.nanmean(p_list)))
                all_metrics["STOI"].append(float(np.nanmean(s_list)))

            # 4.
            for i in range(mix.size(0)):
                if saved >= max_save_samples:
                    break
                base = f"tf_wsj_test_{saved:04d}_perm{perm[i][0]}{perm[i][1]}"
                mix_i = mix[i].detach().cpu()
                s1_ref = targets[i, 0].detach().cpu()
                s2_ref = targets[i, 1].detach().cpu()
                s1_est = est_aligned[i, 0].detach().cpu()
                s2_est = est_aligned[i, 1].detach().cpu()

                torchaudio.save(os.path.join(output_dir, f"{base}_mix.wav"), mix_i.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_ref.wav"), s1_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_ref.wav"), s2_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s1_est.wav"), s1_est.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(output_dir, f"{base}_s2_est.wav"), s2_est.unsqueeze(0), sample_rate)
                saved += 1

    # Summary results
    def nanmean(x):
        x = np.array(x, dtype=np.float64)
        return float(np.nanmean(x)) if x.size else float("nan")

    results = {k: nanmean(v) for k, v in all_metrics.items()}

    print(f"\n[TF-BLSTM WSJ Baseline] 最终平均成绩 (PIT-aligned):")
    for k, v in results.items():
        print(f"  {k:8s}: {v:.4f}")

    return results


# ========================================================
# 4. Main program
# ========================================================
def main():
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = "../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"
    sample_rate = 8000

    model_path = "../../checkpoints/train_tf_baseline_wsj_20260216-232737/best_model.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到权重文件: {model_path}\n请修改脚本中的 model_path 变量！")

    # Results save path
    folder_name = os.path.basename(os.path.dirname(model_path))
    output_dir = os.path.join("../../Results/TF_Baseline_WSJ", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the test set (Batch_size can be set to 16 or 32 to speed up testing)
    print("Loading WSJ0-2Mix Test Set...")
    _, _, test_loader = MakeWSJLoader(root=root, batch_size=8, num_workers=4, segment=4.0, sample_rate=sample_rate)

    # 2. Initialize the model and load weights
    print(f"Loading weights from {model_path}...")
    model = STFTMaskNet(n_fft=512, hop_length=128).to(device)
    state_dict = torch.load(model_path, map_location=device)

    # Compatibility processing: Because EMA (AveragedModel) is used during training, sometimes saving state_dict directly will be prefixed with 'module.'.
    # So here is a simple key name cleaning
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    print("Weights loaded successfully!")

    # 3. Start evaluating
    results = evaluate_tf_baseline(model, test_loader, device, sample_rate, output_dir)

    # 4. Write report file
    ts = time.strftime("%Y%m%d-%H%M%S")
    txt_path = os.path.join(output_dir, f"tf_baseline_wsj_metrics_{ts}.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("TF-Domain Baseline (PIT-BLSTM) on WSJ0-2Mix\n")
        f.write(f"Weights: {model_path}\n")
        f.write(f"Sample rate: {sample_rate} Hz\n")
        f.write("-" * 40 + "\n")

        for k, v in results.items():
            if isinstance(v, float) and math.isnan(v):
                f.write(f"  {k:8s}: NaN\n")
            else:
                f.write(f"  {k:8s}: {v:.4f}\n")
        f.write("-" * 40 + "\n")

    print(f"\n所有评估结果和音频试听样本已保存至:\n{output_dir}")
    print(f"文本报告位于: {txt_path}")


if __name__ == "__main__":
    main()