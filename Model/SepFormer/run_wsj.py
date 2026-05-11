import os
import time
import math
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# ========================================================
# [Security Patch 1] Prevent PyTorch 2.6 from intercepting old pre-trained weights
# ========================================================
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load

# ========================================================
# [Dependency Patch 2 final Edition] Fix parameter conflicts + intercept 404 errors and forge files
# ========================================================
import huggingface_hub

_orig_hf_hub_download = huggingface_hub.hf_hub_download


def _patched_hf_hub_download(*args, **kwargs):
    # Fix obsolete parameters
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')

    # Get the name of the file it wants to download
    filename = kwargs.get('filename', args[1] if len(args) > 1 else '')

    try:
        return _orig_hf_hub_download(*args, **kwargs)
    except Exception as e:
        # If custom.py is requested and a 404 Not Found error is encountered
        if filename == "custom.py" and "404" in str(e):
            print("\n[拦截提醒] 发现 SpeechBrain 索要不存在的 custom.py，已自动伪造空文件放行！")
            dummy_path = os.path.join(os.getcwd(), "dummy_custom.py")
            if not os.path.exists(dummy_path):
                with open(dummy_path, "w", encoding="utf-8") as f:
                    f.write("# 这是一个为了骗过 SpeechBrain 404 报错的空文件\n")
            return dummy_path
        raise e


huggingface_hub.hf_hub_download = _patched_hf_hub_download

# ========================================================
# [Permissions Patch 3] Solve the problem that Windows cannot create soft links (WinError 1314)
# ========================================================
import pathlib
import shutil

_orig_symlink_to = pathlib.Path.symlink_to


def _patched_symlink_to(self, target, target_is_directory=False):
    try:
        _orig_symlink_to(self, target, target_is_directory)
    except OSError as e:
        # If WinError 1314 (no privileges), fall back to directly physically copying the file
        if getattr(e, 'winerror', None) == 1314:
            if target.is_dir():
                shutil.copytree(target, self)
            else:
                shutil.copy(target, self)
        else:
            raise


pathlib.Path.symlink_to = _patched_symlink_to
# ========================================================

# Introducing the separation module of SpeechBrain
try:
    from speechbrain.inference.separation import SepformerSeparation
except ImportError:
    raise ImportError("请先安装 SpeechBrain: pip install speechbrain")

# DataLoader
from DataLoader.WSJDataLoader import MakeDataLoader as MakeWSJLoader
from Evaluation.SI_SNR import si_snr, si_snr_improvement, pit_align_by_sisnr
from Evaluation.SNR import snr

try:
    from Evaluation.PESQ import pesqTest
    from Evaluation.STOI import stoiTest
except ImportError:
    pass


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


def eval_speechbrain_model(model_name, model, loader, device, sample_rate, output_dir, max_save_samples=20):
    print(f"\n==============================================")
    print(f" 正在运行 SpeechBrain [{model_name}] 基准测试")
    print(f"==============================================")

    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    all_metrics = {
        "SI-SNR": [], "SI-SNRi": [], "SNR": [], "PESQ": [], "STOI": []
    }
    saved = 0

    with torch.no_grad():
        for mix, targets in tqdm(loader, desc=f"Evaluating {model_name}"):
            mix = ensure_mix_bt(mix).to(device)
            targets = ensure_ref_b2t(targets).to(device)

            # Forward propagation of SpeechBrain SepFormer [B, T]
            est = model.separate_batch(mix)

            # Convert shape: [Batch, Time, Spk] -> [Batch, Spk, Time]
            if est.dim() == 3:
                est = est.permute(0, 2, 1)

            # alignment length
            min_len = min(est.shape[-1], targets.shape[-1], mix.shape[-1])
            est = est[..., :min_len]
            targets = targets[..., :min_len]
            mix = mix[..., :min_len]

            # 1. PIT
            est_aligned, perm, sisnr_best = pit_align_by_sisnr(est, targets)

            # Implementation note.
            est_aligned = est_aligned - est_aligned.mean(dim=-1, keepdim=True)

            # 2.
            si_snr_vals = sisnr_best
            si_snri_vals = si_snr_improvement(est_aligned, targets, mix)
            snr_vals = 0.5 * (snr(est_aligned[:, 0, :], targets[:, 0, :]) + snr(est_aligned[:, 1, :], targets[:, 1, :]))

            all_metrics["SI-SNR"].extend(si_snr_vals.detach().cpu().tolist())
            all_metrics["SI-SNRi"].extend(si_snri_vals.detach().cpu().tolist())
            all_metrics["SNR"].extend(snr_vals.detach().cpu().tolist())

            # 3. CPU PESQ STOI
            pesq_vals = []
            stoi_vals = []
            est_cpu = est_aligned.detach().cpu().float().numpy()
            ref_cpu = targets.detach().cpu().float().numpy()

            for b in range(est_cpu.shape[0]):
                p_list, s_list = [], []
                for spk in range(2):
                    e = est_cpu[b, spk]
                    r = ref_cpu[b, spk]
                    p_list.append(pesqTest(e.astype(np.float32), r.astype(np.float32), sample_rate))
                    s_list.append(stoiTest(e.astype(np.float32), r.astype(np.float32), sample_rate))
                pesq_vals.append(float(np.nanmean(p_list)))
                stoi_vals.append(float(np.nanmean(s_list)))

            all_metrics["PESQ"].extend(pesq_vals)
            all_metrics["STOI"].extend(stoi_vals)

            # 4.
            for i in range(mix.size(0)):
                if saved >= max_save_samples:
                    break
                base = f"{model_name}_{saved:04d}_perm{perm[i][0]}{perm[i][1]}"
                mix_i = mix[i].detach().cpu()
                s1_ref = targets[i, 0].detach().cpu()
                s2_ref = targets[i, 1].detach().cpu()
                s1_est = est_aligned[i, 0].detach().cpu()
                s2_est = est_aligned[i, 1].detach().cpu()

                torchaudio.save(os.path.join(model_output_dir, f"{base}_mix.wav"), mix_i.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(model_output_dir, f"{base}_s1_ref.wav"), s1_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(model_output_dir, f"{base}_s2_ref.wav"), s2_ref.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(model_output_dir, f"{base}_s1_est.wav"), s1_est.unsqueeze(0), sample_rate)
                torchaudio.save(os.path.join(model_output_dir, f"{base}_s2_est.wav"), s2_est.unsqueeze(0), sample_rate)
                saved += 1

    # Summary results
    def nanmean(x):
        x = np.array(x, dtype=np.float64)
        return float(np.nanmean(x)) if x.size else float("nan")

    results = {k: nanmean(v) for k, v in all_metrics.items()}

    print(f"\n[{model_name}] 最终平均成绩 (PIT-aligned):")
    for k, v in results.items():
        print(f"  {k:8s}: {v:.4f}")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = "../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"
    sample_rate = 8000

    output_dir = "../../Results/SpeechBrain_Baselines_WSJ"
    os.makedirs(output_dir, exist_ok=True)

    # Reduce batch_size to prevent Transformer from exploding video memory
    batch_size = 32
    print(f"Loading WSJ0-2Mix Test Set...")
    _, _, test_loader = MakeWSJLoader(root=root, batch_size=batch_size, num_workers=4, segment=4.0,
                                      sample_rate=sample_rate)

    all_results = {}

    # ===============================================
    # Evaluating SpeechBrains SepFormer
    # ===============================================
    print("\nLoading SepFormer from SpeechBrain HuggingFace Hub...")
    # The pretrained model will be automatically downloaded from HuggingFace to the pretrained_models folder.
    model_sepformer = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir="pretrained_models/sepformer-wsj02mix",
        run_opts={"device": device}  # Make sure the model runs on GPU
    )

    res_sepformer = eval_speechbrain_model("SepFormer", model_sepformer, test_loader, device, sample_rate, output_dir)
    all_results["SepFormer"] = res_sepformer

    # ===============================================
    # Implementation note.
    # ===============================================
    ts = time.strftime("%Y%m%d-%H%M%S")
    txt_path = os.path.join(output_dir, f"speechbrain_sepformer_wsj_{ts}.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SpeechBrain Pre-trained SepFormer Baseline on WSJ0-2Mix\n")
        f.write(f"Sample rate: {sample_rate} Hz\n")
        f.write("-" * 40 + "\n")

        for model_name, metrics in all_results.items():
            f.write(f"Model: {model_name}\n")
            for k, v in metrics.items():
                if isinstance(v, float) and math.isnan(v):
                    f.write(f"  {k:8s}: NaN\n")
                else:
                    f.write(f"  {k:8s}: {v:.4f}\n")
            f.write("-" * 40 + "\n")

    print(f"\n评估结果已保存至: {txt_path}")


if __name__ == "__main__":
    main()