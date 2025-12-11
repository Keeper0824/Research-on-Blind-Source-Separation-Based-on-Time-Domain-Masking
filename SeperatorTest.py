import torch
import torchaudio
import os
from LossFunction.SI_SNR import pit_si_snr_loss
from Model.Model_TasNet import Model
from DataLoader import MakeDataLoader


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
            min_len_needed = 16
            if mix.shape[-1] < min_len_needed:
                pad_len = min_len_needed - mix.shape[-1]
                mix = torch.nn.functional.pad(mix, (0, pad_len), mode="constant", value=0.0)

                # targets 同样 pad（否则对应不上）
                targets = torch.nn.functional.pad(targets, (0, pad_len), mode="constant", value=0.0)

            mix = mix.to(device)  # [B, T]
            targets = targets.to(device)  # [B, 2, T]

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


def run_test(model_path, test_root, batch_size, sample_rate, device):
    model = Model(
        enc_dim=512,
        enc_kernel=16,
        enc_stride=8,
        bottleneck_channels=128,
        hidden_channels=512,
        num_blocks=8,
        num_repeats=3,
        num_speakers=2
    ).to(device)

    # 2. 加载训练好的权重
    print(f"Loading weights: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 3. 建立测试集 DataLoader
    _,_, test_loader = MakeDataLoader(
        root=test_root,
        batch_size=batch_size,
        num_workers=4,
        sample_rate=sample_rate,
        segment=4
    )

    # 4. 输出目录
    output_dir = "./test_results"

    # 5. 调用你给的 eval_test_full()
    eval_test_full(
        model=model,
        test_loader=test_loader,
        loss_fn=pit_si_snr_loss,  # ★ 重要：PIT loss
        sample_rate=sample_rate,
        device=device,
        output_dir=output_dir,
        max_samples=20
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_test(
        model_path="checkpoints/train_convtasnet_20251209-193249/conv_tasnet_final.pt",  # 修改为你实际的 pt 文件
        test_root="Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min",  # 你的数据集路径
        batch_size=1,  # test 可以 batch=1
        sample_rate=8000,  # 按你的数据集来的
        device=device
    )
