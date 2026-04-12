import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader


def find_repo_root(start_path: Path) -> Path:
    candidates = [start_path] + list(start_path.parents)
    for candidate in candidates:
        if (
            (candidate / "DataLoader" / "CommDataLoader.py").exists()
            and (candidate / "Evaluation" / "CommBER.py").exists()
        ):
            return candidate
    raise FileNotFoundError("Unable to locate repository root for BPSK small evaluation.")


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPO_ROOT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_comm_dataloader_module = load_module_from_path(
    "bpsk_comm_dataloader_eval",
    REPO_ROOT / "DataLoader" / "CommDataLoader.py",
)
CommMixDataset = _comm_dataloader_module.CommMixDataset

from Evaluation.CommBER import bpsk_symbol_ber, resolve_bpsk_samples_per_symbol_pair
from Evaluation.SI_SNR import pit_align_by_sisnr, si_snr_improvement
from Evaluation.SNR import snr
from Model.train_bpsk_small_common import build_model, resolve_device


SOFT_THRESHOLD = 0.02
HARD_THRESHOLD = 0.0038


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def build_test_loader(
    dataset_root: str,
    batch_size: int,
    num_workers: int,
    segment_len: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = CommMixDataset(root=dataset_root, split="tt", segment=segment_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )


def load_model_checkpoint(model_name: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(model_name).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_single_model(
    model_name: str,
    checkpoint_path: str,
    dataset_root: str,
    batch_size: int,
    num_workers: int,
    segment_len: int,
    device: torch.device,
    sps1: int,
    sps2: int,
) -> Dict[str, float]:
    pin_memory = device.type == "cuda"
    test_loader = build_test_loader(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
        segment_len=segment_len,
        pin_memory=pin_memory,
    )
    model = load_model_checkpoint(model_name=model_name, checkpoint_path=checkpoint_path, device=device)

    sisnr_values = []
    sisnri_values = []
    snr_values = []
    source_bers = []
    sample_bers = []
    sample_count = 0

    with torch.no_grad():
        for mix, targets in test_loader:
            mix = mix.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            est_sources = model(mix.unsqueeze(1))
            min_len = min(est_sources.shape[-1], targets.shape[-1], mix.shape[-1])
            est_sources = est_sources[..., :min_len]
            targets = targets[..., :min_len]
            mix = mix[..., :min_len]

            est_aligned, _, sisnr_pair = pit_align_by_sisnr(est_sources, targets)
            sisnri_pair = si_snr_improvement(est_sources, targets, mix)
            snr_pair = 0.5 * (
                snr(est_aligned[:, 0, :], targets[:, 0, :]) +
                snr(est_aligned[:, 1, :], targets[:, 1, :])
            )

            sisnr_values.extend(sisnr_pair.detach().cpu().tolist())
            sisnri_values.extend(sisnri_pair.detach().cpu().tolist())
            snr_values.extend(snr_pair.detach().cpu().tolist())

            batch_size_now = est_aligned.size(0)
            sample_count += batch_size_now
            for batch_index in range(batch_size_now):
                ber_s1 = bpsk_symbol_ber(
                    est_waveform=est_aligned[batch_index, 0].detach().cpu(),
                    ref_waveform=targets[batch_index, 0].detach().cpu(),
                    samples_per_symbol=sps1,
                )
                ber_s2 = bpsk_symbol_ber(
                    est_waveform=est_aligned[batch_index, 1].detach().cpu(),
                    ref_waveform=targets[batch_index, 1].detach().cpu(),
                    samples_per_symbol=sps2,
                )
                source_bers.extend([ber_s1, ber_s2])
                sample_bers.append(0.5 * (ber_s1 + ber_s2))

    if sample_count == 0:
        raise RuntimeError("Test loader is empty. Increase the number of test samples or reduce batch size.")

    source_bers_tensor = torch.tensor(source_bers, dtype=torch.float64)
    sample_bers_tensor = torch.tensor(sample_bers, dtype=torch.float64)

    return {
        "SI-SNR": float(torch.tensor(sisnr_values, dtype=torch.float64).mean().item()),
        "SI-SNRi": float(torch.tensor(sisnri_values, dtype=torch.float64).mean().item()),
        "SNR": float(torch.tensor(snr_values, dtype=torch.float64).mean().item()),
        "BER": float(source_bers_tensor.mean().item()),
        "BER_median": float(source_bers_tensor.median().item()),
        "WorstSourceBER": float(source_bers_tensor.max().item()),
        "SampleBER": float(sample_bers_tensor.mean().item()),
        "soft_pass_rate": float((source_bers_tensor < SOFT_THRESHOLD).to(torch.float32).mean().item()),
        "hard_pass_rate": float((source_bers_tensor < HARD_THRESHOLD).to(torch.float32).mean().item()),
        "num_test_examples": int(sample_count),
        "samples_per_symbol_s1": int(sps1),
        "samples_per_symbol_s2": int(sps2),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
    }


def write_comparison(
    output_dir: Path,
    baseline_metrics: Dict[str, float],
    multiscale_metrics: Dict[str, float],
) -> None:
    baseline_ber = baseline_metrics["BER"]
    multiscale_ber = multiscale_metrics["BER"]
    target_reached = multiscale_ber < SOFT_THRESHOLD and baseline_ber > SOFT_THRESHOLD

    lines = [
        "BPSK Small Experiment Comparison",
        "----------------------------------------",
        f"Baseline   BER: {baseline_ber * 100:.4f}%",
        f"MultiScale BER: {multiscale_ber * 100:.4f}%",
        f"Baseline below 2%: {baseline_ber < SOFT_THRESHOLD}",
        f"MultiScale below 2%: {multiscale_ber < SOFT_THRESHOLD}",
        f"Baseline below 0.38%: {baseline_ber < HARD_THRESHOLD}",
        f"MultiScale below 0.38%: {multiscale_ber < HARD_THRESHOLD}",
        f"Result: {'target reached' if target_reached else 'target not reached yet; adjust SNR/SIR or source-rate gap.'}",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.txt").write_text("\n".join(lines), encoding="utf-8")

    save_json(
        output_dir / "summary.json",
        {
            "baseline": baseline_metrics,
            "multiscale": multiscale_metrics,
            "thresholds": {
                "soft_decision_ber": SOFT_THRESHOLD,
                "hard_decision_ber": HARD_THRESHOLD,
            },
            "target_reached": target_reached,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Baseline vs MultiScale on the BPSK-only small communication experiment."
    )
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT / "Dataset" / "CommBPSK_4v16"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--baseline-ckpt",
        default=str(REPO_ROOT / "checkpoints" / "BPSKSmall" / "baseline" / "best.pt"),
        help="Checkpoint path for the Baseline model.",
    )
    parser.add_argument(
        "--multiscale-ckpt",
        default=str(REPO_ROOT / "checkpoints" / "BPSKSmall" / "multiscale" / "best.pt"),
        help="Checkpoint path for the MultiScale model.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "Results" / "BPSKSmall_4v16"),
        help="Directory used to save evaluation results.",
    )
    parser.add_argument("--segment-len", type=int, default=32768, help="Waveform segment length used for evaluation.")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Evaluation device.")
    parser.add_argument("--sps1", type=int, default=None, help="Optional override for source 1 samples per symbol.")
    parser.add_argument("--sps2", type=int, default=None, help="Optional override for source 2 samples per symbol.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    sps1, sps2 = resolve_bpsk_samples_per_symbol_pair(
        dataset_root=args.root,
        default_s1=args.sps1,
        default_s2=args.sps2,
    )

    output_dir = Path(args.output_dir)
    baseline_metrics = evaluate_single_model(
        model_name="baseline",
        checkpoint_path=args.baseline_ckpt,
        dataset_root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        segment_len=args.segment_len,
        device=device,
        sps1=sps1,
        sps2=sps2,
    )
    multiscale_metrics = evaluate_single_model(
        model_name="multiscale",
        checkpoint_path=args.multiscale_ckpt,
        dataset_root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        segment_len=args.segment_len,
        device=device,
        sps1=sps1,
        sps2=sps2,
    )

    save_json(output_dir / "baseline_eval_metrics.json", baseline_metrics)
    save_json(output_dir / "multiscale_eval_metrics.json", multiscale_metrics)
    write_comparison(output_dir, baseline_metrics, multiscale_metrics)

    print(f"Baseline BER: {baseline_metrics['BER'] * 100:.4f}%")
    print(f"MultiScale BER: {multiscale_metrics['BER'] * 100:.4f}%")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
