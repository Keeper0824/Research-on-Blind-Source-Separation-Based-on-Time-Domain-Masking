import argparse
import copy
import importlib.util
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader


def find_repo_root(start_path: Path) -> Path:
    candidates = [start_path] + list(start_path.parents)
    for candidate in candidates:
        if (
            (candidate / "DataLoader" / "CommDataLoader.py").exists()
            and (candidate / "LossFunction" / "SI_SNR.py").exists()
        ):
            return candidate
    raise FileNotFoundError("Unable to locate repository root containing DataLoader/CommDataLoader.py.")


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
    "bpsk_comm_dataloader",
    REPO_ROOT / "DataLoader" / "CommDataLoader.py",
)
CommMixDataset = _comm_dataloader_module.CommMixDataset

from LossFunction.SI_SNR import pit_si_snr_loss
from Model.Baseline.Model_TasNet import Model as BaselineTasNet
from Model.MultiScale.Model_TasNet_MultiScale import Model as MultiScaleTasNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but --device cuda was requested.")
        return torch.device("cuda")

    if device_name == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {device_name}")


def build_baseline_model() -> torch.nn.Module:
    return BaselineTasNet(
        enc_dim=512,
        enc_kernel=16,
        enc_stride=8,
        bottleneck_channels=128,
        hidden_channels=256,
        num_blocks=8,
        num_repeats=3,
        num_speakers=2,
    )


def build_multiscale_model() -> torch.nn.Module:
    return MultiScaleTasNet(
        enc_dim=516,
        enc_kernel=[8, 16, 24],
        enc_stride=8,
        bottleneck_channels=128,
        hidden_channels=256,
        num_blocks=8,
        num_repeats=3,
        num_speakers=2,
    )


def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "baseline":
        return build_baseline_model()
    if model_name == "multiscale":
        return build_multiscale_model()
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_dataloaders(
    dataset_root: str,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    segment_len: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = CommMixDataset(root=dataset_root, split="tr", segment=segment_len)
    valid_dataset = CommMixDataset(root=dataset_root, split="cv", segment=segment_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=len(train_dataset) >= batch_size,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def evaluate_valid(
    model: torch.nn.Module,
    valid_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for mix, targets in valid_loader:
            mix = mix.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            autocast_context = (
                autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                est_sources = model(mix.unsqueeze(1))
                min_len = min(est_sources.shape[-1], targets.shape[-1])
                est_sources = est_sources[..., :min_len]
                targets = targets[..., :min_len]
                val_loss = pit_si_snr_loss(est_sources, targets)

            total_val_loss += float(val_loss.item())
            num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Validation loader is empty. Reduce eval_batch_size or add more validation samples.")

    return total_val_loss / num_batches


def add_common_train_arguments(
    parser: argparse.ArgumentParser,
    default_dataset_root: str,
    default_save_dir: str,
) -> argparse.ArgumentParser:
    parser.add_argument("--root", default=default_dataset_root, help="Dataset root directory.")
    parser.add_argument("--save-dir", default=default_save_dir, help="Directory used to save checkpoints and logs.")
    parser.add_argument("--segment-len", type=int, default=32768, help="Waveform segment length used for training.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Validation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate.")
    parser.add_argument("--max-lr", type=float, default=1e-3, help="OneCycleLR max learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stop patience after warmup.")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Minimum epochs before early stop.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Training device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def train_bpsk_model(
    model: torch.nn.Module,
    model_name: str,
    args: argparse.Namespace,
) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda"
    pin_memory = use_amp

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader = build_dataloaders(
        dataset_root=args.root,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        segment_len=args.segment_len,
        pin_memory=pin_memory,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty. Reduce batch_size or increase the number of training samples.")

    model = model.to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    scaler = GradScaler(enabled=use_amp)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    best_state_dict = None

    train_config = {
        "model_name": model_name,
        "dataset_root": args.root,
        "segment_len": args.segment_len,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_lr": args.max_lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "warmup_epochs": args.warmup_epochs,
        "device": args.device,
        "seed": args.seed,
    }
    save_json(save_dir / "train_config.json", train_config)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0

        for mix, targets in train_loader:
            mix = mix.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                est_sources = model(mix.unsqueeze(1))
                min_len = min(est_sources.shape[-1], targets.shape[-1])
                est_sources = est_sources[..., :min_len]
                targets = targets[..., :min_len]
                loss = pit_si_snr_loss(est_sources, targets)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            scheduler.step()
            ema_model.update_parameters(model)

            total_train_loss += float(loss.item())
            total_grad_norm += float(grad_norm)
            num_batches += 1

        avg_train_loss = total_train_loss / max(num_batches, 1)
        avg_grad_norm = total_grad_norm / max(num_batches, 1)
        avg_val_loss = evaluate_valid(
            model=ema_model,
            valid_loader=valid_loader,
            device=device,
            use_amp=use_amp,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "avg_grad_norm": avg_grad_norm,
            }
        )
        save_json(save_dir / "train_history.json", {"history": history})

        print(
            f"[{model_name}] epoch {epoch:03d} | "
            f"train={avg_train_loss:.4f} | val={avg_val_loss:.4f} | grad={avg_grad_norm:.3f}"
        )

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            best_state_dict = copy.deepcopy(
                ema_model.module.state_dict() if hasattr(ema_model, "module") else ema_model.state_dict()
            )
            torch.save(
                {
                    "model_name": model_name,
                    "model_state": best_state_dict,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "train_config": train_config,
                },
                save_dir / "best.pt",
            )
        elif epoch >= args.warmup_epochs:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                print(f"[{model_name}] early stopping at epoch {epoch}.")
                break

    if best_state_dict is None:
        raise RuntimeError("Training ended without saving a best checkpoint.")

    save_json(
        save_dir / "train_summary.json",
        {
            "model_name": model_name,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "num_recorded_epochs": len(history),
            "checkpoint_path": str((save_dir / "best.pt").resolve()),
        },
    )
    print(f"[{model_name}] best checkpoint saved to {save_dir / 'best.pt'}")
