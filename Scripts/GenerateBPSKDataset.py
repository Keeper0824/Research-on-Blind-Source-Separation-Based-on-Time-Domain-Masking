import argparse
import json
import math
import os
import shutil
from typing import Dict, Optional, Tuple

import numpy as np


def generate_bits(num_symbols: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=num_symbols, dtype=np.int8)


def bits_to_bpsk_waveform(
    bits: np.ndarray,
    samples_per_symbol: int,
    sample_len: Optional[int] = None,
) -> np.ndarray:
    symbols = 2.0 * bits.astype(np.float32) - 1.0
    waveform = np.repeat(symbols, samples_per_symbol).astype(np.float32)
    if sample_len is not None:
        waveform = waveform[:sample_len]
    return waveform


def generate_bpsk_waveform(
    sample_len: int,
    samples_per_symbol: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    num_symbols = int(math.ceil(sample_len / samples_per_symbol))
    bits = generate_bits(num_symbols, rng)
    waveform = bits_to_bpsk_waveform(bits, samples_per_symbol, sample_len=sample_len)
    return bits, waveform


def scale_to_target_sir(
    src_a: np.ndarray,
    src_b: np.ndarray,
    sir_db: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    ratio = 10.0 ** (sir_db / 20.0)
    src_a = src_a.copy()
    src_b = src_b.copy()

    if rng.random() < 0.5:
        src_a *= ratio
        stronger = "s1"
    else:
        src_b *= ratio
        stronger = "s2"

    return src_a, src_b, {"target_sir_db": float(sir_db), "stronger_source": stronger}


def add_awgn(
    mixture: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    signal_power = float(np.mean(mixture ** 2) + 1e-12)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=mixture.shape).astype(np.float32)
    return (mixture + noise).astype(np.float32), {
        "target_snr_db": float(snr_db),
        "signal_power": signal_power,
        "noise_power": float(noise_power),
    }


def normalize_triplet(
    mix: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    peak = float(np.max(np.abs(mix)) + 1e-8)
    return (
        (mix / peak).astype(np.float32),
        (s1 / peak).astype(np.float32),
        (s2 / peak).astype(np.float32),
    )


def resolve_samples_per_symbol_pair(args: argparse.Namespace) -> Tuple[int, int]:
    if args.sps1 is None and args.sps2 is None and args.sps is None:
        sps1, sps2 = 4, 16
    else:
        sps1 = args.sps1 if args.sps1 is not None else (args.sps if args.sps is not None else 4)
        sps2 = args.sps2 if args.sps2 is not None else (args.sps if args.sps is not None else 16)

    if sps1 <= 0 or sps2 <= 0:
        raise ValueError("samples per symbol must be positive.")

    return int(sps1), int(sps2)


def maybe_recreate_root(root_dir: str, force_recreate: bool) -> None:
    if not os.path.exists(root_dir):
        return

    if not force_recreate:
        raise FileExistsError(
            f"Output directory already exists: {root_dir}. "
            f"Use --force-recreate if you want to overwrite it."
        )

    shutil.rmtree(root_dir)


def make_split(
    split_root: str,
    num_examples: int,
    sample_len: int,
    sps1: int,
    sps2: int,
    sir_low: float,
    sir_high: float,
    snr_low: float,
    snr_high: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    mix_dir = os.path.join(split_root, "mix")
    s1_dir = os.path.join(split_root, "s1")
    s2_dir = os.path.join(split_root, "s2")
    meta_dir = os.path.join(split_root, "meta")
    for path in (mix_dir, s1_dir, s2_dir, meta_dir):
        os.makedirs(path, exist_ok=True)

    for index in range(num_examples):
        bits1, s1 = generate_bpsk_waveform(sample_len=sample_len, samples_per_symbol=sps1, rng=rng)
        bits2, s2 = generate_bpsk_waveform(sample_len=sample_len, samples_per_symbol=sps2, rng=rng)

        sir_db = float(rng.uniform(sir_low, sir_high))
        snr_db = float(rng.uniform(snr_low, snr_high))

        s1_scaled, s2_scaled, sir_meta = scale_to_target_sir(s1, s2, sir_db, rng)
        mix_clean = (s1_scaled + s2_scaled).astype(np.float32)
        mix_noisy, snr_meta = add_awgn(mix_clean, snr_db, rng)
        mix_final, s1_final, s2_final = normalize_triplet(mix_noisy, s1_scaled, s2_scaled)

        file_name = f"{index:05d}.npy"
        np.save(os.path.join(mix_dir, file_name), mix_final)
        np.save(os.path.join(s1_dir, file_name), s1_final)
        np.save(os.path.join(s2_dir, file_name), s2_final)

        sample_meta = {
            "num_symbols_s1": int(bits1.size),
            "num_symbols_s2": int(bits2.size),
            "samples_per_symbol_s1": int(sps1),
            "samples_per_symbol_s2": int(sps2),
            "waveform_length": int(sample_len),
            **sir_meta,
            **snr_meta,
        }
        with open(os.path.join(meta_dir, file_name.replace(".npy", ".json")), "w", encoding="utf-8") as f:
            json.dump(sample_meta, f, ensure_ascii=True, indent=2)


def create_dataset(
    root_dir: str,
    num_train: int,
    num_val: int,
    num_test: int,
    sample_len: int,
    sps1: int,
    sps2: int,
    sir_low: float,
    sir_high: float,
    snr_low: float,
    snr_high: float,
    seed: int,
    force_recreate: bool,
) -> None:
    maybe_recreate_root(root_dir=root_dir, force_recreate=force_recreate)

    split_sizes = {"tr": num_train, "cv": num_val, "tt": num_test}
    for split_index, (split_name, split_size) in enumerate(split_sizes.items()):
        print(f"Generating {split_name} split: {split_size} examples")
        make_split(
            split_root=os.path.join(root_dir, split_name),
            num_examples=split_size,
            sample_len=sample_len,
            sps1=sps1,
            sps2=sps2,
            sir_low=sir_low,
            sir_high=sir_high,
            snr_low=snr_low,
            snr_high=snr_high,
            seed=seed + split_index,
        )

    metadata = {
        "dataset_type": "bpsk_baseband_two_source",
        "modulation": "BPSK",
        "num_sources": 2,
        "num_train": int(num_train),
        "num_val": int(num_val),
        "num_test": int(num_test),
        "waveform_length": int(sample_len),
        "sample_len": int(sample_len),
        "samples_per_symbol_s1": int(sps1),
        "samples_per_symbol_s2": int(sps2),
        "samples_per_symbol": int(sps1) if sps1 == sps2 else None,
        "sir_db_range": [float(sir_low), float(sir_high)],
        "snr_db_range": [float(snr_low), float(snr_high)],
        "seed": int(seed),
        "experiment_note": "Two-source BPSK-only communication mixture with optional different symbol rates.",
        "ber_note": "Use symbol-level hard decisions with PIT alignment and polarity correction.",
    }
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    print(f"Dataset ready: {root_dir}")
    print(json.dumps(metadata, ensure_ascii=True, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small two-source BPSK-only communication dataset."
    )
    parser.add_argument("--root", default="./Dataset/CommBPSK_4v16", help="Output dataset root directory.")
    parser.add_argument("--num-train", type=int, default=20000, help="Number of training examples.")
    parser.add_argument("--num-val", type=int, default=5000, help="Number of validation examples.")
    parser.add_argument("--num-test", type=int, default=5000, help="Number of test examples.")
    parser.add_argument("--sample-len", type=int, default=32768, help="Waveform length in samples.")
    parser.add_argument("--sps", type=int, default=None, help="Shared samples per symbol for both sources.")
    parser.add_argument("--sps1", type=int, default=None, help="Samples per symbol for source 1.")
    parser.add_argument("--sps2", type=int, default=None, help="Samples per symbol for source 2.")
    parser.add_argument("--sir-low", type=float, default=0.0, help="Lower bound of source SIR in dB.")
    parser.add_argument("--sir-high", type=float, default=1.5, help="Upper bound of source SIR in dB.")
    parser.add_argument("--snr-low", type=float, default=12.0, help="Lower bound of mixture SNR in dB.")
    parser.add_argument("--snr-high", type=float, default=18.0, help="Upper bound of mixture SNR in dB.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sps1, sps2 = resolve_samples_per_symbol_pair(args)
    create_dataset(
        root_dir=args.root,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        sample_len=args.sample_len,
        sps1=sps1,
        sps2=sps2,
        sir_low=args.sir_low,
        sir_high=args.sir_high,
        snr_low=args.snr_low,
        snr_high=args.snr_high,
        seed=args.seed,
        force_recreate=args.force_recreate,
    )
