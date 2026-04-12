import json
import os
from typing import Dict, Optional, Tuple

import torch


def load_comm_dataset_metadata(dataset_root: str) -> Dict[str, object]:
    """
    Load optional dataset metadata written by communication dataset generators.
    Returns an empty dict if the metadata file does not exist.
    """
    meta_path = os.path.join(dataset_root, "metadata.json")
    if not os.path.exists(meta_path):
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_waveform(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform.detach().float().reshape(-1)
    if waveform.numel() == 0:
        raise ValueError("Waveform is empty.")
    return waveform


def bpsk_symbol_bits(waveform: torch.Tensor, samples_per_symbol: int) -> torch.Tensor:
    """
    Convert a 1-D BPSK baseband waveform into hard-decision bits by
    averaging each symbol window and applying a sign decision.
    """
    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive.")

    waveform = _flatten_waveform(waveform)
    usable = (waveform.numel() // samples_per_symbol) * samples_per_symbol
    if usable == 0:
        raise ValueError(
            f"Waveform length {waveform.numel()} is shorter than one symbol "
            f"(samples_per_symbol={samples_per_symbol})."
        )

    symbols = waveform[:usable].view(-1, samples_per_symbol).mean(dim=1)
    return (symbols >= 0).to(torch.int64)


def effective_ber(raw_ber: float, allow_polarity_flip: bool = True) -> float:
    """
    For BPSK source separation, a global polarity inversion corresponds to
    flipping every decoded bit. The effective BER should therefore be capped at 50%.
    """
    if not allow_polarity_flip:
        return raw_ber
    return min(raw_ber, 1.0 - raw_ber)


def bpsk_symbol_ber(
    est_waveform: torch.Tensor,
    ref_waveform: torch.Tensor,
    samples_per_symbol: int,
    allow_polarity_flip: bool = True,
) -> float:
    est_bits = bpsk_symbol_bits(est_waveform, samples_per_symbol)
    ref_bits = bpsk_symbol_bits(ref_waveform, samples_per_symbol)

    usable = min(est_bits.numel(), ref_bits.numel())
    if usable == 0:
        raise ValueError("No decoded symbols are available for BER calculation.")

    raw_ber = (est_bits[:usable] != ref_bits[:usable]).float().mean().item()
    return effective_ber(raw_ber, allow_polarity_flip=allow_polarity_flip)


def resolve_bpsk_samples_per_symbol(
    dataset_root: Optional[str] = None,
    default: Optional[int] = None,
) -> int:
    """
    Try to resolve the BPSK samples-per-symbol setting from dataset metadata.
    Falls back to `default` when the metadata file is unavailable.
    """
    if dataset_root:
        meta = load_comm_dataset_metadata(dataset_root)
        value = meta.get("samples_per_symbol")
        if isinstance(value, int) and value > 0:
            return value

    if default is not None and default > 0:
        return default

    raise ValueError(
        "Unable to determine samples_per_symbol. Provide dataset metadata "
        "or pass an explicit default value."
    )


def resolve_bpsk_source_samples_per_symbol(
    dataset_root: Optional[str],
    source_index: int,
    default: Optional[int] = None,
) -> int:
    """
    Resolve samples_per_symbol for a specific source in a two-source BPSK
    communication dataset. Falls back to the shared samples_per_symbol field
    for backward compatibility with older datasets.
    """
    if source_index not in (1, 2):
        raise ValueError("source_index must be 1 or 2.")

    if dataset_root:
        meta = load_comm_dataset_metadata(dataset_root)
        keyed_value = meta.get(f"samples_per_symbol_s{source_index}")
        if isinstance(keyed_value, int) and keyed_value > 0:
            return keyed_value

        shared_value = meta.get("samples_per_symbol")
        if isinstance(shared_value, int) and shared_value > 0:
            return shared_value

    if default is not None and default > 0:
        return default

    raise ValueError(
        f"Unable to determine samples_per_symbol for source {source_index}. "
        "Provide dataset metadata or pass an explicit default value."
    )


def resolve_bpsk_samples_per_symbol_pair(
    dataset_root: Optional[str] = None,
    default_s1: Optional[int] = None,
    default_s2: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Resolve the per-source samples_per_symbol pair for a two-source BPSK
    communication dataset.
    """
    sps1 = resolve_bpsk_source_samples_per_symbol(
        dataset_root=dataset_root,
        source_index=1,
        default=default_s1,
    )
    sps2 = resolve_bpsk_source_samples_per_symbol(
        dataset_root=dataset_root,
        source_index=2,
        default=default_s2,
    )
    return sps1, sps2
