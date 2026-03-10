import numpy as np
from pystoi import stoi


def stoiTest(est: np.ndarray, ref: np.ndarray, fs: int) -> float:
    try:
        return float(stoi(ref, est, fs, extended=False))
    except Exception:
        return float("nan")
