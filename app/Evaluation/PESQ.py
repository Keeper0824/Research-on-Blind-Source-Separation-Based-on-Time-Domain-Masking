import numpy as np
from pesq import pesq


def pesqTest(est, ref, fs):
    """
    est/ref: [T] numpy float32
    """
    try:
        mode = "wb" if fs >= 16000 else "nb"
        return float(pesq(fs, ref, est, mode))
    except Exception:
        return float("nan")
