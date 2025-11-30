import numpy as np
from pesq import pesq

def pesq_batch(pred, target, fs=16000, mode="wb"):
    """
    Compute mean PESQ over a batch.
    Args:
        pred: estimated speech, tensor [B, T]
        target: reference speech, tensor [B, T]
        fs: sampling rate (8000 or 16000)
        mode: "nb" (narrow-band) or "wb" (wide-band)
    Returns:
        mean_pesq: scalar float
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    scores = []
    for b in range(pred_np.shape[0]):
        x = pred_np[b]
        s = target_np[b]

        # Align length (truncate to the shorter one)
        L = min(len(x), len(s))
        x = x[:L]
        s = s[:L]

        # pesq(fs, reference, degraded)
        score = pesq(fs, s, x, mode)
        scores.append(score)

    return float(np.mean(scores))
