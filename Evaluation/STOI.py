import numpy as np
from pystoi import stoi

def stoi_batch(pred, target, fs=16000, extended=False):
    """
    Compute mean STOI over a batch using pystoi.
    Args:
        pred: estimated speech, tensor [B, T]
        target: reference speech, tensor [B, T]
        fs: sampling rate
        extended: if True, use ESTOI; otherwise standard STOI
    Returns:
        mean_stoi: scalar float in [0, 1]
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    scores = []
    for b in range(pred_np.shape[0]):
        x = pred_np[b]
        s = target_np[b]

        L = min(len(x), len(s))
        x = x[:L]
        s = s[:L]

        score = stoi(s, x, fs, extended=extended)
        scores.append(score)

    return float(np.mean(scores))
