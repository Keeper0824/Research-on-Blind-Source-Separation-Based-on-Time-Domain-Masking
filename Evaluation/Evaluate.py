import PESQ
import SNR
import SI_SNR
import STOI
import SIR_SAR


def evaluateMethod(est, ref, fs=16000):
    """
    Compute a set of metrics for a given method.
    Args:
        est: estimated speech, tensor [B, T]
        ref: reference speech, tensor [B, T]
    Returns:
        dict with SI-SNR, SI-SDR, PESQ, STOI
        (SDR/SIR/SAR need multi-source, handled separately)
    """
    metrics = {}

    metrics["SI-SNR"] = SI_SNR.si_snr_metric(est, ref, keep_batch=False).item()
    metrics["SNR"] = SNR.snr_metric(est, ref, keep_batch=False).item()
    metrics["PESQ"] = PESQ.pesq_batch(est, ref, fs=fs, mode="wb")
    metrics["STOI"] = STOI.stoi_batch(est, ref, fs=fs, extended=False)

    return metrics
