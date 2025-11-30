import mir_eval

def bss(ref_sources, est_sources):
    """
    Compute SDR, SIR, SAR using mir_eval.separation.bss_eval_sources.
    Args:
        ref_sources: numpy array, shape [n_sources, T]
        est_sources: numpy array, shape [n_sources, T]
    Returns:
        sdr: numpy array, shape [n_sources]
        sir: numpy array, shape [n_sources]
        sar: numpy array, shape [n_sources]
    """
    # mir_eval expects shape [n_sources, T]
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(ref_sources, est_sources)
    return sdr, sir, sar, perm


def bss_eval(ref_sources, est_sources):
    """
    Wrapper for torch → numpy → mir_eval.
    Args:
        ref_sources: tensor, shape [n_sources, T]
        est_sources: tensor, shape [n_sources, T]
    Returns:
        sdr, sir, sar, perm (all numpy arrays)
    """
    ref_np = ref_sources.detach().cpu().numpy()
    est_np = est_sources.detach().cpu().numpy()
    return bss(ref_np, est_np)
