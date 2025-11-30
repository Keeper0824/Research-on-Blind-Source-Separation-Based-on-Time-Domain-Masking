import torch.nn.functional as F
def mae_loss(x, s):
    """
    Mean Absolute Error (MAE) in the time domain.
    Args:
        x: enhanced/estimated speech, shape [B, T]
        s: reference/clean speech, shape [B, T]
    Returns:
        scalar MAE loss
    """
    return F.l1_loss(x, s)