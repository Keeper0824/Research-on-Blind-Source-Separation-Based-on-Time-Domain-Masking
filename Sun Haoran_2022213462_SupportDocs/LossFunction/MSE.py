import torch.nn.functional as F

def mse_loss(x, s):
    """
    Mean Squared Error (MSE) in the time domain.
    Args:
        x: enhanced/estimated speech, shape [B, T]
        s: reference/clean speech, shape [B, T]
    Returns:
        scalar MSE loss
    """
    return F.mse_loss(x, s)

