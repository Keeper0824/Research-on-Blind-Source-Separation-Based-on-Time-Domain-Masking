from torch import nn
from torch_stoi import NegSTOILoss

class STOILoss(nn.Module):
    """
    STOI-based intelligibility loss (differentiable).
    Computes negative STOI, which can be used as a training loss.
    """
    def __init__(self, sample_rate=16000, extended=False):
        """
        Args:
            sample_rate: audio sampling rate
            extended: whether to use ESTOI (extended STOI)
        """
        super().__init__()
        self.loss_fn = NegSTOILoss(sample_rate=sample_rate, extended=extended)

    def forward(self, x, s):
        """
        Args:
            x: enhanced speech [B, T]
            s: clean reference speech [B, T]
        Returns:
            Mean negative STOI loss
        """
        loss_batch = self.loss_fn(x, s)
        return loss_batch.mean()
