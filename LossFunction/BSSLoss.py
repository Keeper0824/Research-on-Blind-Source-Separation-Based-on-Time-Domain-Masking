import torch
from torch import nn
from SI_SNR import negative_si_snr
from MAE import mae_loss
from STOI import STOILoss


class BSSLoss(nn.Module):
    """
    Composite loss for time-domain blind source separation.

    Three modes are supported:

    1) "si_snr"
       - Pure negative SI-SNR loss.
       - This is the standard setup used in many time-domain separation papers.

    2) "si_snr_mae"
       - Negative SI-SNR + lambda_mae * MAE.
       - MAE acts as an auxiliary loss to stabilize training.

       loss = negative_si_snr(x, s) + lambda_mae * mae_loss(x, s)

    3) "si_snr_mae_stoi"
       - Negative SI-SNR + lambda_mae * MAE + lambda_stoi * Neg-STOI.
       - STOI encourages better speech intelligibility.

       loss = negative_si_snr(x, s)
              + lambda_mae * mae_loss(x, s)
              + lambda_stoi * stoi_loss(x, s)

    Args:
        mode: one of {"si_snr", "si_snr_mae", "si_snr_mae_stoi"}.
        lambda_mae: weight for MAE auxiliary loss.
        lambda_stoi: weight for STOI auxiliary loss.
        sample_rate: sampling rate used for STOI.
        extended_stoi: whether to use ESTOI (extended STOI).
    """

    def __init__(
        self,
        mode: str = "si_snr",
        lambda_mae: float = 0.05,
        lambda_stoi: float = 0.1,
        sample_rate: int = 16000,
        extended_stoi: bool = False,
    ):
        super().__init__()

        valid_modes = {"si_snr", "si_snr_mae", "si_snr_mae_stoi"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. "
                             f"Expected one of {valid_modes}.")

        self.mode = mode
        self.lambda_mae = lambda_mae
        self.lambda_stoi = lambda_stoi

        # STOI loss is only needed in the third mode
        self.stoi_loss = None
        if mode == "si_snr_mae_stoi":
            self.stoi_loss = STOILoss(
                sample_rate=sample_rate,
                extended=extended_stoi,
            )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Compute the selected composite loss.

        Args:
            x: estimated/enhanced speech, shape [B, T]
            s: reference/clean speech, shape [B, T]

        Returns:
            loss: scalar tensor (to be minimized).
        """
        # 1) main SI-SNR loss (already negative in your implementation)
        loss = negative_si_snr(x, s)

        # 2) add MAE if required
        if self.mode in {"si_snr_mae", "si_snr_mae_stoi"}:
            loss_mae = mae_loss(x, s)
            loss = loss + self.lambda_mae * loss_mae

        # 3) add STOI if required
        if self.mode == "si_snr_mae_stoi":
            loss_stoi = self.stoi_loss(x, s)  # already negative
            loss = loss + self.lambda_stoi * loss_stoi

        return loss
