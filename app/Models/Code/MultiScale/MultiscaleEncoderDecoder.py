import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def _crop_to_min_len(xs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Crop tensors in xs to the minimum time length (last dimension)."""
    min_len = min(x.size(-1) for x in xs)
    if all(x.size(-1) == min_len for x in xs):
        return xs
    return [x[..., :min_len].contiguous() for x in xs]


class Encoder(nn.Module):
    """
    Multi-scale Conv-TasNet style encoder (time-domain filterbank).

    Input:
        wav: [B, 1, T]
    Output:
        feat: [B, N, K]   (K depends on stride and cropping alignment)

    Design:
        branches: Conv1d(1 -> N_i, kernel=k_i, stride=stride)
        fuse: concat along channel -> 1x1 Conv -> N
    """

    def __init__(
            self,
            out_channels: int = 256,
            kernel_sizes: List[int] = (16, 32, 64),
            stride: int = 8,
            branch_channels: Optional[List[int]] = None,
            activation: str = "prelu",
            bias: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_sizes = list(kernel_sizes)
        self.stride = stride

        if branch_channels is None:
            # Even split by default (last branch gets the remainder).
            base = out_channels // len(self.kernel_sizes)
            branch_channels = [base] * len(self.kernel_sizes)
            branch_channels[-1] += out_channels - sum(branch_channels)
        assert len(branch_channels) == len(self.kernel_sizes)
        assert sum(branch_channels) == out_channels, \
            "Sum(branch_channels) must equal out_channels for clean parameter control."

        self.branches = nn.ModuleList()
        for k, c in zip(self.kernel_sizes, branch_channels):
            self.branches.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=c,
                    kernel_size=k,
                    stride=stride,
                    padding=0,  # keep as 'valid', we will crop after
                    bias=bias,
                )
            )

        self.fuse = nn.Conv1d(
            in_channels=sum(branch_channels),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        if activation == "prelu":
            self.act = nn.PReLU()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [B, 1, T] or [B, T]
        """
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)

        outs = [branch(wav) for branch in self.branches]  # each: [B, C_i, K_i]
        outs = _crop_to_min_len(outs)  # align time length
        x = torch.cat(outs, dim=1)  # [B, sum(C_i), K]
        x = self.fuse(x)  # [B, N, K]
        x = self.act(x)
        return x


class SharedDecoder(nn.Module):
    """
    Shared decoder (single ConvTranspose1d).

    Input:
        feat: [B, N, K]
    Output:
        wav_hat: [B, 1, T_hat]
    """

    def __init__(
            self,
            in_channels: int = 256,
            kernel_size: int = 16,
            stride: int = 8,
            bias: bool = False,
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat:
          [B, N, K]      -> [B, 1, T]
          [B, S, N, K]   -> [B, S, T]
        """
        if feat.dim() == 3:
            return self.deconv(feat)

        if feat.dim() == 4:
            B, S, N, K = feat.shape
            x = feat.reshape(B * S, N, K)  # [B*S, N, K]
            y = self.deconv(x)  # [B*S, 1, T]
            y = y.squeeze(1).reshape(B, S, -1)  # [B, S, T]
            return y.contiguous()

        raise ValueError(f"SharedDecoder got unsupported shape: {tuple(feat.shape)}")


class MultiBranchDecoder(nn.Module):
    """
    Optional: multi-branch decoder (ConvTranspose1d with multiple kernel sizes) and sum.

    This is closer to a symmetric multi-scale synthesis filterbank, but can be slightly
    more sensitive in training. Use SharedDecoder first for stability.
    """

    def __init__(
            self,
            in_channels: int = 256,
            kernel_sizes: List[int] = (16, 32, 64),
            stride: int = 8,
            bias: bool = False,
    ):
        super().__init__()
        self.deconvs = nn.ModuleList([
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=k,
                stride=stride,
                padding=0,
                bias=bias,
            )
            for k in kernel_sizes
        ])

    def _decode_3d(self, feat_3d: torch.Tensor) -> torch.Tensor:
        # feat_3d: [B, C, T]
        outs = [d(feat_3d) for d in self.deconvs]  # each: [B, 1, T_i]
        outs = _crop_to_min_len(outs)
        wav_hat = torch.stack(outs, dim=0).sum(dim=0)  # [B, 1, T]
        return wav_hat

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # [B, S, C, T] -> merge -> [B*S, C, T]
        B, S, C, T = feat.shape
        feat = feat.reshape(B * S, C, T)
        wav = self._decode_3d(feat)  # [B*S, 1, T_w]
        wav = wav.reshape(B, S, wav.size(-1))  # [B, S, T_w]
        return wav
