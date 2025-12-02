import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import LibriMix
from torchaudio.models import ConvTasNet
import LossFunction.SI_SNR as SI_SNR


class LibriMixSeparationDataset(Dataset):
    def __init__(self, root, subset="train-100",
                 num_speakers=2, sample_rate=8000,
                 task="sep_clean", mode="min"):

        self.ds = LibriMix(
            root=root,
            subset=subset,
            num_speakers=num_speakers,
            sample_rate=sample_rate,
            task=task,
            mode=mode,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # sr: int
        # mixture: Tensor [C, T]
        # sources: list[Tensor[T]]，长度=num_speakers
        sr, mixture, sources = self.ds[idx]

        # 保守一点，转成 mono 单声道 [1, T]
        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)  # [1, T]
        else:
            mixture = mixture.mean(0, keepdim=True)  # 多声道 -> [1, T]

        # sources: list[T] -> Tensor[num_speakers, T]
        sources = torch.stack(sources, dim=0)

        return mixture, sources, sr


def make_dataloaders(root, batch_size=8, num_workers=0):
    train_set = LibriMixSeparationDataset(root, subset="train-100")
    val_set = LibriMixSeparationDataset(root, subset="dev")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # 先保持 0，Windows 更稳
        collate_fn=separation_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=separation_collate_fn,
    )

    return train_loader, val_loader


def build_model(num_sources=2):
    model = ConvTasNet(
        num_sources=num_sources,
        # 下面这些是 paper 里类似的默认值，可以先不动
        enc_kernel_size=16,
        enc_num_feats=512,
        msk_kernel_size=3,
        msk_num_feats=128,
        msk_num_hidden_feats=512,
        msk_num_layers=8,
        msk_num_stacks=3,
    )
    return model

def separation_collate_fn(batch):
    """
    batch: list of (mixture, sources, sr)
      mixture: [1, T_i]
      sources: [num_speakers, T_i]
    返回:
      mixtures_pad: [B, 1, T_max]
      sources_pad:  [B, num_speakers, T_max]
      sr: int (假设 batch 里都是同一个采样率)
    """
    mixtures, sources_list, srs = zip(*batch)  # 各是 tuple 长度 B

    # 检查一下 num_speakers 一致
    num_speakers = sources_list[0].size(0)

    # 计算这个 batch 中的最长长度
    lengths = [m.shape[-1] for m in mixtures]
    max_len = max(lengths)

    # pad mixture
    padded_mixtures = []
    padded_sources = []

    for m, s in zip(mixtures, sources_list):
        # m: [1, T_i]；s: [C, T_i]
        T = m.shape[-1]
        if T < max_len:
            pad_size = max_len - T
            # 在右侧补零: (left, right) for last dim
            m = torch.nn.functional.pad(m, (0, pad_size))
            s = torch.nn.functional.pad(s, (0, pad_size))
        padded_mixtures.append(m)
        padded_sources.append(s)

    mixtures_pad = torch.stack(padded_mixtures, dim=0)  # [B, 1, T_max]
    sources_pad = torch.stack(padded_sources, dim=0)  # [B, C, T_max]

    # 默认取第一个 sr
    sr = srs[0]
    return mixtures_pad, sources_pad, sr


def train_convtasnet(
        librimix_root,
        num_epochs=5,
        batch_size=1,
        lr=1e-3,
        device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # 1. 数据
    train_loader, val_loader = make_dataloaders(
        root=librimix_root,
        batch_size=batch_size,
        num_workers=4,
    )

    # 2. 模型
    model = build_model(num_sources=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for i, (mixture, sources, sr) in enumerate(train_loader):
            # mixture: [B, 1, T]；sources: [B, 2, T]
            mixture = mixture.to(device)  # [B, 1, T]
            sources = sources.to(device)  # [B, 2, T]

            optimizer.zero_grad()
            est_sources = model(mixture)  # [B, 2, T]

            loss = SI_SNR.negative_si_snr(est_sources, sources)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch} | Step {i + 1}/{len(train_loader)} "
                      f"| Train loss (−SI-SNR): {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} finished. Avg train loss: {avg_train_loss:.4f}")

        # 简单做个 val（可选）
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for mixture, sources, sr in val_loader:
                mixture = mixture.to(device)
                sources = sources.to(device)
                est_sources = model(mixture)
                loss = SI_SNR.negative_si_snr(est_sources, sources)
                val_loss += loss.item()
            val_loss /= len(val_loader)
        print(f"Epoch {epoch} | Val loss (−SI-SNR): {val_loss:.4f}")

    return model


if __name__ == "__main__":
    librimix_root = "Dataset/LibriMix"  # 这里换成你自己的路径
    model = train_convtasnet(
        librimix_root=librimix_root,
        num_epochs=3,  # 先跑 1～3 个 epoch 看看能不能收敛
        batch_size=1,  # 后面再考虑 padding + 大 batch
        lr=1e-3,
    )
