import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CommMixDataset(Dataset):
    def __init__(self, root, split="tr", segment=None):
        """
        root: 数据集根目录 (e.g., './CommDataset')
        split: 'tr', 'cv', or 'tt'
        segment: 裁剪长度 (秒)。注意：这里我们假设 sample_rate=1 来处理索引，或者直接传点数。
                 建议直接传 int 类型的点数 (e.g., 32000)。
        """
        super().__init__()
        self.segment = segment

        # 搜索 .npy 文件
        self.mix_list = sorted(glob.glob(os.path.join(root, split, "mix", "*.npy")))
        self.s1_list = sorted(glob.glob(os.path.join(root, split, "s1", "*.npy")))
        self.s2_list = sorted(glob.glob(os.path.join(root, split, "s2", "*.npy")))

        assert len(self.mix_list) > 0, f"No .npy files found in {os.path.join(root, split)}"
        assert len(self.mix_list) == len(self.s1_list) == len(self.s2_list), "Number of files mismatch!"

    def __len__(self):
        return len(self.mix_list)

    def __getitem__(self, idx):
        # 读取 numpy 数据
        mix = np.load(self.mix_list[idx])
        s1 = np.load(self.s1_list[idx])
        s2 = np.load(self.s2_list[idx])

        # 转为 Tensor
        mix = torch.from_numpy(mix).float()
        s1 = torch.from_numpy(s1).float()
        s2 = torch.from_numpy(s2).float()

        # 长度裁剪 (如果需要)
        # 我们的生成脚本已经保证了长度一致，但为了鲁棒性保留此逻辑
        if self.segment is not None:
            # 如果 segment 是点数 (比如 32000)
            target_len = int(self.segment)
            cur_len = mix.shape[0]

            if cur_len > target_len:
                # 随机裁剪
                start = np.random.randint(0, cur_len - target_len)
                mix = mix[start: start + target_len]
                s1 = s1[start: start + target_len]
                s2 = s2[start: start + target_len]
            elif cur_len < target_len:
                # 补零
                pad = target_len - cur_len
                mix = torch.nn.functional.pad(mix, (0, pad))
                s1 = torch.nn.functional.pad(s1, (0, pad))
                s2 = torch.nn.functional.pad(s2, (0, pad))

        # 模型通常需要 [Batch, Length] 或 [Batch, 1, Length]，这里返回单样本
        # 你的 WSJDataLoader 返回的是 s1, s2 stack 后的结果
        return mix, torch.stack([s1, s2], dim=0)


def MakeDataLoader(root, batch_size, num_workers, segment):
    """
    root: 数据集路径
    segment: 每个样本的长度 (点数)，例如 32000
    """
    train_set = CommMixDataset(root=root, split="tr", segment=segment)
    trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    valid_set = CommMixDataset(root=root, split="cv", segment=segment)
    validLoader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    test_set = CommMixDataset(root=root, split="tt", segment=segment)
    testLoader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return trainLoader, validLoader, testLoader