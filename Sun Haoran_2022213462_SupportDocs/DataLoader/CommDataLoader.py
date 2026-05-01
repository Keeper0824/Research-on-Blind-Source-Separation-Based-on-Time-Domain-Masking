import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CommMixDataset(Dataset):
    def __init__(self, root, split="tr", segment=None):
        super().__init__()
        self.segment = segment
        self.mix_list = sorted(glob.glob(os.path.join(root, split, "mix", "*.npy")))
        self.s1_list = sorted(glob.glob(os.path.join(root, split, "s1", "*.npy")))
        self.s2_list = sorted(glob.glob(os.path.join(root, split, "s2", "*.npy")))

    def __len__(self):
        return len(self.mix_list)

    def __getitem__(self, idx):
        mix = np.load(self.mix_list[idx])
        s1 = np.load(self.s1_list[idx])
        s2 = np.load(self.s2_list[idx])
        mix = torch.from_numpy(mix).float()
        s1 = torch.from_numpy(s1).float()
        s2 = torch.from_numpy(s2).float()

        if self.segment is not None:
            target_len = int(self.segment)
            cur_len = mix.shape[0]

            if cur_len > target_len:
                start = np.random.randint(0, cur_len - target_len)
                mix = mix[start: start + target_len]
                s1 = s1[start: start + target_len]
                s2 = s2[start: start + target_len]
            elif cur_len < target_len:
                pad = target_len - cur_len
                mix = torch.nn.functional.pad(mix, (0, pad))
                s1 = torch.nn.functional.pad(s1, (0, pad))
                s2 = torch.nn.functional.pad(s2, (0, pad))

        return mix, torch.stack([s1, s2], dim=0)


def MakeDataLoader(root, batch_size, num_workers, segment):
    train_set = CommMixDataset(root=root, split="tr", segment=segment)
    trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    valid_set = CommMixDataset(root=root, split="cv", segment=segment)
    validLoader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_set = CommMixDataset(root=root, split="tt", segment=segment)
    testLoader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    return trainLoader, validLoader, testLoader