import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


def load_audio(path, target_sr=8000):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


class WSJ0MixDataset(Dataset):
    def __init__(self, root, split="tr", sample_rate=8000, segment=4.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment

        self.mix_list = sorted(glob.glob(os.path.join(root, split, "mix", "*.wav")))
        self.s1_list  = sorted(glob.glob(os.path.join(root, split, "s1", "*.wav")))
        self.s2_list  = sorted(glob.glob(os.path.join(root, split, "s2", "*.wav")))

        assert len(self.mix_list) == len(self.s1_list) == len(self.s2_list)
        self.seg_len = int(segment * sample_rate)

    def __len__(self):
        return len(self.mix_list)

    def __getitem__(self, idx):
        mix = load_audio(self.mix_list[idx], self.sample_rate)[0]
        s1  = load_audio(self.s1_list[idx],  self.sample_rate)[0]
        s2  = load_audio(self.s2_list[idx],  self.sample_rate)[0]

        min_len = min(mix.shape[-1], s1.shape[-1], s2.shape[-1])
        mix, s1, s2 = mix[:min_len], s1[:min_len], s2[:min_len]

        if min_len < self.seg_len:
            pad = self.seg_len - min_len
            mix = torch.nn.functional.pad(mix, (0, pad))
            s1  = torch.nn.functional.pad(s1, (0, pad))
            s2  = torch.nn.functional.pad(s2, (0, pad))
        else:
            mix = mix[:self.seg_len]
            s1  = s1[:self.seg_len]
            s2  = s2[:self.seg_len]

        return mix, torch.stack([s1, s2], dim=0)

def MakeDataLoader():
    train_set = WSJ0MixDataset(root="Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min", split="tr", sample_rate=8000, segment=4.0)
    trainLoader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    valid_set = WSJ0MixDataset(root="Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min", split="cv", sample_rate=8000, segment=4.0)
    validLoader = DataLoader(valid_set, batch_size=4, shuffle=False, num_workers=4, drop_last=True)
    test_set = WSJ0MixDataset(root="Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min", split="cv", sample_rate=8000,
                               segment=4.0)
    testLoader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4, drop_last=True)

    return trainLoader, validLoader, testLoader

if __name__ == "__main__":
    trainLoader, validLoader, testLoader = MakeDataLoader()
    for mix, targets in validLoaders:
        # mix: (B, T)
        # targets: (B, 2, T)
        print(mix.shape, targets.shape)
        break