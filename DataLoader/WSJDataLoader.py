import os
import glob
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

def load_audio(path, target_sr=8000):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav

class WSJ0MixDataset(Dataset):
    def __init__(self, root, split="tr", sample_rate=8000, segment=4.0, num_speakers=2):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.num_speakers = num_speakers
        self.mix_list = sorted(glob.glob(os.path.join(root, split, "mix", "*.wav")))
        self.s_lists = []
        for i in range(1, num_speakers + 1):
            s_list = sorted(glob.glob(os.path.join(root, split, f"s{i}", "*.wav")))
            assert len(s_list) == len(self.mix_list), f"数量不匹配: mix有{len(self.mix_list)}个, s{i}有{len(s_list)}个"
            self.s_lists.append(s_list)
        self.seg_len = int(segment * sample_rate)
    def __len__(self):
        return len(self.mix_list)

    def __getitem__(self, idx):
        mix = load_audio(self.mix_list[idx], self.sample_rate)[0]
        sources = [load_audio(s_list[idx], self.sample_rate)[0] for s_list in self.s_lists]
        min_len = min([mix.shape[-1]] + [s.shape[-1] for s in sources])
        if min_len <= self.seg_len:
            pad = self.seg_len - min_len
            mix = torch.nn.functional.pad(mix, (0, pad))
            sources = [torch.nn.functional.pad(s, (0, pad)) for s in sources]
        else:
            start = random.randint(0, min_len - self.seg_len)
            end = start + self.seg_len
            mix = mix[start:end]
            sources = [s[start:end] for s in sources]
        return mix, torch.stack(sources, dim=0)

def MakeDataLoader(root, batch_size, num_workers, segment, sample_rate, num_speakers=2):
    train_set = WSJ0MixDataset(root=root, split="tr", sample_rate=sample_rate, segment=segment,
                               num_speakers=num_speakers)
    trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    valid_set = WSJ0MixDataset(root=root, split="cv", sample_rate=sample_rate, segment=segment,
                               num_speakers=num_speakers)
    validLoader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_set = WSJ0MixDataset(root=root, split="tt", sample_rate=sample_rate, segment=segment,
                              num_speakers=num_speakers)
    testLoader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    return trainLoader, validLoader, testLoader

if __name__ == "__main__":
    root = "../Dataset/WSJ0-3Mix/3speakers/wav8k/min"
    sample_rate = 8000
    trainLoader, validLoader, testLoader = MakeDataLoader(
        root=root, batch_size=4, num_workers=4, segment=4.0, sample_rate=sample_rate, num_speakers=3
    )
    for mix, targets in validLoader:
        print("Mix shape:", mix.shape)
        print("Targets shape:", targets.shape)
        break