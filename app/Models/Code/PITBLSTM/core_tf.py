import os
import torch
import torch.nn as nn


class STFTMaskNet(nn.Module):
    """A clean version of the time-frequency domain benchmark model (PITBLSTM), removing all code unrelated to inference."""

    def __init__(self, n_fft=512, hop_length=128, hidden_size=300, num_layers=3, num_spks=2):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_spks = num_spks
        input_dim = n_fft // 2 + 1

        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.linear = nn.Linear(hidden_size * 2, input_dim * num_spks)
        self.activation = nn.Sigmoid()

    # Force AMP off to prevent STFT numerical overflow at half precision (float16/bfloat16)
    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        x = x.float()
        B = x.shape[0]
        device = x.device

        window = torch.hann_window(self.n_fft).to(device)
        stft_res = torch.stft(x, self.n_fft, self.hop_length, window=window, return_complex=True)
        mag = stft_res.abs()
        phase = stft_res.angle()

        log_mag = torch.log1p(mag).permute(0, 2, 1)

        rnn_out, _ = self.lstm(log_mag)
        mask_out = self.linear(rnn_out)
        mask_out = self.activation(mask_out)

        mask_out = mask_out.view(B, -1, self.num_spks, self.n_fft // 2 + 1)
        mask_out = mask_out.permute(0, 2, 3, 1)

        est_mags = mag.unsqueeze(1) * mask_out

        est_wavs = []
        for i in range(self.num_spks):
            m = est_mags[:, i]
            comp = m * torch.exp(1j * phase)
            wav = torch.istft(comp, self.n_fft, self.hop_length, window=window, length=x.shape[-1])
            est_wavs.append(wav)

        return torch.stack(est_wavs, dim=1)


class TFBaselineInferenceCore:
    def __init__(self, model_type="tf_baseline_wsj", ckpt_path=None, device=None):
        """Pure inference interface for time-frequency domain benchmark models :param model_type: 'tf_baseline_wsj' (speech) or 'tf_baseline_comm' (communication) :param ckpt_path: the local local.pt weight path"""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.model = None

        self._build_and_load()

    def _build_and_load(self):
        if self.model_type == "tf_baseline_wsj":
            # Strictly align the configuration of core_baseline.py
            self.model = STFTMaskNet(n_fft=512, hop_length=128).to(self.device)
        elif self.model_type == "tf_baseline_comm":
            # Strictly align the configuration of run_Comm.py
            self.model = STFTMaskNet(n_fft=1024, hop_length=256).to(self.device)
        else:
            raise ValueError(f"不支持的基准模型类型: {self.model_type}")

        self._load_local_weights()
        self.model.eval()

    def _load_local_weights(self):
        """Load local weights and output a Warning: if they don't exist (no crash on GUI startup)"""
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            print(f"⚠️ 警告: 找不到本地权重文件 {self.ckpt_path}。")
            print("系统已加载网络架构，如果用于 GUI 演示，分离结果将基于随机初始化的参数。")
            return

        print(f"正在加载 TF-Baseline 本地权重: {self.ckpt_path}")
        state_dict = torch.load(self.ckpt_path, map_location=self.device)

        # Compatible with EMA saved module. prefix
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state_dict, strict=True)
        print("本地权重加载成功！")

    @torch.no_grad()
    def separate(self, mix_waveform):
        """Core detached functions called by the GUI."""
        # Unified dimensions are organized into (Batch, Time) to adapt to STFTMaskNet
        if mix_waveform.dim() == 1:
            mix_waveform = mix_waveform.unsqueeze(0)
        elif mix_waveform.dim() == 2 and mix_waveform.size(0) != 1:
            mix_waveform = mix_waveform.transpose(0, 1)
        elif mix_waveform.dim() == 3:
            mix_waveform = mix_waveform.squeeze(1)

        mix_waveform = mix_waveform.to(self.device)

        # forward reasoning
        est_sources = self.model(mix_waveform)

        # Length alignment and DC component removal
        min_len = min(est_sources.shape[-1], mix_waveform.shape[-1])
        est_sources = est_sources[..., :min_len]
        est_sources = est_sources - est_sources.mean(dim=-1, keepdim=True)

        return est_sources.squeeze(0).cpu()