import os
import torch

# ========================================================
# 【关键修复】保留你原代码中的 PyTorch 2.6 安检劫持
# 防止界面加载老权重或 Asteroid 预训练模型时崩溃
# ========================================================
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load
# ========================================================

# 引入 Asteroid 模型
from asteroid.models import DPRNNTasNet


class DPRNNSignalSeparator:
    def __init__(self, model_type="dprnn_comm", ckpt_path=None, device=None):
        """
        DPRNN 专属推理核心类，完美适配你的两种应用场景。
        :param model_type: "dprnn_comm" (你自己训的通信版) 或 "dprnn_pretrained_wsj" (Asteroid官方语音版)
        :param ckpt_path: 自定义通信版模型的权重路径
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.model = None

        self._load_model()

    def _load_model(self):
        if self.model_type == "dprnn_comm":
            # 你的自定义轻量化 DPRNN 配置 (针对通信长序列优化)
            print("Initializing custom DPRNNTasNet (Comm)...")
            self.model = DPRNNTasNet(
                n_src=2, out_chan=64, in_chan=64, bn_chan=128,
                hid_size=128, chunk_size=100, hop_size=50, n_repeats=4
            ).to(self.device)

            if self.ckpt_path and os.path.exists(self.ckpt_path):
                self._load_weights(self.ckpt_path)
            else:
                print(f"Warning: Checkpoint not found at {self.ckpt_path}.")

        elif self.model_type == "dprnn_pretrained_wsj":
            # 直接调用 Asteroid Hub 上最强的 WHAM 预训练模型
            print("Downloading/Loading pre-trained DPRNNTasNet from Asteroid Hub...")
            self.model = DPRNNTasNet.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean").to(self.device)
            print("Pre-trained model loaded successfully.")

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # 锁定推理模式
        self.model.eval()

    def _load_weights(self, path):
        """稳健的 EMA 权重加载机制"""
        print(f"Loading weights from {path}...")
        ckpt = torch.load(path, map_location=self.device)

        if isinstance(ckpt, dict):
            if "ema_state" in ckpt:
                sd = ckpt["ema_state"]
            elif "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            else:
                sd = ckpt
        else:
            sd = ckpt

        # 清洗模块前缀和平均计数器
        if isinstance(sd, dict):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            if "n_averaged" in sd:
                sd.pop("n_averaged")

        self.model.load_state_dict(sd, strict=True)
        print("Model weights loaded successfully.")

    @torch.no_grad()
    def separate(self, mix_waveform):
        """
        执行信号分离 (适配 Asteroid 张量形状)
        :param mix_waveform: 混合音频 Tensor
        """
        # Asteroid DPRNN 通常直接接受 (Batch, Time) 格式
        # 如果传入的是 (1, T) 或 (T,)，我们将其整理为 (1, T)
        if mix_waveform.dim() == 1:
            mix_waveform = mix_waveform.unsqueeze(0)
        elif mix_waveform.dim() == 2 and mix_waveform.size(0) != 1:
            # 假设传入了 (T, 1)，转置为 (1, T)
            mix_waveform = mix_waveform.transpose(0, 1)
        elif mix_waveform.dim() == 3:
            # 去除多余的通道维度 (1, 1, T) -> (1, T)
            mix_waveform = mix_waveform.squeeze(1)

        mix_waveform = mix_waveform.to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            est_sources = self.model(mix_waveform)

        # 抹平 Asteroid 输出维度的不确定性：
        # 如果模型输出是 (Batch, 1, Spk, Time)，压缩掉多余维度 -> (Batch, Spk, Time)
        if est_sources.dim() == 4 and est_sources.size(1) == 1:
            est_sources = est_sources.squeeze(1)

        # 长度对齐
        min_len = min(est_sources.shape[-1], mix_waveform.shape[-1])
        est_sources = est_sources[..., :min_len]

        # 去除直流分量 (你在代码里做的优化操作，有助于后续播放或画图)
        est_sources = est_sources - est_sources.mean(dim=-1, keepdim=True)

        return est_sources.squeeze(0).cpu()  # 返回形状 (2, T)