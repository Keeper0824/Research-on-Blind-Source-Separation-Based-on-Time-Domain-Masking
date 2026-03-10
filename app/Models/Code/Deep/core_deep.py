import os
import torch
from app.Models.Code.Deep.Model_TasNet import Model


class DeepSignalSeparator:
    def __init__(self, model_type="deep_tasnet", ckpt_path=None, device=None):
        """
        深度优化版模型的推理核心类，专为交互式界面设计。
        :param model_type: "deep_tasnet" 或其他预训练模型标识
        :param ckpt_path: 你的 .pt 权重文件路径
        :param device: 强制指定设备，默认自动选择 cuda 或 cpu
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.model = None

        self._load_model()

    def _load_model(self):
        if self.model_type == "deep_tasnet":
            # 初始化你在 Deep 版本中确定的最佳超参数
            self.model = Model(
                enc_dim=512, enc_kernel=16, enc_stride=8,
                bottleneck_channels=128, hidden_channels=256,
                num_blocks=8, num_repeats=3, num_speakers=2
            ).to(self.device)

            if self.ckpt_path and os.path.exists(self.ckpt_path):
                self._load_weights(self.ckpt_path)
            else:
                print(f"Warning: Checkpoint not found at {self.ckpt_path}. Using uninitialized weights.")

        elif self.model_type == "tasnet_deep_comm":
            self.model = Model(
                enc_dim=512, enc_kernel=16, enc_stride=8,
                bottleneck_channels=128, hidden_channels=256,
                num_blocks=8, num_repeats=3, num_speakers=2
            ).to(self.device)

            if self.ckpt_path and os.path.exists(self.ckpt_path):
                self._load_weights(self.ckpt_path)
            else:
                print(f"Warning: Checkpoint not found at {self.ckpt_path}. Using uninitialized weights.")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # 锁定为推理模式
        self.model.eval()

    def _load_weights(self, path):
        """复用你在测试脚本中写好的、最稳健的 EMA 权重加载机制"""
        print(f"Loading weights from {path}...")
        ckpt = torch.load(path, map_location=self.device)

        # 优先寻找并使用 ema_state 获取更好的分离性能
        if isinstance(ckpt, dict):
            if "ema_state" in ckpt:
                print("Found 'ema_state'. Loading EMA weights for better performance...")
                sd = ckpt["ema_state"]
            elif "ema_state_dict" in ckpt:
                print("Found 'ema_state_dict'. Loading EMA weights...")
                sd = ckpt["ema_state_dict"]
            elif "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            elif "model_state" in ckpt:
                sd = ckpt["model_state"]
            elif "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            else:
                sd = ckpt
        else:
            sd = ckpt

        # 清洗权重字典的键值
        if isinstance(sd, dict):
            # 去掉可能的 DataParallel 产生的 module. 前缀
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            # 去掉 EMA 产生的计数器，防止严格加载模式报错
            if "n_averaged" in sd:
                sd.pop("n_averaged")

        self.model.load_state_dict(sd, strict=True)
        print("Model weights loaded successfully.")

    @torch.no_grad()
    def separate(self, mix_waveform):
        """
        接收混合音频 Tensor 并输出分离后的音频 Tensor。
        :param mix_waveform: 形状为 (1, T) 或 (T,) 的 Tensor
        :return: 形状为 (num_speakers, T) 的分离后 Tensor (位于 CPU 上，方便后续保存或画图)
        """
        # 维度整理，适配网络输入 (Batch, 1, Time)
        if mix_waveform.dim() == 1:
            mix_waveform = mix_waveform.unsqueeze(0).unsqueeze(0)
        elif mix_waveform.dim() == 2:
            mix_waveform = mix_waveform.unsqueeze(1)

        mix_waveform = mix_waveform.to(self.device)

        # 开启 bfloat16 混合精度加速推理
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            est_sources = self.model(mix_waveform)  # 输出 (1, 2, T_est)

        # 截断由于卷积 padding 可能多出来的一点点长度
        min_len = min(est_sources.shape[-1], mix_waveform.shape[-1])
        est_sources = est_sources[..., :min_len]

        # 返回到 CPU，降维为 (2, T)
        return est_sources.squeeze(0).cpu()