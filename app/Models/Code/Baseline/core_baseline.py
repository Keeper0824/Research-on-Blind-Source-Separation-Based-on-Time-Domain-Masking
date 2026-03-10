import os
import torch
import torchaudio
from app.Models.Code.Baseline.Model_TasNet import Model


class SignalSeparator:
    def __init__(self, model_type="custom_tasnet", ckpt_path=None, device=None):
        """
        推理核心类，用于 GUI 调用。
        :param model_type: 模型类型，比如 "custom_tasnet" 或是未来要加的 "pretrained_xxx"
        :param ckpt_path: 自定义模型的权重路径
        :param device: "cuda" 或 "cpu"
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.model = None

        self._load_model()

    def _load_model(self):
        if self.model_type == "custom_tasnet":
            # 这里对应你之前消融实验确定的最佳架构参数
            self.model = Model(
                enc_dim=512, enc_kernel=16, enc_stride=8,
                bottleneck_channels=128, hidden_channels=256,
                num_blocks=8, num_repeats=3, num_speakers=2
            ).to(self.device)

            if self.ckpt_path and os.path.exists(self.ckpt_path):
                self._load_weights(self.ckpt_path)
            else:
                print(f"Warning: Checkpoint not found at {self.ckpt_path}. Using uninitialized weights.")

        elif self.model_type == "tasnet_base_comm":
            # 预留给你的直接调用的现成模型 (例如 SpeechBrain 的 SepFormer 等)
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

        self.model.eval()  # 强制设置为推理模式，关闭 Dropout 和 BatchNorm 等

    def _load_weights(self, path):
        """复用你写好的稳健的权重加载逻辑，提取 ema_state"""
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

        # 去掉 module. 前缀并剔除 n_averaged 防止加载报错
        if isinstance(sd, dict):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            if "n_averaged" in sd:
                sd.pop("n_averaged")

        self.model.load_state_dict(sd, strict=True)
        print("Model weights loaded successfully.")

    @torch.no_grad()  # 显式禁用梯度计算，节省显存并加速推理
    def separate(self, mix_waveform):
        """
        执行信号分离
        :param mix_waveform: Tensor of shape (1, T) or (T,)
        :return: separated_waveforms: Tensor of shape (num_speakers, T)
        """
        # 1. 维度整理：确保输入形状为 (Batch, 1, Time)
        if mix_waveform.dim() == 1:
            mix_waveform = mix_waveform.unsqueeze(0).unsqueeze(0)
        elif mix_waveform.dim() == 2:
            mix_waveform = mix_waveform.unsqueeze(1)

        mix_waveform = mix_waveform.to(self.device)

        # 2. 推理执行 (可选择是否开启 AMP 加速)
        # 由于是纯推理，混合精度 (bfloat16) 依然可以用来提速
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            est_sources = self.model(mix_waveform)  # 输出 shape: (1, 2, T_est)

        # 3. 长度对齐：如果卷积后长度有微小变化，裁剪到原长
        min_len = min(est_sources.shape[-1], mix_waveform.shape[-1])
        est_sources = est_sources[..., :min_len]

        # 返回形状为 (2, T) 的张量放回 CPU，方便 GUI 进行绘图或保存
        return est_sources.squeeze(0).cpu()