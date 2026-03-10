import os
import torch
import torchaudio

# 导入你的 MultiScaleDeep 模型
from app.Models.Code.MultiScaleDeep.Model_TasNet_MultiScale import Model

# 如果没有本地权重，我们调用现成的 Asteroid 开源模型作为替补
try:
    from asteroid.models import ConvTasNet
except ImportError:
    ConvTasNet = None


class UnifiedInferenceCore:
    def __init__(self, model_type="multiscale_comm_2spk", ckpt_path=None, device=None):
        """
        统一的纯推理接口，供 GUI 直接调用。

        :param model_type: 支持以下三种模式
            - 'multiscale_comm_2spk': 通信信号 2 源分离 (对应 trainTasNet2.py)
            - 'multiscale_speech_4spk': 语音信号 4 源分离 (对应 trainTasNet.py)
            - 'pretrained_asteroid': 网上现成的开源模型 (无参数文件时的替补方案)
        :param ckpt_path: 你的 .pt 权重文件路径
        :param device: 强制指定设备，默认自动选择 cuda 或 cpu
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.model = None

        self._build_and_load()

    def _build_and_load(self):
        if self.model_type == "multiscale_comm_2spk":
            # 严格对齐 trainTasNet2.py 的配置
            self.model = Model(
                enc_dim=512, enc_kernel=(11, 21, 31), enc_stride=8,
                bottleneck_channels=128, hidden_channels=256,
                num_blocks=8, num_repeats=3, num_speakers=2
            ).to(self.device)
            self._load_local_weights()

        elif self.model_type == "multiscale_speech_3spk":
            # 严格对齐 trainTasNet.py 的配置
            self.model = Model(
                enc_dim=516, enc_kernel=(11, 21, 31), enc_stride=8,
                bottleneck_channels=128, hidden_channels=256,
                num_blocks=8, num_repeats=3, num_speakers=3
            ).to(self.device)
            self._load_local_weights()

        elif self.model_type == "multiscale_speech_4spk":
            # 严格对齐 trainTasNet.py 的配置
            self.model = Model(
                enc_dim=516, enc_kernel=(11, 21, 31), enc_stride=8,
                bottleneck_channels=128, hidden_channels=256,
                num_blocks=8, num_repeats=3, num_speakers=4
            ).to(self.device)
            self._load_local_weights()

        elif self.model_type == "pretrained_asteroid":
            # 现成模型的调用接口 (例如 WHAM 数据集上预训练的 ConvTasNet)
            if ConvTasNet is None:
                raise ImportError("请先安装 asteroid: pip install asteroid")
            print("正在从网络加载现成的预训练模型...")
            self.model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM!_sepclean").to(self.device)

        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        # 强制切换到评估模式，关闭 Dropout 和 BatchNorm 的动态更新
        self.model.eval()

    def _load_local_weights(self):
        """复用你写好的稳健加载逻辑，优先提取 ema_state"""
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            print(f"警告: 找不到权重文件 {self.ckpt_path}，将使用随机初始化权重。")
            return

        print(f"正在加载本地模型权重: {self.ckpt_path}")
        ckpt = torch.load(self.ckpt_path, map_location=self.device)

        if isinstance(ckpt, dict):
            if "ema_state" in ckpt:
                sd = ckpt["ema_state"]
            elif "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            else:
                sd = ckpt
        else:
            sd = ckpt

        # 清洗键名，去除 module. 前缀并剔除 n_averaged
        if isinstance(sd, dict):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            if "n_averaged" in sd:
                sd.pop("n_averaged")

        self.model.load_state_dict(sd, strict=True)
        print("本地权重加载成功！")

    @torch.no_grad()  # 显式禁用梯度，严格控制显存使用
    def separate(self, mix_waveform):
        """
        供 GUI 调用的核心分离函数。
        :param mix_waveform: 输入的混合音频 Tensor (可为一维或二维)
        :return: 分离后的多源音频 Tensor，位于 CPU 上，形状为 (num_speakers, T)
        """
        # 统一维度整理为 (Batch, 1, Time)
        if mix_waveform.dim() == 1:
            mix_waveform = mix_waveform.unsqueeze(0).unsqueeze(0)
        elif mix_waveform.dim() == 2:
            mix_waveform = mix_waveform.unsqueeze(1)

        mix_waveform = mix_waveform.to(self.device)

        # 开启 bfloat16 混合精度加速推理，提升界面响应速度
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            est_sources = self.model(mix_waveform)

        # 如果使用的是 Asteroid 等现成模型，抹平输出维度差异
        if est_sources.dim() == 4 and est_sources.size(1) == 1:
            est_sources = est_sources.squeeze(1)

        # 长度对齐并去除直流分量 (避免播放时出现爆音)
        min_len = min(est_sources.shape[-1], mix_waveform.shape[-1])
        est_sources = est_sources[..., :min_len]
        est_sources = est_sources - est_sources.mean(dim=-1, keepdim=True)

        # 将结果移至 CPU 并压缩 Batch 维度，方便前端 UI 绘图和保存
        return est_sources.squeeze(0).cpu()