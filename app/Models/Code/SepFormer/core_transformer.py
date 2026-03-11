import os
import torch

# ========================================================
# 继承自 run_wsj.py 的终极兼容性补丁 (GUI 稳定运行的保障)
# ========================================================
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load

import huggingface_hub

_orig_hf_hub_download = huggingface_hub.hf_hub_download


def _patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs: kwargs['token'] = kwargs.pop('use_auth_token')
    filename = kwargs.get('filename', args[1] if len(args) > 1 else '')
    try:
        return _orig_hf_hub_download(*args, **kwargs)
    except Exception as e:
        if filename == "custom.py" and "404" in str(e):
            dummy_path = os.path.join(os.getcwd(), "dummy_custom.py")
            if not os.path.exists(dummy_path):
                with open(dummy_path, "w", encoding="utf-8") as f:
                    f.write("# 占位文件，绕过 404\n")
            return dummy_path
        raise e


huggingface_hub.hf_hub_download = _patched_hf_hub_download

import pathlib
import shutil

_orig_symlink_to = pathlib.Path.symlink_to


def _patched_symlink_to(self, target, target_is_directory=False):
    try:
        _orig_symlink_to(self, target, target_is_directory)
    except OSError as e:
        if getattr(e, 'winerror', None) == 1314:
            if target.is_dir():
                shutil.copytree(target, self)
            else:
                shutil.copy(target, self)
        else:
            raise


pathlib.Path.symlink_to = _patched_symlink_to
# ========================================================

try:
    from asteroid.models import DPTNet
except ImportError:
    DPTNet = None

try:
    from speechbrain.inference.separation import SepformerSeparation
except ImportError:
    SepformerSeparation = None


class DPTNetInferenceCore:
    def __init__(self, ckpt_path=None, device=None):
        """Asteroid DPTNet 推理核心"""
        if DPTNet is None:
            raise ImportError("请先安装 asteroid: pip install asteroid")

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path
        self.model = DPTNet(
            n_src=2, out_chan=64, in_chan=64, ff_hid=128,
            chunk_size=100, hop_size=50, n_repeats=6,
            norm_type="gLN", mask_act="relu"
        ).to(self.device)

        self._load_weights()
        self.model.eval()

    def _load_weights(self):
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            print(f"正在加载 DPTNet 本地权重: {self.ckpt_path}")
            sd = torch.load(self.ckpt_path, map_location=self.device)
            if isinstance(sd, dict):
                sd = sd.get("ema_state", sd.get("model_state_dict", sd))
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
                sd.pop("n_averaged", None)
            self.model.load_state_dict(sd, strict=False)
        else:
            print("⚠️ 警告: 未找到 DPTNet 本地权重，将使用随机初始化参数进行演示。")

    @torch.no_grad()
    def separate(self, mix_waveform):
        # 🚀 修复核心：正确地把音频整理成 [Batch, Time] (即 [1, T]) 格式
        if mix_waveform.dim() == 1:
            mix_waveform = mix_waveform.unsqueeze(0)
        elif mix_waveform.dim() == 2 and mix_waveform.shape[0] != 1:
            mix_waveform = mix_waveform.mean(dim=0, keepdim=True)
        elif mix_waveform.dim() == 3:
            mix_waveform = mix_waveform.squeeze(1)

        mix_waveform = mix_waveform.to(self.device).float()

        # 强制使用 float32，避免 DPTNet 内部的 Transformer 产生精度溢出
        est = self.model(mix_waveform)
        if est.dim() == 3: est = est.squeeze(0)  # (Spk, T)
        return (est - est.mean(dim=-1, keepdim=True)).cpu()


class SepFormerInferenceCore:
    def __init__(self, device=None):
        """SpeechBrain SepFormer 现成模型推理核心"""
        if SepformerSeparation is None:
            raise ImportError("请先安装 speechbrain: pip install speechbrain")

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print("正在初始化并下载/加载 SpeechBrain SepFormer 现成预训练模型...")
        self.model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_models/sepformer-wsj02mix",
            run_opts={"device": self.device}
        )
        self.model.eval()

    @torch.no_grad()
    def separate(self, mix_waveform):
        # 🚀 修复核心：同上，防止时间维度被一刀切成 1
        if mix_waveform.dim() == 1:
            mix_waveform = mix_waveform.unsqueeze(0)
        elif mix_waveform.dim() == 2 and mix_waveform.shape[0] != 1:
            mix_waveform = mix_waveform.mean(dim=0, keepdim=True)
        elif mix_waveform.dim() == 3:
            mix_waveform = mix_waveform.squeeze(1)

        mix_waveform = mix_waveform.to(self.device).float()

        # 调用 SpeechBrain 专属的推理函数
        est = self.model.separate_batch(mix_waveform)

        # 修正输出形状 (Batch, Time, Spk) -> (Spk, Time)
        if est.dim() == 3:
            est = est.permute(0, 2, 1).squeeze(0)

        return (est - est.mean(dim=-1, keepdim=True)).cpu()