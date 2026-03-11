import os
import torch
import torchaudio
import numpy as np
import gc

# ========================================================
# 【全局系统级护城河】
# 集中执行所有的猴子补丁 (Monkey Patches)
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

# 导入所有独立的推理核心
try:
    from app.Models.Code.Baseline.core_baseline import SignalSeparator
    from app.Models.Code.Deep.core_deep import DeepSignalSeparator
    from app.Models.Code.MultiScale.core_multiscale import MultiScaleInferenceCore
    from app.Models.Code.MultiScaleDeep.core_multideep import UnifiedInferenceCore
    from app.Models.Code.PITBLSTM.core_tf import TFBaselineInferenceCore
    from app.Models.Code.DPRNN.core_asteroid import DPRNNSignalSeparator
    from app.Models.Code.SepFormer.core_transformer import DPTNetInferenceCore, SepFormerInferenceCore
    from app.Models.Code.IRM.core_irm import IRMInferenceCore
except ImportError as e:
    print(f"导入 Core 模块失败，请确保 core_xxx.py 文件在正确位置: {e}")


class SeparationManager:
    """
    GUI 前端的终极总控台。支持语音(.wav)与通信基带(.npy)双模读取与推理。
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"总控台初始化完毕，当前使用设备: {self.device}")

        self.active_model = None
        self.current_model_name = ""

        # ==========================================
        # 🚀 使用动态绝对路径，彻底告别路径报错！
        # ==========================================
        # 1. 获取当前 ModelManager.py 所在的真实目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. 向上追溯到项目根目录 (app文件夹)。
        # 如果你的 ModelManager 和 Speech 文件夹在同一级，直接用 base_dir = current_dir 即可。
        # 如果在 app/Models/Code 下，就需要向上退两级：
        base_dir = os.path.normpath(os.path.join(current_dir, ".."))

        # 打印出来检查一下到底对不对
        print(f"[路径探针] 动态解析的项目根目录为: {base_dir}")

        # 3. 动态拼接所有模型的绝对路径
        self.ckpt_paths = {
            # --- 语音部分 (Speech) ---
            "tasnet_base": os.path.join(base_dir, "Speech", "Baseline.pt"),
            "tasnet_deep": os.path.join(base_dir, "Speech", "Deep.pt"),
            "tasnet_multi": os.path.join(base_dir, "Speech", "Multiscale.pt"),
            "tasnet_multideep": os.path.join(base_dir, "Speech", "MultiDeep.pt"),
            "tf_baseline": os.path.join(base_dir, "Speech", "PIT-BLSTM.pt"),
            "3speaker": os.path.join(base_dir, "Speech", "3speaker.pt"),
            "4speaker": os.path.join(base_dir, "Speech", "4speaker.pt"),

            # --- 通信部分 (Comm) ---
            "tasnet_base_comm": os.path.join(base_dir, "Comm", "baseline.pt"),
            "tasnet_deep_comm": os.path.join(base_dir, "Comm", "deep.pt"),
            "tasnet_multi_comm": os.path.join(base_dir, "Comm", "multiscale.pt"),
            "tf_baseline_comm": os.path.join(base_dir, "Comm", "PIT-BLSTM.pt"),
            "dprnn_comm": os.path.join(base_dir, "Comm", "DPRNN.pt"),
            "sep_comm": os.path.join(base_dir, "Comm", "SepFormer.pt")
        }

    def load_model(self, model_choice):
        """
        动态装载对应的模型。精准区分了 Speech 和 Comm 领域模型。
        """
        if model_choice == self.current_model_name and self.active_model is not None:
            return

        print(f"\n[ModelManager] 正在切换模型后端至 🚀: {model_choice} ...")

        if self.active_model is not None:
            del self.active_model
            self.active_model = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # -------------------- 【语音分离 (Speech)】系列 --------------------
        if model_choice == "【语音】TasNet (基础架构)":
            self.active_model = SignalSeparator(ckpt_path=self.ckpt_paths["tasnet_base"], device=self.device)

        elif model_choice == "【语音】Deep TasNet (深度优化)":
            self.active_model = DeepSignalSeparator(ckpt_path=self.ckpt_paths["tasnet_deep"], device=self.device)

        elif model_choice == "【语音】MultiScale-Deep (综合终极版)":
            # 默认调用双源语音模型
            self.active_model = UnifiedInferenceCore(model_type="multiscale_speech_2spk",
                                                     ckpt_path=self.ckpt_paths["tasnet_multideep"], device=self.device)

        elif model_choice == "【语音】TasNet (3说话人分离)":
            self.active_model = UnifiedInferenceCore(model_type="multiscale_speech_3spk",
                                                     ckpt_path=self.ckpt_paths["3speaker"], device=self.device)

        elif model_choice == "【语音】TasNet (4说话人分离)":
            self.active_model = UnifiedInferenceCore(model_type="multiscale_speech_4spk",
                                                     ckpt_path=self.ckpt_paths["4speaker"], device=self.device)

        elif model_choice == "【语音】TF-BLSTM (时频域基准)":
            self.active_model = TFBaselineInferenceCore(model_type="tf_baseline_wsj",
                                                        ckpt_path=self.ckpt_paths["tf_baseline"], device=self.device)

        elif model_choice == "【语音】Asteroid DPRNN (现成预训练)":
            self.active_model = DPRNNSignalSeparator(model_type="dprnn_pretrained_wsj", device=self.device)

        elif model_choice == "【语音】SpeechBrain SepFormer (SOTA)":
            self.active_model = SepFormerInferenceCore(device=self.device)

        # -------------------- 【通信信号 (Comm)】系列 --------------------
        elif model_choice == "【通信】TasNet (基础架构)":
            self.active_model = SignalSeparator(ckpt_path=self.ckpt_paths["tasnet_base_comm"], device=self.device)

        elif model_choice == "【通信】Deep TasNet (深度优化)":
            self.active_model = DeepSignalSeparator(ckpt_path=self.ckpt_paths["tasnet_deep_comm"], device=self.device)

        elif model_choice == "【通信】MultiScale TasNet (多尺度版)":
            self.active_model = MultiScaleInferenceCore(ckpt_path=self.ckpt_paths["tasnet_multi_comm"],
                                                        device=self.device)

        elif model_choice == "【通信】TF-BLSTM (时频域基准)":
            self.active_model = TFBaselineInferenceCore(model_type="tf_baseline_comm",
                                                        ckpt_path=self.ckpt_paths["tf_baseline_comm"],
                                                        device=self.device)

        elif model_choice == "【通信】Asteroid DPRNN (长序列优化)":
            self.active_model = DPRNNSignalSeparator(model_type="dprnn_comm",
                                                     ckpt_path=self.ckpt_paths["dprnn_comm"], device=self.device)

        elif model_choice == "【通信】SepFormer / DPTNet (Transformer)":
            # 通信专用 Transformer 架构
            self.active_model = DPTNetInferenceCore(ckpt_path=self.ckpt_paths["sep_comm"], device=self.device)

        # -------------------- 【物理理论上限】 --------------------
        elif model_choice == "【理论上限】IRM Oracle (非盲分离)":
            self.active_model = IRMInferenceCore(device=self.device)

        else:
            raise ValueError(f"未知的模型选项: {model_choice}")

        self.current_model_name = model_choice
        print(f"[ModelManager] {model_choice} 加载完毕！✅\n")

    def _read_signal(self, file_path):
        """智能读取 .wav 或 .npy 文件"""
        if file_path.lower().endswith(".npy"):
            data = np.load(file_path)
            waveform = torch.from_numpy(data).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() > 2:
                waveform = waveform.squeeze()
                if waveform.dim() == 1: waveform = waveform.unsqueeze(0)
            sr = 100000
            sr = 8000
            return waveform, sr
        else:
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform, sr

    @torch.no_grad()
    def separate_audio(self, mix_path, model_choice, ref1_path=None, ref2_path=None):
        self.load_model(model_choice)
        print(f"[ModelManager] 正在读取混合信号: {mix_path}")
        mix_wav, sr = self._read_signal(mix_path)

        if "IRM Oracle" in model_choice:
            if not ref1_path or not ref2_path:
                raise ValueError("使用 IRM 模式必须同时上传两个干净的参考信号文件！")
            ref1_wav, _ = self._read_signal(ref1_path)
            ref2_wav, _ = self._read_signal(ref2_path)
            est_sources = self.active_model.separate(mix_wav, ref1_wav, ref2_wav)
        else:
            est_sources = self.active_model.separate(mix_wav)

        print("[ModelManager] 信号分离完成！")
        return est_sources, sr