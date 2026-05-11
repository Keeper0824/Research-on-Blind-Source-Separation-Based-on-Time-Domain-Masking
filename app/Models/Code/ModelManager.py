import os
import torch
import torchaudio
import numpy as np
import gc

# ========================================================
# [System-level monkey patches for compatibility]
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
                    f.write("# Dummy file to bypass 404\n")
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

# Import all independent inference cores
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
    print(f"[Error] Failed to import Core modules. Check paths: {e}")


class SeparationManager:
    """
    Backend manager for the GUI. Supports speech (.wav) and communication (.npy) inputs.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ModelManager] Initialized. Current device: {self.device}")

        self.active_model = None
        self.current_model_name = ""

        # Dynamic absolute paths setup
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.normpath(os.path.join(current_dir, ".."))
        print(f"[ModelManager] Base directory resolved to: {base_dir}")

        self.ckpt_paths = {
            # --- Speech ---
            "tasnet_base": os.path.join(base_dir, "Speech", "Baseline.pt"),
            "tasnet_deep": os.path.join(base_dir, "Speech", "Deep.pt"),
            "tasnet_multi": os.path.join(base_dir, "Speech", "Multiscale.pt"),
            "tasnet_multideep": os.path.join(base_dir, "Speech", "MultiDeep.pt"),
            "tf_baseline": os.path.join(base_dir, "Speech", "PIT-BLSTM.pt"),
            "3speaker": os.path.join(base_dir, "Speech", "3speaker.pt"),
            "4speaker": os.path.join(base_dir, "Speech", "4speaker.pt"),

            # --- Comm ---
            "tasnet_base_comm": os.path.join(base_dir, "Comm", "baseline.pt"),
            "tasnet_deep_comm": os.path.join(base_dir, "Comm", "deep.pt"),
            "tasnet_multi_comm": os.path.join(base_dir, "Comm", "multiscale.pt"),
            "tf_baseline_comm": os.path.join(base_dir, "Comm", "PIT-BLSTM.pt"),
            "dprnn_comm": os.path.join(base_dir, "Comm", "DPRNN.pt"),
            "sep_comm": os.path.join(base_dir, "Comm", "SepFormer.pt")
        }

    def load_model(self, model_choice):
        if model_choice == self.current_model_name and self.active_model is not None:
            return

        print(f"\n[ModelManager] Switching backend to: {model_choice} ...")

        if self.active_model is not None:
            del self.active_model
            self.active_model = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # -------------------- Speech Models --------------------
        if model_choice == "[Speech] TasNet (Baseline)":
            self.active_model = SignalSeparator(ckpt_path=self.ckpt_paths["tasnet_base"], device=self.device)
        elif model_choice == "[Speech] Deep TasNet":
            self.active_model = DeepSignalSeparator(ckpt_path=self.ckpt_paths["tasnet_deep"], device=self.device)
        elif model_choice == "[Speech] MultiScale TasNet":
            self.active_model = MultiScaleInferenceCore(ckpt_path=self.ckpt_paths["tasnet_multi"], device=self.device)
        elif model_choice == "[Speech] MultiScale-Deep (Ultimate)":
            self.active_model = UnifiedInferenceCore(model_type="multiscale_speech_2spk", ckpt_path=self.ckpt_paths["tasnet_multideep"], device=self.device)
        elif model_choice == "[Speech] TasNet (3 Speakers)":
            self.active_model = UnifiedInferenceCore(model_type="multiscale_speech_3spk", ckpt_path=self.ckpt_paths["3speaker"], device=self.device)
        elif model_choice == "[Speech] TasNet (4 Speakers)":
            self.active_model = UnifiedInferenceCore(model_type="multiscale_speech_4spk", ckpt_path=self.ckpt_paths["4speaker"], device=self.device)
        elif model_choice == "[Speech] TF-BLSTM (TF-Domain Baseline)":
            self.active_model = TFBaselineInferenceCore(model_type="tf_baseline_wsj", ckpt_path=self.ckpt_paths["tf_baseline"], device=self.device)
        elif model_choice == "[Speech] Asteroid DPRNN (Pre-trained)":
            self.active_model = DPRNNSignalSeparator(model_type="dprnn_pretrained_wsj", device=self.device)
        elif model_choice == "[Speech] SpeechBrain SepFormer (SOTA)":
            self.active_model = SepFormerInferenceCore(device=self.device)

        # -------------------- Comm Models --------------------
        elif model_choice == "[Comm] TasNet (Baseline)":
            self.active_model = SignalSeparator(ckpt_path=self.ckpt_paths["tasnet_base_comm"], device=self.device)
        elif model_choice == "[Comm] Deep TasNet":
            self.active_model = DeepSignalSeparator(ckpt_path=self.ckpt_paths["tasnet_deep_comm"], device=self.device)
        elif model_choice == "[Comm] MultiScale TasNet":
            self.active_model = MultiScaleInferenceCore(ckpt_path=self.ckpt_paths["tasnet_multi_comm"], device=self.device)
        elif model_choice == "[Comm] TF-BLSTM (TF-Domain Baseline)":
            self.active_model = TFBaselineInferenceCore(model_type="tf_baseline_comm", ckpt_path=self.ckpt_paths["tf_baseline_comm"], device=self.device)
        elif model_choice == "[Comm] Asteroid DPRNN (Long Seq)":
            self.active_model = DPRNNSignalSeparator(model_type="dprnn_comm", ckpt_path=self.ckpt_paths["dprnn_comm"], device=self.device)
        elif model_choice == "[Comm] SepFormer / DPTNet":
            self.active_model = DPTNetInferenceCore(ckpt_path=self.ckpt_paths["sep_comm"], device=self.device)

        # -------------------- IRM Oracle --------------------
        elif model_choice == "[Theoretical] IRM Oracle (Non-blind)":
            self.active_model = IRMInferenceCore(device=self.device)
        else:
            raise ValueError(f"Unknown model selection: {model_choice}")

        self.current_model_name = model_choice
        print(f"[ModelManager] {model_choice} successfully loaded!\n")

    def _read_signal(self, file_path):
        if file_path.lower().endswith(".npy"):
            data = np.load(file_path)
            waveform = torch.from_numpy(data).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() > 2:
                waveform = waveform.squeeze()
                if waveform.dim() == 1: waveform = waveform.unsqueeze(0)
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
        print(f"[ModelManager] Reading mixed signal: {mix_path}")
        mix_wav, sr = self._read_signal(mix_path)

        if "IRM Oracle" in model_choice:
            if not ref1_path or not ref2_path:
                raise ValueError("IRM Mode requires two clean reference signals!")
            ref1_wav, _ = self._read_signal(ref1_path)
            ref2_wav, _ = self._read_signal(ref2_path)
            est_sources = self.active_model.separate(mix_wav, ref1_wav, ref2_wav)
        else:
            est_sources = self.active_model.separate(mix_wav)

        print("[ModelManager] Signal separation completed!")
        return est_sources, sr