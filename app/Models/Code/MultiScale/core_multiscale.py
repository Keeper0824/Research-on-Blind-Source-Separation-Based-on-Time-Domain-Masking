import os
import torch

# Point to the local multi-scale model architecture
from app.Models.Code.MultiScale.Model_TasNet_MultiScale import Model


class MultiScaleInferenceCore:
    def __init__(self, ckpt_path=None, device=None):
        """The inference core class of the MultiScale final optimized model. Use Golden Config: enc_kernel=[8, 16, 24], enc_dim=516"""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path
        self.model = None

        self._load_model()

    def _load_model(self):
        # Strict alignment Golden Config in trainTasNet2.py
        print("Initializing Multi-Scale Model with Golden Config...")
        self.model = Model(
            enc_dim=516,
            enc_kernel=[8, 16, 24],
            enc_stride=8,
            bottleneck_channels=128,
            hidden_channels=256,
            num_blocks=8,
            num_repeats=3,
            num_speakers=2
        ).to(self.device)

        if self.ckpt_path and os.path.exists(self.ckpt_path):
            self._load_weights(self.ckpt_path)
        else:
            print(f"Warning: Checkpoint not found at {self.ckpt_path}.")

        self.model.eval()

    def _load_weights(self, path):
        """Reuse the compatible weight-loading path from core_baseline.py"""
        print(f"Loading MultiScale weights from {path}...")
        ckpt = torch.load(path, map_location=self.device)

        if isinstance(ckpt, dict):
            if "ema_state" in ckpt:
                print("Found 'ema_state'. Loading EMA weights...")
                sd = ckpt["ema_state"]
            elif "model_state" in ckpt:
                print("Found 'model_state'. Loading best model weights...")
                sd = ckpt["model_state"]
            elif "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            else:
                sd = ckpt
        else:
            sd = ckpt

        # Clean key names
        if isinstance(sd, dict):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            if "n_averaged" in sd:
                sd.pop("n_averaged")

        self.model.load_state_dict(sd, strict=True)
        print("MultiScale Model weights loaded successfully.")

    @torch.no_grad()
    def separate(self, mix_waveform):
        # The sorting dimension is (Batch, 1, Time)
        if mix_waveform.dim() == 1:
            mix_waveform = mix_waveform.unsqueeze(0).unsqueeze(0)
        elif mix_waveform.dim() == 2:
            mix_waveform = mix_waveform.unsqueeze(1)

        mix_waveform = mix_waveform.to(self.device)

        est_sources = self.model(mix_waveform)

        # Use float16 mixed precision acceleration (consistent with trainTasNet2.py)
        min_len = min(est_sources.shape[-1], mix_waveform.shape[-1])
        est_sources = est_sources[..., :min_len]

        return est_sources.squeeze(0).cpu()