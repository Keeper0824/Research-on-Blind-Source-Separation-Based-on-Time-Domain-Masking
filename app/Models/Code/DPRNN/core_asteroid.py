import os
import torch

# ========================================================
# Compatibility patch: Keep PyTorch 2.6 load override in this code.
# Prevent the interface from crashing when loading old weights or Asteroid pretrained models
# ========================================================
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load
# ========================================================

# Import the Asteroid model
from asteroid.models import DPRNNTasNet


class DPRNNSignalSeparator:
    def __init__(self, model_type="dprnn_comm", ckpt_path=None, device=None):
        """DPRNN inference core for the two supported use cases. :param model_type: "dprnn_comm" (locally trained communication variant) or "dprnn_pretrained_wsj" (Asteroid official speech variant) :param ckpt_path: Path to the custom communication-model weights"""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.model = None

        self._load_model()

    def _load_model(self):
        if self.model_type == "dprnn_comm":
            # the local custom lightweight DPRNN configuration (optimized for communicating long sequences)
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
            # Load the WHAM pretrained model from Asteroid Hub
            print("Downloading/Loading pretrained DPRNNTasNet from Asteroid Hub...")
            self.model = DPRNNTasNet.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean").to(self.device)
            print("Pre-trained model loaded successfully.")

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Set the model to inference mode
        self.model.eval()

    def _load_weights(self, path):
        """Robust EMA weight loading mechanism"""
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

        # Clean module prefix and average counter
        if isinstance(sd, dict):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            if "n_averaged" in sd:
                sd.pop("n_averaged")

        self.model.load_state_dict(sd, strict=True)
        print("Model weights loaded successfully.")

    @torch.no_grad()
    def separate(self, mix_waveform):
        """Perform signal separation (matching Asteroid tensor shapes) :param mix_waveform: mixed-audio tensor"""
        # Asteroid DPRNN usually accepts the (Batch, Time) format directly
        # If (1, T) or (T,) is passed in, normalize it to (1, T)
        if mix_waveform.dim() == 1:
            mix_waveform = mix_waveform.unsqueeze(0)
        elif mix_waveform.dim() == 2 and mix_waveform.size(0) != 1:
            # Assume (T, 1) is passed in and is transposed to (1, T)
            mix_waveform = mix_waveform.transpose(0, 1)
        elif mix_waveform.dim() == 3:
            # Remove redundant channel dimensions (1, 1, T) -> (1, T)
            mix_waveform = mix_waveform.squeeze(1)

        mix_waveform = mix_waveform.to(self.device)
        est_sources = self.model(mix_waveform)

        # Smooth out the uncertainty in the Asteroid output dimensions:
        # If the model output is (Batch, 1, Spk, Time), squeeze the extra dimension -> (Batch, Spk, Time)
        if est_sources.dim() == 4 and est_sources.size(1) == 1:
            est_sources = est_sources.squeeze(1)

        # Length alignment
        min_len = min(est_sources.shape[-1], mix_waveform.shape[-1])
        est_sources = est_sources[..., :min_len]

        # Remove DC components (the optimization operations you do in the code will help subsequent playback or plotting)
        est_sources = est_sources - est_sources.mean(dim=-1, keepdim=True)

        return est_sources.squeeze(0).cpu()  # Return shape (2, T)