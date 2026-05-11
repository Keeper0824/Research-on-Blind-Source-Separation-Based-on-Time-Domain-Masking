import os
import torch
import torchaudio
from app.Models.Code.Baseline.Model_TasNet import Model


class SignalSeparator:
    def __init__(self, model_type="custom_tasnet", ckpt_path=None, device=None):
        """Inference core used by the GUI.

        Args:
            model_type: Model type, such as 'custom_tasnet' or 'pretrained_xxx'.
            ckpt_path: Path to custom model weights.
            device: Runtime device, either 'cuda' or 'cpu'.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.model = None

        self._load_model()

    def _load_model(self):
        if self.model_type == "custom_tasnet":
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

        self.model.eval()  # Switch to inference mode and freeze Dropout/BatchNorm behavior.

    def _load_weights(self, path):
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

        if isinstance(sd, dict):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            if "n_averaged" in sd:
                sd.pop("n_averaged")

        self.model.load_state_dict(sd, strict=True)
        print("Model weights loaded successfully.")

    @torch.no_grad()
    def separate(self, mix_waveform):
        """ :param mix_waveform: Tensor of shape (1, T) or (T,) :return: separated_waveforms: Tensor of shape (num_speakers, T) """
        if mix_waveform.dim() == 1:
            mix_waveform = mix_waveform.unsqueeze(0).unsqueeze(0)
        elif mix_waveform.dim() == 2:
            mix_waveform = mix_waveform.unsqueeze(1)

        mix_waveform = mix_waveform.to(self.device)

        est_sources = self.model(mix_waveform)  # Output shape: (1, 2, T_est).

        min_len = min(est_sources.shape[-1], mix_waveform.shape[-1])
        est_sources = est_sources[..., :min_len]

        return est_sources.squeeze(0).cpu()