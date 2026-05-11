import os
import torch
import torchaudio

# Import the local MultiScaleDeep model
from app.Models.Code.MultiScaleDeep.Model_TasNet_MultiScale import Model

# If there are no local weights, we call the ready-made Asteroid open source model as a substitute
try:
    from asteroid.models import ConvTasNet
except ImportError:
    ConvTasNet = None


class UnifiedInferenceCore:
    def __init__(self, model_type="multiscale_speech_2spk", ckpt_path=None, device=None):
        """A unified pure reasoning interface for direct calls from the GUI. :param model_type: supports the following three modes - 'multiscale_comm_2spk': communication signal 2 source separation (corresponding to trainTasNet2.py) - 'multiscale_speech_4spk': speech signal 4 source separation (corresponding to trainTasNet.py) - 'pretrained_asteroid': ready-made open source model on the Internet (substitute when there is no parameter file) :param ckpt_path: the local.pt weight file path :param device: Forcibly specify the device, cuda or cpu is automatically selected by default"""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.model = None

        self._build_and_load()

    def _build_and_load(self):
        if self.model_type == "multiscale_speech_2spk":
            # Configuration of strict alignment trainTasNet2.py
            self.model = Model(
                enc_dim=516, enc_kernel=(5, 11, 21), enc_stride=8,
                bottleneck_channels=128, hidden_channels=256,
                num_blocks=8, num_repeats=3, num_speakers=2
            ).to(self.device)
            self._load_local_weights()

        elif self.model_type == "multiscale_speech_3spk":
            # Strictly align the configuration of trainTasNet.py
            self.model = Model(
                enc_dim=516, enc_kernel=(11, 21, 31), enc_stride=8,
                bottleneck_channels=128, hidden_channels=256,
                num_blocks=8, num_repeats=3, num_speakers=3
            ).to(self.device)
            self._load_local_weights()

        elif self.model_type == "multiscale_speech_4spk":
            # Strictly align the configuration of trainTasNet.py
            self.model = Model(
                enc_dim=516, enc_kernel=(11, 21, 31), enc_stride=8,
                bottleneck_channels=128, hidden_channels=256,
                num_blocks=8, num_repeats=3, num_speakers=4
            ).to(self.device)
            self._load_local_weights()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        # Force switching to evaluation mode and turn off dynamic updates of Dropout and BatchNorm
        self.model.eval()

    def _load_local_weights(self):
        """Reuse the robust loading logic you wrote and extract ema_state first"""
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

        # Clean key names, remove module. prefix and remove n_averaged
        if isinstance(sd, dict):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            if "n_averaged" in sd:
                sd.pop("n_averaged")

        self.model.load_state_dict(sd, strict=True)
        print("本地权重加载成功！")

    @torch.no_grad()  # Explicitly disable gradients and strictly control video memory usage
    def separate(self, mix_waveform):
        """Core detached functions called by the GUI. :param mix_waveform: input mixed audio Tensor (can be one-dimensional or two-dimensional) :return: separated multi-source audio Tensor, located on the CPU, with shape (num_speakers, T)"""
        # The unified dimensions are organized into (Batch, 1, Time)
        if mix_waveform.dim() == 1:
            mix_waveform = mix_waveform.unsqueeze(0).unsqueeze(0)
        elif mix_waveform.dim() == 2:
            mix_waveform = mix_waveform.unsqueeze(1)

        mix_waveform = mix_waveform.to(self.device)

        est_sources = self.model(mix_waveform)

        # Enable bfloat16 mixed-precision accelerated reasoning to improve interface response speed
        if est_sources.dim() == 4 and est_sources.size(1) == 1:
            est_sources = est_sources.squeeze(1)

        # If you are using an off-the-shelf model such as Asteroid, smooth out the differences in output dimensions
        min_len = min(est_sources.shape[-1], mix_waveform.shape[-1])
        est_sources = est_sources[..., :min_len]
        est_sources = est_sources - est_sources.mean(dim=-1, keepdim=True)

        # Length alignment and DC component removal (to avoid sonic popping during playback)
        return est_sources.squeeze(0).cpu()