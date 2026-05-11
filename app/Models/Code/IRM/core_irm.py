import torch

class IRMInferenceCore:
    def __init__(self, n_fft=512, hop=128, device=None):
        """IRM (Ideal Ratio Mask) core class for theoretical upper bound reasoning. There is no need to load the weight file, it is only used as a baseline for extreme comparison. :param n_fft: STFT window size (512 commonly used in speech) :param hop: step size (commonly 128 or 256)"""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_fft = n_fft
        self.hop = hop

    @torch.no_grad()
    def separate(self, mix_waveform, s1_waveform, s2_waveform):
        """Calculate IRM and separate signals. Note: Unlike the blind separation model, ground truth must be provided here! :param mix_waveform: mixed signal Tensor, (1, T) or (T,) :param s1_waveform: clean source 1 Tensor, (1, T) or (T,) :param s2_waveform: clean source 2 Tensor, (1, T) or (T,) :return: est_sources, shape (2, T), located on CPU"""
        # 1. Dimensions and equipment are organized and unified into one dimension (T,) to adapt to stft batch operations.
        mix = mix_waveform.squeeze().to(self.device)
        s1 = s1_waveform.squeeze().to(self.device)
        s2 = s2_waveform.squeeze().to(self.device)

        # Check length alignment, take shortest length
        min_len = min(mix.shape[-1], s1.shape[-1], s2.shape[-1])
        mix = mix[:min_len]
        s1 = s1[:min_len]
        s2 = s2[:min_len]

        eps = 1e-8
        window = torch.hann_window(self.n_fft).to(self.device)

        # 2. Calculate STFT spectrum
        X_complex = torch.stft(mix, self.n_fft, self.hop, window=window, return_complex=True)
        X_mag = X_complex.abs()

        S1_mag = torch.stft(s1, self.n_fft, self.hop, window=window, return_complex=True).abs()
        S2_mag = torch.stft(s2, self.n_fft, self.hop, window=window, return_complex=True).abs()

        # 3. Construct IRM Mask (ideal ratio mask)
        mask1 = S1_mag / (S1_mag + S2_mag + eps)
        mask2 = S2_mag / (S1_mag + S2_mag + eps)

        # 4. Apply Mask (Oracle separation)
        S1_est_mag = X_mag * mask1
        S2_est_mag = X_mag * mask2

        # 5. iSTFT (use the phase of the original mixed signal to restore to the time domain)
        phase = torch.angle(X_complex)

        S1_est_complex = S1_est_mag * torch.exp(1j * phase)
        S2_est_complex = S2_est_mag * torch.exp(1j * phase)

        s1_recon = torch.istft(S1_est_complex, self.n_fft, self.hop, window=window, length=mix.shape[-1])
        s2_recon = torch.istft(S2_est_complex, self.n_fft, self.hop, window=window, length=mix.shape[-1])

        # 6. Remove the DC component (keep the same post-processing as the local deep learning model output)
        s1_recon = s1_recon - s1_recon.mean(dim=-1, keepdim=True)
        s2_recon = s2_recon - s2_recon.mean(dim=-1, keepdim=True)

        # 7. Splice and return (2, T), transfer to CPU for UI plotting
        est_sources = torch.stack((s1_recon, s2_recon), dim=0)
        return est_sources.cpu()