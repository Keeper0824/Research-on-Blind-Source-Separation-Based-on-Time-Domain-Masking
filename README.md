# Blind Source Separation with Time-Domain Masking

This project studies how the encoder and decoder affect single-channel blind source separation. It compares three variants of a Conv-TasNet-style baseline on two-speaker speech mixtures and synthetic communication signals. A PyQt5 desktop demo runs the models on audio files, video soundtracks, and NumPy baseband signals.

**Haoran Sun (孙浩然)** · 2025–26 undergraduate final-year project in Internet of Things Engineering · Beijing University of Posts and Telecommunications (BUPT) and Queen Mary University of London (QMUL) Joint Programme / Joint Education Institute (JP/JEI).

## Model design

```text
Mixture waveform → Encoder → Latent features → Source masks predicted by a TCN
Latent features × each source mask → Decoder → Estimated source waveform
```

The encoder learns a time-domain representation of the mixture. A temporal convolutional network (TCN) predicts one mask per source. The decoder maps the masked features back to waveforms. The main encoder/decoder comparison uses a common TCN design to study the effect of depth and temporal scale.

| Model | Encoder and decoder design |
| --- | --- |
| Baseline | Conv-TasNet-style learned waveform encoder and decoder |
| Deep | Deeper convolutional blocks with GroupNorm and PReLU |
| MultiScale | Parallel branches with different kernel sizes for short-, medium-, and long-range features |
| MultiScale-Deep | Deeper parallel branches with a refinement stage after feature fusion |

Training uses permutation-invariant training (PIT) with a scale-invariant signal-to-noise ratio (SI-SNR) objective to handle the arbitrary ordering of estimated sources. The speech ablations compare encoder width, convolution kernel sizes, encoder stride, and refinement depth.

## Desktop demo

Select a tab and model, browse for an input file, and start inference. Each tab shows progress in a log and saves the results automatically.

| Tab | Input | Run button | Output |
| --- | --- | --- | --- |
| **Audio Separation** | `.wav` speech mixture | **Execute Chunked Separation** | Waveform plots, playback controls, and `GUI_Outputs/Audio/est_source_1.wav`, `est_source_2.wav`, etc. |
| **Video Vocal Extraction** | `.mp4`, `.avi`, or `.mkv` with an audio track | **Start Video Audio Separation** | One video per estimated source, using the original video frames and a separated soundtrack. Files are saved as `GUI_Outputs/Video/Separated_Source_1.mp4`, `Separated_Source_2.mp4`, etc. |
| **Comm Signal Parsing** | `.npy` baseband mixture in the format expected by the selected model | **Parse Comm Signal** | Waveform plots and `GUI_Outputs/Comm/est_signal_1.npy`, `est_signal_2.npy`, etc. |

### Quick start

1. Open the desktop application with the model weights required by the selected backend.
2. In **Audio Separation**, select **[Speech] MultiScale-Deep (Ultimate)**.
3. Click **Browse Audio File** and choose a short `.wav` mixture with two overlapping speakers.
4. Click **Execute Chunked Separation** and wait for completion in the log.
5. Compare the waveform plots, then use **Play Original Mix** and **Play Source 1** / **Play Source 2** to listen.

The audio pipeline averages multichannel input to mono and resamples it to 8 kHz. For long recordings, it processes four-second chunks, zero-pads a short final chunk, and joins the separated chunks in order. Video mode extracts the audio track and runs it through the speech pipeline before writing the result back into video files.

The project uses Python, PyTorch, torchaudio, PyQt5, NumPy, Matplotlib, and MoviePy. The experimental environment also used SpeechBrain for benchmark model evaluation.

## Experimental results

The numbers below are from the final-year project report. SI-SNR measures waveform reconstruction quality. PESQ and STOI measure speech quality and intelligibility, respectively. Bit error rate (BER) measures how many bits are recovered incorrectly after demodulation.

### Speech separation on WSJ0-2mix

The speech experiments used 8 kHz two-speaker mixtures. The training, validation, and test sets contained 20,000, 5,000, and 3,000 mixtures, respectively.

| Model | Parameters | SI-SNR (dB) ↑ | PESQ ↑ | STOI ↑ |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2.63M | 14.74 | 3.021 | 0.9531 |
| Deep | 7.36M | 15.61 | 3.186 | 0.9584 |
| MultiScale | 2.96M | 15.91 | 3.259 | 0.9624 |
| MultiScale-Deep | 16.11M | **16.45** | **3.341** | **0.9633** |

MultiScale-Deep gained **1.72 dB SI-SNR** over the project's baseline. MultiScale gained **1.18 dB** with a smaller increase in parameter count. In the report's broader comparison, DPRNN and SepFormer reached 19.21 and 22.19 dB SI-SNR, respectively. (Final report, Tables 5 and 11.)

### Separation of synthetic communication signals

The `comm_mix` dataset contains mixtures generated from BPSK, QPSK, and 16-QAM source signals, with simulated carrier-frequency offsets, phase offsets, and additive white Gaussian noise.

| Model | Parameters | SI-SNR (dB) ↑ | BER ↓ |
| --- | ---: | ---: | ---: |
| Baseline | 2.63M | 16.43 | 7.37% |
| Deep | 7.36M | 15.72 | 8.14% |
| MultiScale | 2.96M | **18.30** | **6.87%** |
| MultiScale-Deep | 21.82M | 12.12 | 32.78% |

MultiScale had the lowest BER of the four variants. The combined model performed best among the four on speech. In a separate controlled test, MultiScale reached **0.9264% BER** on two BPSK sources at 4 and 16 samples per symbol, with a simulated channel SNR of 12–18 dB. This result is specific to that setup. Evaluation under more realistic communication channels and noise conditions is future work. (Final report, Table 10 and §4.3.)

## References

- Haoran Sun, *Research on Blind Source Separation Based on Time-Domain Masking*, undergraduate final-year project report, BUPT–QMUL, 2026.
- Yi Luo and Nima Mesgarani, *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*, IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2019.

## Licensing

Section 4.4 of the *JP/JEI Project Student Handbook 2025–26* assigns project work and its associated intellectual property to the JP/JEI, with an exception for industry projects where different arrangements were agreed in advance. Use the code, model weights, and datasets under the terms granted by their respective rights holders.
