import sys
import os
import torch
import torchaudio
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from app.Models.Code.ModelManager import SeparationManager

def process_long_audio_by_chunks(manager, model_choice, waveform, sr, chunk_sec=4.0):
    chunk_samples = int(chunk_sec * sr)
    total_length = waveform.shape[-1]
    out_chunks = []

    for start in range(0, total_length, chunk_samples):
        end = min(start + chunk_samples, total_length)
        chunk = waveform[:, start:end]
        actual_len = chunk.shape[1]

        if actual_len < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - actual_len))

        chunk_est = manager.active_model.separate(chunk.float())
        chunk_est = chunk_est[:, :actual_len]
        out_chunks.append(chunk_est)

    est_sources = torch.cat(out_chunks, dim=1)

    for i in range(est_sources.shape[0]):
        est_sources[i] = torch.clamp(est_sources[i], min=-5.0, max=5.0)
        rms_mix = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
        rms_est = torch.sqrt(torch.mean(est_sources[i] ** 2) + 1e-8)
        est_sources[i] = est_sources[i] * (rms_mix / rms_est)
        est_sources[i] = torch.clamp(est_sources[i], -0.99, 0.99)

    return est_sources


class AudioWorker(QThread):
    finished_signal = pyqtSignal(list, str)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, manager, model_choice, mix_path):
        super().__init__()
        self.manager = manager
        self.model_choice = model_choice
        self.mix_path = mix_path

    def run(self):
        try:
            self.log_signal.emit("[Audio Mode] Loading model & executing 4s chunk inference...")
            self.manager.load_model(self.model_choice)

            mix_wav, sr = torchaudio.load(self.mix_path)
            if mix_wav.shape[0] > 1:
                mix_wav = mix_wav.mean(dim=0, keepdim=True)

            if sr != 8000:
                self.log_signal.emit(f"Auto-resampling: {sr}Hz -> 8000Hz")
                mix_wav = torchaudio.functional.resample(mix_wav, sr, 8000)
                sr = 8000

            est_sources = process_long_audio_by_chunks(self.manager, self.model_choice, mix_wav, sr)

            out_dir = os.path.join(os.getcwd(), "GUI_Outputs", "Audio")
            os.makedirs(out_dir, exist_ok=True)
            out_paths = []

            for i in range(est_sources.shape[0]):
                out_path = os.path.join(out_dir, f"est_source_{i + 1}.wav")
                torchaudio.save(out_path, est_sources[i].unsqueeze(0).float(), sr)
                out_paths.append(out_path)

            self.log_signal.emit("Audio inference & reconstruction successful!")
            self.finished_signal.emit(out_paths, self.mix_path)
        except Exception as e:
            self.error_signal.emit(str(e))


class VideoWorker(QThread):
    finished_signal = pyqtSignal(list, str)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, manager, model_choice, video_path):
        super().__init__()
        self.manager = manager
        self.model_choice = model_choice
        self.video_path = video_path

    def run(self):
        try:
            import moviepy.editor as mp
            self.log_signal.emit("[Video Mode] Extracting original audio track...")
            out_dir = os.path.join(os.getcwd(), "GUI_Outputs", "Video")
            os.makedirs(out_dir, exist_ok=True)

            video_clip = mp.VideoFileClip(self.video_path)
            temp_audio_path = os.path.join(out_dir, "temp_extracted.wav")
            video_clip.audio.write_audiofile(temp_audio_path, fps=8000, logger=None)

            self.log_signal.emit("Starting smart chunked blind separation...")
            self.manager.load_model(self.model_choice)
            mix_wav, sr = torchaudio.load(temp_audio_path)
            if mix_wav.shape[0] > 1: mix_wav = mix_wav.mean(dim=0, keepdim=True)

            est_sources = process_long_audio_by_chunks(self.manager, self.model_choice, mix_wav, sr)

            num_spk = est_sources.shape[0]
            self.log_signal.emit(f"Separation complete! Encoding {num_spk} independent video tracks...")
            out_vid_paths = []

            for i in range(num_spk):
                temp_sep_wav = os.path.join(out_dir, f"temp_sep_{i + 1}.wav")
                torchaudio.save(temp_sep_wav, est_sources[i].unsqueeze(0).float(), sr)

                new_audio_clip = mp.AudioFileClip(temp_sep_wav)
                new_video_clip = video_clip.set_audio(new_audio_clip)

                out_vid_path = os.path.join(out_dir, f"Separated_Source_{i + 1}.mp4")
                self.log_signal.emit(f"Encoding in progress: Separated_Source_{i + 1}.mp4")

                new_video_clip.write_videofile(
                    out_vid_path, codec="libx264", audio_codec="aac",
                    preset="ultrafast", ffmpeg_params=["-pix_fmt", "yuv420p"], logger=None
                )
                out_vid_paths.append(out_vid_path)

            video_clip.close()
            self.log_signal.emit("Video processing fully complete!")
            self.finished_signal.emit(out_vid_paths, self.video_path)
        except Exception as e:
            self.error_signal.emit(str(e))

class CommWorker(QThread):
    finished_signal = pyqtSignal(list, str)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, manager, model_choice, mix_path):
        super().__init__()
        self.manager = manager
        self.model_choice = model_choice
        self.mix_path = mix_path

    def run(self):
        try:
            self.log_signal.emit("[Comm Mode] Loading model & executing baseband inference...")

            est_sources, _ = self.manager.separate_audio(
                mix_path=self.mix_path,
                model_choice=self.model_choice
            )

            out_dir = os.path.join(os.getcwd(), "GUI_Outputs", "Comm")
            os.makedirs(out_dir, exist_ok=True)
            out_paths = []

            for i in range(est_sources.shape[0]):
                out_path = os.path.join(out_dir, f"est_signal_{i + 1}.npy")
                np.save(out_path, est_sources[i].numpy())
                out_paths.append(out_path)

            self.log_signal.emit(f"Comm signal separated! Saved to {out_dir}")
            self.finished_signal.emit(out_paths, self.mix_path)
        except Exception as e:
            self.error_signal.emit(str(e))

class SeparationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Source Signal Separation Workstation (Audio/Video/Comm)")
        self.resize(1500, 950)  # Implementation note.
        self.manager = SeparationManager()
        self.audio_player = QMediaPlayer()

        self.initUI()
        self.apply_dark_theme()

    def initUI(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(self.create_audio_tab(), " Audio Separation ")
        self.tabs.addTab(self.create_video_tab(), "  Video Vocal Extraction  ")
        self.tabs.addTab(self.create_comm_tab(), " Comm Signal Parsing ")

    def create_audio_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)

        left = QVBoxLayout()
        left.addWidget(QLabel("1. Select Speech Separation Model:"))
        self.a_model_cb = QComboBox()
        speech_models = [
            "[Speech] TasNet (Baseline)",
            "[Speech] Deep TasNet",
            "[Speech] MultiScale-Deep (Ultimate)",
            "[Speech] TasNet (3 Speakers)",
            "[Speech] TasNet (4 Speakers)",
            "[Speech] TF-BLSTM (TF-Domain Baseline)",
            "[Speech] Asteroid DPRNN (Pre-trained)",
            "[Speech] SpeechBrain SepFormer (SOTA)"
        ]
        self.a_model_cb.addItems(speech_models)
        self.a_model_cb.setCurrentText("[Speech] MultiScale-Deep (Ultimate)")
        left.addWidget(self.a_model_cb)

        left.addSpacing(20)
        left.addWidget(QLabel("2. Upload Mixed Audio (.wav):"))
        self.a_mix_btn = QPushButton("📂 Browse Audio File")
        self.a_mix_btn.clicked.connect(lambda: self.select_file("audio"))
        self.a_mix_label = QLabel("No file selected")
        self.a_mix_label.setStyleSheet("color: #888888; font-style: italic;")
        left.addWidget(self.a_mix_btn)
        left.addWidget(self.a_mix_label)

        left.addStretch()
        self.a_run_btn = QPushButton("Execute Chunked Separation")
        self.a_run_btn.setMinimumHeight(75)  # Implementation note.
        self.a_run_btn.setObjectName("runButtonAudio")
        self.a_run_btn.clicked.connect(self.run_audio)
        left.addWidget(self.a_run_btn)

        left.addSpacing(20)
        left.addWidget(QLabel("Execution Logs:"))
        self.a_log_box = QTextEdit()
        self.a_log_box.setReadOnly(True)
        self.a_log_box.setMaximumHeight(220)
        left.addWidget(self.a_log_box)

        right = QVBoxLayout()
        self.a_figure = Figure(figsize=(7, 6), dpi=100)
        self.a_figure.patch.set_facecolor('#1E1E1E')
        self.a_canvas = FigureCanvas(self.a_figure)
        right.addWidget(self.a_canvas, stretch=5)

        self.a_play_layout = QHBoxLayout()
        self.a_stop_btn = QPushButton("Stop Playback")
        self.a_stop_btn.setMinimumHeight(55)
        self.a_stop_btn.setObjectName("stopButton")
        self.a_stop_btn.clicked.connect(self.audio_player.stop)
        self.a_play_layout.addWidget(self.a_stop_btn)
        right.addLayout(self.a_play_layout, stretch=1)

        layout.addLayout(left, 3)
        layout.addLayout(right, 7)
        self.a_mix_path = ""
        return widget

    def create_video_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)

        left = QVBoxLayout()
        left.addWidget(QLabel("1. Select Speech Separation Model:"))
        self.v_model_cb = QComboBox()
        self.v_model_cb.addItems([
            "[Speech] MultiScale-Deep (Ultimate)",
            "[Speech] Deep TasNet",
            "[Speech] SpeechBrain SepFormer (SOTA)"
        ])
        left.addWidget(self.v_model_cb)

        left.addSpacing(20)
        left.addWidget(QLabel("2. Upload Video with Vocals (.mp4):"))
        self.v_mix_btn = QPushButton("📂 Browse Video File")
        self.v_mix_btn.clicked.connect(lambda: self.select_file("video"))
        self.v_mix_label = QLabel("No file selected")
        self.v_mix_label.setStyleSheet("color: #888888; font-style: italic;")
        left.addWidget(self.v_mix_btn)
        left.addWidget(self.v_mix_label)

        left.addStretch()
        self.v_run_btn = QPushButton("Start Video Audio Separation")
        self.v_run_btn.setMinimumHeight(75)  # Implementation note.
        self.v_run_btn.setObjectName("runButtonVideo")
        self.v_run_btn.clicked.connect(self.run_video)
        left.addWidget(self.v_run_btn)

        left.addSpacing(20)
        left.addWidget(QLabel("Execution Logs:"))
        self.v_log_box = QTextEdit()
        self.v_log_box.setReadOnly(True)
        self.v_log_box.setMaximumHeight(220)
        left.addWidget(self.v_log_box)

        right = QVBoxLayout()
        self.v_grid = QGridLayout()
        self.v_grid.setSpacing(15)
        right.addLayout(self.v_grid)
        right.addStretch()

        layout.addLayout(left, 3)
        layout.addLayout(right, 7)
        self.v_mix_path = ""
        return widget

    def create_comm_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)

        left = QVBoxLayout()
        left.addWidget(QLabel("1. Select Comm Separation Model:"))
        self.c_model_cb = QComboBox()
        comm_models = [
            "[Comm] TasNet (Baseline)",
            "[Comm] Deep TasNet",
            "[Comm] MultiScale TasNet",
            "[Comm] TF-BLSTM (TF-Domain Baseline)",
            "[Comm] Asteroid DPRNN (Long Seq)",
            "[Comm] SepFormer / DPTNet"
        ]
        self.c_model_cb.addItems(comm_models)
        self.c_model_cb.setCurrentText("[Comm] MultiScale TasNet")
        left.addWidget(self.c_model_cb)

        left.addSpacing(20)
        left.addWidget(QLabel("2. Upload Mixed Baseband Signal (.npy):"))
        self.c_mix_btn = QPushButton("📂 Browse Signal File")
        self.c_mix_btn.clicked.connect(lambda: self.select_file("comm"))
        self.c_mix_label = QLabel("No file selected")
        self.c_mix_label.setStyleSheet("color: #888888; font-style: italic;")
        left.addWidget(self.c_mix_btn)
        left.addWidget(self.c_mix_label)

        left.addStretch()
        self.c_run_btn = QPushButton("Parse Comm Signal")
        self.c_run_btn.setMinimumHeight(75)  # Implementation note.
        self.c_run_btn.setObjectName("runButtonComm")
        self.c_run_btn.clicked.connect(self.run_comm)
        left.addWidget(self.c_run_btn)

        left.addSpacing(20)
        left.addWidget(QLabel("Execution Logs:"))
        self.c_log_box = QTextEdit()
        self.c_log_box.setReadOnly(True)
        self.c_log_box.setMaximumHeight(220)
        left.addWidget(self.c_log_box)

        right = QVBoxLayout()
        self.c_figure = Figure(figsize=(7, 6), dpi=100)
        self.c_figure.patch.set_facecolor('#1E1E1E')
        self.c_canvas = FigureCanvas(self.c_figure)
        right.addWidget(self.c_canvas)

        layout.addLayout(left, 3)
        layout.addLayout(right, 7)
        self.c_mix_path = ""
        return widget

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QWidget { font-family: 'Segoe UI', 'Helvetica Neue', 'Arial'; font-size: 18px; color: #E0E0E0; }

            QTabWidget::pane { border: 1px solid #3A3A3A; background: #1E1E1E; border-radius: 8px; top: -1px; }
            QTabBar::tab { background: #2D2D30; color: #9E9E9E; padding: 18px 40px; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 4px; font-weight: bold; font-size: 18px;}
            QTabBar::tab:selected { background: #007ACC; color: white; border-bottom: 3px solid #007ACC;}
            QTabBar::tab:hover:!selected { background: #3E3E42; color: white; }

            QLabel { color: #CCCCCC; font-weight: 600; font-size: 18px;}

            QComboBox { background-color: #252526; border: 1px solid #444; border-radius: 6px; padding: 12px; font-weight: 500; font-size: 18px;}
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { background-color: #252526; selection-background-color: #007ACC; border: 1px solid #444; font-size: 18px; padding: 5px;}

            QPushButton { background-color: #333337; color: white; border: 1px solid #555; border-radius: 6px; padding: 12px 24px; font-weight: bold; font-size: 18px;}
            QPushButton:hover { background-color: #3E3E42; border: 1px solid #666; }
            QPushButton:pressed { background-color: #1E1E1E; }

            QPushButton#runButtonAudio { background-color: #28A745; border: none; font-size: 20px; padding: 18px;}
            QPushButton#runButtonAudio:hover { background-color: #218838; }

            QPushButton#runButtonVideo { background-color: #007ACC; border: none; font-size: 20px; padding: 18px;}
            QPushButton#runButtonVideo:hover { background-color: #005A9E; }

            QPushButton#runButtonComm { background-color: #6F42C1; border: none; font-size: 20px; padding: 18px;}
            QPushButton#runButtonComm:hover { background-color: #5A32A3; }

            QPushButton#stopButton { background-color: #DC3545; border: none; font-size: 18px;}
            QPushButton#stopButton:hover { background-color: #C82333; }

            QTextEdit { background-color: #151515; color: #4AF626; font-family: 'Consolas', 'Courier New'; font-size: 16px; border: 1px solid #333; border-radius: 6px; padding: 15px; }

            QGroupBox { border: 1px solid #444; border-radius: 8px; margin-top: 2.5ex; padding-top: 30px; font-weight: bold; color: #A9B7C6; font-size: 18px;}
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; }
        """)

    def select_file(self, mode):
        if mode == "audio":
            path, _ = QFileDialog.getOpenFileName(self, "Select Mixed Audio", "", "Audio Files (*.wav)")
            if path:
                self.a_mix_path = path
                self.a_mix_label.setText(os.path.basename(path))
        elif mode == "video":
            path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
            if path:
                self.v_mix_path = path
                self.v_mix_label.setText(os.path.basename(path))
        elif mode == "comm":
            path, _ = QFileDialog.getOpenFileName(self, "Select Comm Baseband Signal", "", "Numpy Array (*.npy)")
            if path:
                self.c_mix_path = path
                self.c_mix_label.setText(os.path.basename(path))

    def on_error(self, err_msg, btn):
        QMessageBox.critical(self, "Execution Failed", f"System Error:\n{err_msg}")
        btn.setEnabled(True)

    def run_audio(self):
        if not self.a_mix_path: return QMessageBox.warning(self, "Error", "Please select an audio file first!")
        self.a_run_btn.setEnabled(False)
        self.a_log_box.append("-" * 35)
        self.a_worker = AudioWorker(self.manager, self.a_model_cb.currentText(), self.a_mix_path)
        self.a_worker.log_signal.connect(self.a_log_box.append)
        self.a_worker.error_signal.connect(lambda e: self.on_error(e, self.a_run_btn))
        self.a_worker.finished_signal.connect(self.on_audio_finished)
        self.a_worker.start()

    def on_audio_finished(self, out_paths, mix_path):
        self.a_run_btn.setEnabled(True)
        self.plot_waveforms(mix_path, out_paths)
        self.setup_audio_players(mix_path, out_paths)

    def plot_waveforms(self, mix_path, sep_paths):
        self.a_figure.clear()
        num_plots = len(sep_paths) + 1

        ax_mix = self.a_figure.add_subplot(num_plots, 1, 1)
        mix_wav, _ = torchaudio.load(mix_path)
        ax_mix.plot(mix_wav[0].numpy(), color='#888888', linewidth=0.5)
        ax_mix.set_title("[Original] Mixed Signal", color='white', fontsize=16, fontweight='bold')
        ax_mix.set_facecolor('#1E1E1E')
        ax_mix.tick_params(colors='white', bottom=False, labelbottom=False)

        for i, path in enumerate(sep_paths):
            wav, _ = torchaudio.load(path)
            ax = self.a_figure.add_subplot(num_plots, 1, i + 2)
            ax.plot(wav[0].numpy(), color='#00E676', linewidth=0.5)  # Neon Green
            ax.set_title(f"[Separated] Source {i + 1}", color='white', fontsize=16, fontweight='bold')
            ax.set_facecolor('#1E1E1E')
            ax.tick_params(colors='white', bottom=False, labelbottom=False)

        self.a_figure.tight_layout(pad=2.5)
        self.a_canvas.draw()

    def setup_audio_players(self, mix_path, sep_paths):
        for i in reversed(range(1, self.a_play_layout.count())):
            self.a_play_layout.itemAt(i).widget().setParent(None)

        btn_mix = QPushButton("Play Original Mix")
        btn_mix.setMinimumHeight(55)
        btn_mix.clicked.connect(lambda: self.play_media(mix_path))
        self.a_play_layout.addWidget(btn_mix)

        for i, path in enumerate(sep_paths):
            btn = QPushButton(f"Play Source {i + 1}")
            btn.setMinimumHeight(55)
            btn.setStyleSheet("color: #00E676; font-weight: bold; border-color: #00E676;")
            btn.clicked.connect(lambda checked, p=path: self.play_media(p))
            self.a_play_layout.addWidget(btn)

    def play_media(self, path):
        self.audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.audio_player.play()

    def run_video(self):
        if not self.v_mix_path: return QMessageBox.warning(self, "Error", "Please select a video file first!")
        self.v_run_btn.setEnabled(False)
        self.v_log_box.append("-" * 35)
        self.v_worker = VideoWorker(self.manager, self.v_model_cb.currentText(), self.v_mix_path)
        self.v_worker.log_signal.connect(self.v_log_box.append)
        self.v_worker.error_signal.connect(lambda e: self.on_error(e, self.v_run_btn))
        self.v_worker.finished_signal.connect(self.on_video_finished)
        self.v_worker.start()

    def on_video_finished(self, out_vids, mix_vid):
        self.v_run_btn.setEnabled(True)

        for i in reversed(range(self.v_grid.count())):
            widget = self.v_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        all_vids = [mix_vid] + out_vids
        titles = ["Original Video Mix"] + [f"Separated Vocal Track {i + 1}" for i in range(len(out_vids))]

        for i, (path, title) in enumerate(zip(all_vids, titles)):
            row, col = i // 2, i % 2

            box = QGroupBox(title)
            vbox = QVBoxLayout()
            vbox.addStretch()

            abs_path = os.path.abspath(path).replace('\\', '/')
            btn = QPushButton("Open in System Player")
            btn.setMinimumHeight(70)
            btn.setStyleSheet(
                "font-size: 20px; font-weight: bold; color: white; background-color: #007ACC; border-radius: 8px;")
            btn.clicked.connect(lambda checked, p=abs_path: os.startfile(p))

            vbox.addWidget(btn)
            vbox.addStretch()
            box.setLayout(vbox)
            self.v_grid.addWidget(box, row, col)

    def run_comm(self):
        if not self.c_mix_path: return QMessageBox.warning(self, "Error",
                                                           "Please select a Numpy (.npy) signal file first!")
        self.c_run_btn.setEnabled(False)
        self.c_log_box.append("-" * 35)
        self.c_worker = CommWorker(self.manager, self.c_model_cb.currentText(), self.c_mix_path)
        self.c_worker.log_signal.connect(self.c_log_box.append)
        self.c_worker.error_signal.connect(lambda e: self.on_error(e, self.c_run_btn))
        self.c_worker.finished_signal.connect(self.on_comm_finished)
        self.c_worker.start()

    def on_comm_finished(self, out_paths, mix_path):
        self.c_run_btn.setEnabled(True)
        self.plot_comm_waveforms(mix_path, out_paths)

    def plot_comm_waveforms(self, mix_path, sep_paths):
        self.c_figure.clear()
        num_plots = len(sep_paths) + 1

        ax_mix = self.c_figure.add_subplot(num_plots, 1, 1)
        mix_data = np.load(mix_path)
        if mix_data.ndim > 1: mix_data = mix_data.flatten()

        ax_mix.plot(mix_data, color='#888888', linewidth=0.5)
        ax_mix.set_title("[Original] Mixed Baseband Signal", color='white', fontsize=16, fontweight='bold')
        ax_mix.set_facecolor('#1E1E1E')
        ax_mix.tick_params(colors='white', bottom=False, labelbottom=False)

        for i, path in enumerate(sep_paths):
            data = np.load(path)
            if data.ndim > 1: data = data.flatten()

            ax = self.c_figure.add_subplot(num_plots, 1, i + 2)
            ax.plot(data, color='#00D2FF', linewidth=0.5)  # Cyan Blue for Comm signals
            ax.set_title(f"[Separated] Carrier Signal {i + 1}", color='white', fontsize=16, fontweight='bold')
            ax.set_facecolor('#1E1E1E')
            ax.tick_params(colors='white', bottom=False, labelbottom=False)

        self.c_figure.tight_layout(pad=2.5)
        self.c_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SeparationApp()
    window.show()
    sys.exit(app.exec_())