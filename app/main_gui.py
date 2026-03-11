import sys
import os
import torch
import torchaudio
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 导入终极总控台
from app.Models.Code.ModelManager import SeparationManager


# ==========================================
# 1. 核心算法：长音频切片分离 (Chunking)
# ==========================================
def process_long_audio_by_chunks(manager, model_choice, waveform, sr, chunk_sec=4.0):
    """把无限长的音频切成4秒一块，防止显存爆炸，并填补长度"""
    chunk_samples = int(chunk_sec * sr)
    total_length = waveform.shape[-1]
    out_chunks = []

    for start in range(0, total_length, chunk_samples):
        end = min(start + chunk_samples, total_length)
        chunk = waveform[:, start:end]
        actual_len = chunk.shape[1]

        # 如果不足4秒，末尾补零 (Padding)
        if actual_len < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - actual_len))

        # 强制全精度盲分离
        chunk_est = manager.active_model.separate(chunk.float())

        # 裁掉之前补零的多余部分
        chunk_est = chunk_est[:, :actual_len]
        out_chunks.append(chunk_est)

    # 沿时间轴无缝拼接
    est_sources = torch.cat(out_chunks, dim=1)

    # RMS 能量音量恢复 (防止由于切片和模型特性导致声音过小)
    for i in range(est_sources.shape[0]):
        est_sources[i] = torch.clamp(est_sources[i], min=-5.0, max=5.0)
        rms_mix = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
        rms_est = torch.sqrt(torch.mean(est_sources[i] ** 2) + 1e-8)
        est_sources[i] = est_sources[i] * (rms_mix / rms_est)
        est_sources[i] = torch.clamp(est_sources[i], -0.99, 0.99)

    return est_sources


# ==========================================
# 2. 线程：处理纯音频
# ==========================================
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
            self.log_signal.emit("🚀 [音频模式] 正在加载模型并执行4秒分块推理...")
            self.manager.load_model(self.model_choice)

            mix_wav, sr = torchaudio.load(self.mix_path)
            if mix_wav.shape[0] > 1:
                mix_wav = mix_wav.mean(dim=0, keepdim=True)

            # 如果不是8000Hz，统一重采样为8000Hz以适配模型
            if sr != 8000:
                self.log_signal.emit(f"🔄 自动重采样: {sr}Hz -> 8000Hz")
                mix_wav = torchaudio.functional.resample(mix_wav, sr, 8000)
                sr = 8000

            # 调用智能切片算法
            est_sources = process_long_audio_by_chunks(self.manager, self.model_choice, mix_wav, sr)

            out_dir = os.path.join(os.getcwd(), "GUI_Outputs", "Audio")
            os.makedirs(out_dir, exist_ok=True)
            out_paths = []

            for i in range(est_sources.shape[0]):
                out_path = os.path.join(out_dir, f"est_source_{i + 1}.wav")
                torchaudio.save(out_path, est_sources[i].unsqueeze(0).float(), sr)
                out_paths.append(out_path)

            self.log_signal.emit("✅ 音频推理与重组成功！")
            self.finished_signal.emit(out_paths, self.mix_path)
        except Exception as e:
            self.error_signal.emit(str(e))


# ==========================================
# 3. 线程：处理视频 (音轨抽离 -> 分离 -> 视频压制)
# ==========================================
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
            self.log_signal.emit("🎬 [视频模式] 正在提取视频原声音轨...")
            out_dir = os.path.join(os.getcwd(), "GUI_Outputs", "Video")
            os.makedirs(out_dir, exist_ok=True)

            # 1. 加载视频并抽取音频
            video_clip = mp.VideoFileClip(self.video_path)
            temp_audio_path = os.path.join(out_dir, "temp_extracted.wav")
            video_clip.audio.write_audiofile(temp_audio_path, fps=8000, logger=None)

            # 2. 读取并推理
            self.log_signal.emit("🚀 开始执行智能切片盲分离...")
            self.manager.load_model(self.model_choice)
            mix_wav, sr = torchaudio.load(temp_audio_path)
            if mix_wav.shape[0] > 1: mix_wav = mix_wav.mean(dim=0, keepdim=True)

            est_sources = process_long_audio_by_chunks(self.manager, self.model_choice, mix_wav, sr)

            # 3. 压制新视频
            num_spk = est_sources.shape[0]
            self.log_signal.emit(f"✂️ 分离完成！正在压制 {num_spk} 份新的独立干音视频...")
            out_vid_paths = []

            for i in range(num_spk):
                temp_sep_wav = os.path.join(out_dir, f"temp_sep_{i + 1}.wav")
                torchaudio.save(temp_sep_wav, est_sources[i].unsqueeze(0).float(), sr)

                new_audio_clip = mp.AudioFileClip(temp_sep_wav)
                new_video_clip = video_clip.set_audio(new_audio_clip)

                out_vid_path = os.path.join(out_dir, f"Separated_Source_{i + 1}.mp4")
                self.log_signal.emit(f"🔄 正在压制: Separated_Source_{i + 1}.mp4 (耗时较长，请稍候)")
                # 使用基础配置快速导出
                # 强行指定 -pix_fmt yuv420p，这是 PyQt5/Windows 最容易接受的标准格式
                new_video_clip.write_videofile(
                    out_vid_path,
                    codec="libx264",
                    audio_codec="aac",
                    preset="ultrafast",
                    ffmpeg_params=["-pix_fmt", "yuv420p"],
                    logger=None
                )
                out_vid_paths.append(out_vid_path)

            video_clip.close()
            self.log_signal.emit("✅ 视频处理全部完成！")
            self.finished_signal.emit(out_vid_paths, self.video_path)
        except Exception as e:
            self.error_signal.emit(str(e))


# ==========================================
# 4. 主界面设计 (带页签、带播放器)
# ==========================================
class SeparationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PRO 版多源信号分离工作站 (Audio & Video)")
        self.resize(1200, 800)
        self.manager = SeparationManager()

        # 公共播放器引擎 (音频)
        self.audio_player = QMediaPlayer()

        self.initUI()
        self.apply_dark_theme()

    def initUI(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 构建两个子页面
        self.tabs.addTab(self.create_audio_tab(), "🎵 纯语音分离")
        self.tabs.addTab(self.create_video_tab(), "🎬 视频干音提取")

    # ------------------ 页面 1：音频 ------------------
    def create_audio_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # 左面板
        left = QVBoxLayout()
        left.addWidget(QLabel("1. 选择语音分离模型:"))
        self.a_model_cb = QComboBox()
        # 清理了通信模型，只保留语音相关
        speech_models = [
            "【语音】TasNet (基础架构)", "【语音】Deep TasNet (深度优化)",
            "【语音】MultiScale TasNet (多尺度版)", "【语音】MultiScale-Deep (综合终极版)",
            "【语音】TasNet (3说话人分离)", "【语音】TasNet (4说话人分离)",
            "【语音】TF-BLSTM (时频域基准)", "【语音】Asteroid DPRNN (现成预训练)", "【语音】SpeechBrain SepFormer (SOTA)"
        ]
        self.a_model_cb.addItems(speech_models)
        self.a_model_cb.setCurrentText("【语音】MultiScale-Deep (综合终极版)")
        left.addWidget(self.a_model_cb)

        left.addWidget(QLabel("2. 上传混合语音 (.wav):"))
        self.a_mix_btn = QPushButton("📂 选择音频文件")
        self.a_mix_btn.clicked.connect(lambda: self.select_file("audio"))
        self.a_mix_label = QLabel("未选择文件")
        self.a_mix_label.setStyleSheet("color: gray;")
        left.addWidget(self.a_mix_btn)
        left.addWidget(self.a_mix_label)

        left.addStretch()
        self.a_run_btn = QPushButton("⚡ 执行切片分离")
        self.a_run_btn.setMinimumHeight(50)
        self.a_run_btn.setStyleSheet("background-color: #2E8B57;")
        self.a_run_btn.clicked.connect(self.run_audio)
        left.addWidget(self.a_run_btn)

        self.a_log_box = QTextEdit()
        self.a_log_box.setReadOnly(True)
        self.a_log_box.setMaximumHeight(150)
        left.addWidget(self.a_log_box)

        # 右面板 (波形图 + 播放器)
        right = QVBoxLayout()
        self.a_figure = Figure(figsize=(6, 6), dpi=100)
        self.a_figure.patch.set_facecolor('#2b2b2b')
        self.a_canvas = FigureCanvas(self.a_figure)
        right.addWidget(self.a_canvas, stretch=4)

        # 播放器控制区
        self.a_play_layout = QHBoxLayout()
        self.a_stop_btn = QPushButton("⏹️ 停止播放")
        self.a_stop_btn.clicked.connect(self.audio_player.stop)
        self.a_play_layout.addWidget(self.a_stop_btn)
        right.addLayout(self.a_play_layout, stretch=1)

        layout.addLayout(left, 3)
        layout.addLayout(right, 7)
        self.a_mix_path = ""
        return widget

    # ------------------ 页面 2：视频 ------------------
    def create_video_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # 左面板
        left = QVBoxLayout()
        left.addWidget(QLabel("1. 选择语音分离模型:"))
        self.v_model_cb = QComboBox()
        self.v_model_cb.addItems([
            "【语音】MultiScale-Deep (综合终极版)",
            "【语音】Deep TasNet (深度优化)",
            "【语音】SpeechBrain SepFormer (SOTA)"
        ])
        left.addWidget(self.v_model_cb)

        left.addWidget(QLabel("2. 上传包含人声的视频 (.mp4):"))
        self.v_mix_btn = QPushButton("📂 选择视频文件")
        self.v_mix_btn.clicked.connect(lambda: self.select_file("video"))
        self.v_mix_label = QLabel("未选择文件")
        self.v_mix_label.setStyleSheet("color: gray;")
        left.addWidget(self.v_mix_btn)
        left.addWidget(self.v_mix_label)

        left.addStretch()
        self.v_run_btn = QPushButton("🎬 开始分离视频音轨")
        self.v_run_btn.setMinimumHeight(50)
        self.v_run_btn.setStyleSheet("background-color: #0078D7;")
        self.v_run_btn.clicked.connect(self.run_video)
        left.addWidget(self.v_run_btn)

        self.v_log_box = QTextEdit()
        self.v_log_box.setReadOnly(True)
        self.v_log_box.setMaximumHeight(150)
        left.addWidget(self.v_log_box)

        # 右面板 (视频播放器网格)
        right = QVBoxLayout()
        self.v_grid = QGridLayout()
        right.addLayout(self.v_grid)

        layout.addLayout(left, 3)
        layout.addLayout(right, 7)
        self.v_mix_path = ""
        return widget

    # ------------------ 核心逻辑 ------------------
    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget, QTabWidget::pane { background-color: #2b2b2b; color: #ffffff; font-family: Arial; }
            QLabel { color: #e0e0e0; }
            QComboBox, QPushButton, QTextEdit { background-color: #3c3f41; color: white; border: 1px solid #555555; border-radius: 4px; padding: 5px; }
            QPushButton:hover { background-color: #4b4e50; }
            QTabBar::tab { background: #3c3f41; color: gray; padding: 10px; border: 1px solid #555; }
            QTabBar::tab:selected { background: #2b2b2b; color: white; font-weight: bold; border-bottom: none; }
        """)

    def select_file(self, mode):
        if mode == "audio":
            path, _ = QFileDialog.getOpenFileName(self, "选择混合音频", "", "Audio Files (*.wav)")
            if path:
                self.a_mix_path = path
                self.a_mix_label.setText(os.path.basename(path))
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mkv)")
            if path:
                self.v_mix_path = path
                self.v_mix_label.setText(os.path.basename(path))

    # --- 音频部分槽函数 ---
    def run_audio(self):
        if not self.a_mix_path: return QMessageBox.warning(self, "错误", "请先上传音频！")
        self.a_run_btn.setEnabled(False)
        self.a_log_box.append("-" * 20)

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
        num_plots = len(sep_paths) + 1  # 混合音 + 分离音

        # 画混合原音
        ax_mix = self.a_figure.add_subplot(num_plots, 1, 1)
        mix_wav, _ = torchaudio.load(mix_path)
        ax_mix.plot(mix_wav[0].numpy(), color='gray', linewidth=0.5)
        ax_mix.set_title("Original Mixture (Before)", color='white', fontsize=10)
        ax_mix.set_facecolor('#2b2b2b')
        ax_mix.tick_params(colors='white', bottom=False, labelbottom=False)

        # 画分离音
        for i, path in enumerate(sep_paths):
            wav, _ = torchaudio.load(path)
            ax = self.a_figure.add_subplot(num_plots, 1, i + 2)
            ax.plot(wav[0].numpy(), color='#1DB954', linewidth=0.5)
            ax.set_title(f"Separated Source {i + 1} (After)", color='white', fontsize=10)
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white', bottom=False, labelbottom=False)

        self.a_figure.tight_layout()
        self.a_canvas.draw()

    def setup_audio_players(self, mix_path, sep_paths):
        # 清理旧的播放按钮 (除了停止按钮)
        for i in reversed(range(1, self.a_play_layout.count())):
            self.a_play_layout.itemAt(i).widget().setParent(None)

        btn_mix = QPushButton("▶️ 播放原混合音")
        btn_mix.clicked.connect(lambda: self.play_media(mix_path))
        self.a_play_layout.addWidget(btn_mix)

        for i, path in enumerate(sep_paths):
            btn = QPushButton(f"▶️ 播放干音 {i + 1}")
            btn.setStyleSheet("color: #1DB954; font-weight: bold;")
            btn.clicked.connect(lambda checked, p=path: self.play_media(p))
            self.a_play_layout.addWidget(btn)

    def play_media(self, path):
        self.audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.audio_player.play()

    # --- 视频部分槽函数 ---
    def run_video(self):
        if not self.v_mix_path: return QMessageBox.warning(self, "错误", "请先上传视频！")
        self.v_run_btn.setEnabled(False)
        self.v_log_box.append("-" * 20)

        self.v_worker = VideoWorker(self.manager, self.v_model_cb.currentText(), self.v_mix_path)
        self.v_worker.log_signal.connect(self.v_log_box.append)
        self.v_worker.error_signal.connect(lambda e: self.on_error(e, self.v_run_btn))
        self.v_worker.finished_signal.connect(self.on_video_finished)
        self.v_worker.start()

    def on_video_finished(self, out_vids, mix_vid):
        self.v_run_btn.setEnabled(True)

        # 🚀 双重保护：同时保护 Player 和 Widget 不被系统垃圾回收
        if not hasattr(self, 'v_players'):
            self.v_players = []
        if not hasattr(self, 'v_widgets'):
            self.v_widgets = []

        for p in self.v_players:
            p.stop()
        self.v_players.clear()
        self.v_widgets.clear()

        for i in reversed(range(self.v_grid.count())):
            widget = self.v_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        all_vids = [mix_vid] + out_vids
        titles = ["Original Video"] + [f"Separated Spk {i + 1}" for i in range(len(out_vids))]

        for i, (path, title) in enumerate(zip(all_vids, titles)):
            row, col = i // 2, i % 2

            box = QGroupBox(title)
            vbox = QVBoxLayout()

            vw = QVideoWidget()
            player = QMediaPlayer()  # 简化初始化，去掉 VideoSurface 标志，避免冲突

            self.v_players.append(player)
            self.v_widgets.append(vw)  # 核心修复：保住画板的命

            player.setVideoOutput(vw)

            # 强制将 Windows 路径的 \ 替换为 /，防止 QUrl 解析失败
            abs_path = os.path.abspath(path).replace('\\', '/')
            print(f"🎬 [探针] 正在尝试加载视频: {abs_path}")

            player.setMedia(QMediaContent(QUrl.fromLocalFile(abs_path)))

            # 🚀 错误监控探针：如果黑屏，控制台会立刻打印真实原因！
            player.error.connect(lambda err, p=player: print(f"❌ 播放器底层报错: {p.errorString()}"))

            # === 原来的 PyQt 傲娇播放器可以保留画板当个封面占位 ===
            player.setVideoOutput(vw)
            abs_path = os.path.abspath(path).replace('\\', '/')
            player.setMedia(QMediaContent(QUrl.fromLocalFile(abs_path)))

            # === 🚀 修改这里的按钮：直接调用系统最高级播放器！ ===
            btn = QPushButton("🎬 调用系统播放器全屏观看")
            btn.setMinimumHeight(40)
            btn.setStyleSheet("font-weight: bold; color: white; background-color: #0078D7;")

            # 直接用 os.startfile 让 Windows 决定用什么软件打开它，绝对不黑屏！
            btn.clicked.connect(lambda checked, p=abs_path: os.startfile(p))

            vbox.addWidget(vw)
            vbox.addWidget(btn)
            box.setLayout(vbox)
            self.v_grid.addWidget(box, row, col)
        # 🚀 兜底方案：自动弹出输出文件夹！
        # 就算 PyQt5 播放器傲娇不干活，你也可以直接双击用 Windows 自带播放器看效果！
        video_out_dir = os.path.dirname(out_vids[0])
        print(f"🎉 视频处理完毕！已自动为您打开所在文件夹: {video_out_dir}")
        os.startfile(video_out_dir)

    def toggle_video(self, player):
        if player.state() == QMediaPlayer.PlayingState:
            player.pause()
        else:
            player.play()

    def on_error(self, err_msg, btn):
        QMessageBox.critical(self, "执行失败", f"系统报错: {err_msg}")
        btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SeparationApp()
    window.show()
    sys.exit(app.exec_())