import sys
import os
import torchaudio
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 导入我们的终极总控台
from app.Models.Code.ModelManager import SeparationManager


# ==========================================
# 1. 后台推理线程 (防止界面在跑深度学习时卡死白屏)
# ==========================================
class InferenceWorker(QThread):
    finished_signal = pyqtSignal(list)  # 成功完成，返回分离后的音频路径列表
    error_signal = pyqtSignal(str)  # 发生错误，返回报错信息
    log_signal = pyqtSignal(str)  # 传递运行日志

    def __init__(self, manager, model_choice, mix_path, ref1_path, ref2_path):
        super().__init__()
        self.manager = manager
        self.model_choice = model_choice
        self.mix_path = mix_path
        self.ref1_path = ref1_path
        self.ref2_path = ref2_path

    def run(self):
        try:
            self.log_signal.emit(f"🚀 正在加载后端引擎并推理，请耐心等待...")
            # 1. 执行推理
            est_sources, sr = self.manager.separate_audio(
                mix_path=self.mix_path,
                model_choice=self.model_choice,
                ref1_path=self.ref1_path,
                ref2_path=self.ref2_path
            )

            # 2. 保存结果
            out_dir = os.path.join(os.getcwd(), "GUI_Outputs")
            os.makedirs(out_dir, exist_ok=True)

            num_speakers = est_sources.shape[0]
            out_paths = []

            self.log_signal.emit(f"✅ 推理成功！检出 {num_speakers} 个独立音源，正在保存...")

            for i in range(num_speakers):
                out_path = os.path.join(out_dir, f"est_source_{i + 1}.wav")
                # 保存为 float32 的 wav
                torchaudio.save(out_path, est_sources[i].unsqueeze(0).float(), sr)
                out_paths.append(out_path)

            self.finished_signal.emit(out_paths)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))


# ==========================================
# 2. PyQt5 主界面设计
# ==========================================
class SeparationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("深度学习多源信号分离系统 (桌面端)")
        self.resize(1000, 700)

        # 加载总控台
        self.manager = SeparationManager()

        self.initUI()
        self.apply_dark_theme()

    def initUI(self):
        # 主部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ----------------- 左侧控制面板 -----------------
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)

        title = QLabel("🎛️ 信号分离控制台")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        left_panel.addWidget(title)

        # 1. 模型选择
        left_panel.addWidget(QLabel("1. 选择分离模型:"))
        self.model_cb = QComboBox()
        self.model_cb.addItems(list(self.manager.ckpt_paths.keys()))  # 备用，下面用我们整理好的中文名

        model_choices = [
            "【语音】TasNet (基础架构)", "【语音】Deep TasNet (深度优化)",
            "【语音】MultiScale TasNet (多尺度版)", "【语音】MultiScale-Deep (综合终极版)",
            "【语音】TasNet (3说话人分离)", "【语音】TasNet (4说话人分离)",
            "【语音】TF-BLSTM (时频域基准)", "【语音】Asteroid DPRNN (现成预训练)", "【语音】SpeechBrain SepFormer (SOTA)",
            "【通信】TasNet (基础架构)", "【通信】Deep TasNet (深度优化)",
            "【通信】MultiScale TasNet (多尺度版)", "【通信】TF-BLSTM (时频域基准)",
            "【通信】Asteroid DPRNN (长序列优化)", "【通信】SepFormer / DPTNet (Transformer)",
            "【理论上限】IRM Oracle (非盲分离)"
        ]
        self.model_cb.clear()
        self.model_cb.addItems(model_choices)
        self.model_cb.setCurrentText("【语音】MultiScale-Deep (综合终极版)")
        self.model_cb.currentTextChanged.connect(self.on_model_changed)
        left_panel.addWidget(self.model_cb)

        # 2. 混合信号上传
        left_panel.addWidget(QLabel("2. 上传混合信号 (Mix .wav / .npy):"))
        self.mix_btn = QPushButton("📂 选择混合文件")
        self.mix_btn.clicked.connect(lambda: self.select_file("mix"))
        self.mix_label = QLabel("未选择文件")
        self.mix_label.setWordWrap(True)
        self.mix_label.setStyleSheet("color: gray;")
        left_panel.addWidget(self.mix_btn)
        left_panel.addWidget(self.mix_label)

        # 3. IRM 参考信号区 (默认隐藏)
        self.irm_group = QGroupBox("⚙️ IRM 参考信号 (仅非盲分离需要)")
        irm_layout = QVBoxLayout()

        self.ref1_btn = QPushButton("📂 选择参考信号 1 (s1)")
        self.ref1_btn.clicked.connect(lambda: self.select_file("ref1"))
        self.ref1_label = QLabel("未选择文件")

        self.ref2_btn = QPushButton("📂 选择参考信号 2 (s2)")
        self.ref2_btn.clicked.connect(lambda: self.select_file("ref2"))
        self.ref2_label = QLabel("未选择文件")

        irm_layout.addWidget(self.ref1_btn)
        irm_layout.addWidget(self.ref1_label)
        irm_layout.addWidget(self.ref2_btn)
        irm_layout.addWidget(self.ref2_label)
        self.irm_group.setLayout(irm_layout)
        self.irm_group.setVisible(False)
        left_panel.addWidget(self.irm_group)

        # 4. 执行按钮与日志
        left_panel.addStretch()
        self.run_btn = QPushButton("⚡ 开始分离推理")
        self.run_btn.setMinimumHeight(50)
        self.run_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.run_btn.setStyleSheet("background-color: #2E8B57; color: white;")
        self.run_btn.clicked.connect(self.run_inference)
        left_panel.addWidget(self.run_btn)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(100)
        left_panel.addWidget(self.log_box)

        # ----------------- 右侧波形图面板 -----------------
        right_panel = QVBoxLayout()
        self.figure = Figure(figsize=(6, 8), dpi=100)
        self.figure.patch.set_facecolor('#2b2b2b')  # 图表背景设为深色
        self.canvas = FigureCanvas(self.figure)
        right_panel.addWidget(self.canvas)

        # 布局合并 (左侧占30%，右侧占70%)
        main_layout.addLayout(left_panel, 3)
        main_layout.addLayout(right_panel, 7)

        # 变量初始化
        self.mix_path = ""
        self.ref1_path = ""
        self.ref2_path = ""

    def apply_dark_theme(self):
        """应用深色科技风主题"""
        dark_stylesheet = """
        QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; font-family: Arial; }
        QLabel { color: #e0e0e0; }
        QComboBox, QPushButton, QTextEdit { 
            background-color: #3c3f41; color: white; border: 1px solid #555555; border-radius: 4px; padding: 5px; 
        }
        QPushButton:hover { background-color: #4b4e50; }
        QPushButton:pressed { background-color: #5b5e60; }
        QGroupBox { border: 1px solid #555555; margin-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; color: #a9b7c6; }
        """
        self.setStyleSheet(dark_stylesheet)

    def on_model_changed(self, text):
        # 如果选中了 IRM，则显示参考音频上传框
        if "IRM Oracle" in text:
            self.irm_group.setVisible(True)
        else:
            self.irm_group.setVisible(False)

    def select_file(self, file_type):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "音频/信号文件 (*.wav *.npy)")
        if path:
            if file_type == "mix":
                self.mix_path = path
                self.mix_label.setText(os.path.basename(path))
            elif file_type == "ref1":
                self.ref1_path = path
                self.ref1_label.setText(os.path.basename(path))
            elif file_type == "ref2":
                self.ref2_path = path
                self.ref2_label.setText(os.path.basename(path))

    def log(self, text):
        self.log_box.append(text)
        self.log_box.moveCursor(QTextCursor.End)

    def run_inference(self):
        if not self.mix_path:
            QMessageBox.warning(self, "错误", "请先上传混合信号文件 (Mix)！")
            return

        model_choice = self.model_cb.currentText()
        if "IRM Oracle" in model_choice and (not self.ref1_path or not self.ref2_path):
            QMessageBox.warning(self, "错误", "IRM 模式必须上传两个参考信号文件！")
            return

        self.run_btn.setEnabled(False)
        self.log("-----------------------------------------")

        # 启动后台线程
        self.worker = InferenceWorker(self.manager, model_choice, self.mix_path, self.ref1_path, self.ref2_path)
        self.worker.log_signal.connect(self.log)
        self.worker.error_signal.connect(self.on_error)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_error(self, err_msg):
        self.log(f"❌ 运行报错: {err_msg}")
        QMessageBox.critical(self, "执行失败", err_msg)
        self.run_btn.setEnabled(True)

    def on_finished(self, out_paths):
        self.log(f"🎉 文件已保存至: {os.path.dirname(out_paths[0])}")
        self.plot_waveforms(out_paths)
        self.run_btn.setEnabled(True)

        # 弹出保存文件夹
        os.startfile(os.path.dirname(out_paths[0]))

    def plot_waveforms(self, audio_paths):
        """使用 Matplotlib 绘制分离出的波形图"""
        self.figure.clear()
        num_plots = len(audio_paths)

        for i, path in enumerate(audio_paths):
            try:
                wav, sr = torchaudio.load(path)
                wav = wav[0].numpy()  # 转为 1D numpy 数组

                # 绘制子图
                ax = self.figure.add_subplot(num_plots, 1, i + 1)
                ax.plot(wav, color='#1DB954', linewidth=0.5)  # 极客绿
                ax.set_title(f"Separated Source {i + 1}", color='white')
                ax.set_facecolor('#2b2b2b')
                ax.tick_params(colors='white')

                # 隐藏上右边框使图表更清爽
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#555555')
                ax.spines['left'].set_color('#555555')

            except Exception as e:
                self.log(f"画图失败: {e}")

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 强制使用 Fusion 风格，跨平台看着都一样酷
    app.setStyle("Fusion")

    window = SeparationApp()
    window.show()
    sys.exit(app.exec_())