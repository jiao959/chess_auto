from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from auto_move_worker import AutoMoveWorker
from app_paths import app_root, module_command

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = app_root()
DEBUG_DIR = PROJECT_ROOT / "debug_outputs"
CONFIG_PATH = PROJECT_ROOT / "config.json"
VISUALIZATION_PATH = DEBUG_DIR / "bestmove_visualization.png"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

MODULES = [
    ("模块一：截图与棋盘识别", "module1_board_detection.py"),
    ("模块二：棋盘交点定位与 90 点裁剪", "module2_point_crops.py"),
    ("模块三：棋子识别", "module3_piece_recognition.py"),
    ("模块四：合法性检查与 FEN 生成", "module4_fen_generation.py"),
    ("模块五：Pikafish 最佳走法分析", "module5_engine_analysis.py"),
]


def decode_process_line(line: bytes) -> str:
    raw = line.rstrip(b"\r\n")
    for encoding in ("utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


class AnalysisWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(bool, str)
    module_started = Signal(str)
    module_finished = Signal(str)

    def __init__(self, ignore_rects: list[dict[str, int]] | None = None) -> None:
        super().__init__()
        self.stop_requested = False
        self.current_process: subprocess.Popen[str] | None = None
        self.ignore_rects = ignore_rects or []

    def request_stop(self) -> None:
        self.stop_requested = True
        proc = self.current_process
        if proc is None:
            return

        try:
            if proc.poll() is None:
                self.log.emit("正在停止当前模块进程...")
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.log.emit("进程未及时退出，执行 kill。")
                    proc.kill()
                    proc.wait(timeout=2)
        except Exception as exc:
            self.log.emit(f"停止分析时发生异常：{exc}")
        finally:
            self.current_process = None

    def run(self) -> None:
        try:
            for module_name, script_name in MODULES:
                if self.stop_requested:
                    self.log.emit("分析已停止。")
                    self.finished_ok.emit(False, "分析已停止。")
                    return

                self.module_started.emit(module_name)
                self.log.emit(f"\n=== 开始执行：{module_name} ===")

                ok = self.run_module(module_name, script_name)

                if self.stop_requested:
                    self.log.emit("分析已停止。")
                    self.finished_ok.emit(False, "分析已停止。")
                    return

                if not ok:
                    message = f"模块执行失败：{module_name}"
                    self.log.emit(message)
                    self.finished_ok.emit(False, message)
                    return

                self.log.emit(f"=== 完成：{module_name} ===")
                self.module_finished.emit(script_name)

            self.log.emit("\n分析流程完成。")
            self.finished_ok.emit(True, "分析成功完成。")

        except Exception as exc:
            message = f"分析线程异常：{exc}"
            self.log.emit(message)
            self.finished_ok.emit(False, message)
        finally:
            self.current_process = None

    def run_module(self, module_name: str, script_name: str) -> bool:
        script_path = PROJECT_ROOT / script_name
        if not getattr(sys, "frozen", False) and not script_path.exists():
            self.log.emit(f"未找到模块脚本：{script_path}")
            return False

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        if script_name == "module1_board_detection.py" and self.ignore_rects:
            env["CHESS_IGNORE_RECTS"] = json.dumps(self.ignore_rects, ensure_ascii=False)

        try:
            proc = subprocess.Popen(
                module_command(script_name),
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                env=env,
                creationflags=CREATE_NO_WINDOW,
            )
        except Exception as exc:
            self.log.emit(f"启动模块失败：{module_name}，{exc}")
            return False

        self.current_process = proc
        assert proc.stdout is not None

        for line in iter(proc.stdout.readline, b""):
            if line:
                self.log.emit(decode_process_line(line))
            if self.stop_requested:
                self.request_stop()
                return False

        code = proc.wait()
        self.current_process = None

        if code != 0:
            self.log.emit(f"{module_name} 退出码：{code}")
            return False

        return self.validate_module_outputs(module_name, script_name)

    def validate_module_outputs(self, module_name: str, script_name: str) -> bool:
        if script_name == "module1_board_detection.py":
            if not (DEBUG_DIR / "board_crop.png").exists():
                self.log.emit("模块一错误：未生成 debug_outputs/board_crop.png")
                return False

        elif script_name == "module2_point_crops.py":
            required = [
                DEBUG_DIR / "board_geometry.json",
                DEBUG_DIR / "points.json",
                DEBUG_DIR / "points_overlay.png",
                DEBUG_DIR / "point_crops",
            ]
            for path in required:
                if not path.exists():
                    self.log.emit(f"模块二错误：未生成 {path.name}")
                    return False

        elif script_name == "module3_piece_recognition.py":
            if not (DEBUG_DIR / "piece_recognition_results.json").exists():
                self.log.emit("模块三错误：未生成 piece_recognition_results.json")
                return False

        elif script_name == "module4_fen_generation.py":
            legal_path = DEBUG_DIR / "legal_check.json"
            fen_path = DEBUG_DIR / "current_fen.txt"

            if not legal_path.exists():
                self.log.emit("模块四错误：未生成 legal_check.json")
                return False

            try:
                data = json.loads(legal_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.log.emit(f"模块四错误：无法读取 legal_check.json，{exc}")
                return False

            if data.get("ok") is not True:
                errors = data.get("errors") or []
                if errors:
                    for error in errors:
                        if isinstance(error, dict):
                            msg = error.get("message", "未知错误")
                            self.log.emit(f"模块四错误：{msg}")
                        else:
                            self.log.emit(f"模块四错误：{error}")
                else:
                    self.log.emit("模块四错误：棋子错误识别或局面不合法")
                return False

            if not fen_path.exists():
                self.log.emit("模块四错误：未生成 current_fen.txt")
                return False

        elif script_name == "module5_engine_analysis.py":
            result_path = DEBUG_DIR / "engine_result.json"
            if not result_path.exists():
                self.log.emit("模块五错误：未生成 engine_result.json")
                return False

            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.log.emit(f"模块五错误：无法读取 engine_result.json，{exc}")
                return False

            if data.get("ok") is not True:
                error = data.get("error") or "未知错误"

                if str(error).startswith("模块五错误："):
                    self.log.emit(str(error))
                else:
                    self.log.emit(f"模块五错误：{error}")

                return False

        return True


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("中国象棋辅助软件")
        self.resize(1180, 780)

        self.worker: AnalysisWorker | None = None
        self.auto_worker: AutoMoveWorker | None = None

        self.start_button = QPushButton("开始分析")
        self.stop_button = QPushButton("停止分析")
        self.auto_start_button = QPushButton("开始自动走棋")
        self.auto_stop_button = QPushButton("停止自动走棋")

        self.stop_button.setEnabled(False)
        self.auto_stop_button.setEnabled(False)

        self.depth_radio = QRadioButton("按深度")
        self.time_radio = QRadioButton("按时间")
        self.depth_radio.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.depth_radio)
        self.mode_group.addButton(self.time_radio)

        self.depth_input = QLineEdit("20")
        self.movetime_input = QLineEdit("1000")
        self.hash_input = QLineEdit("1024")
        self.threads_input = QLineEdit("8")

        self.fen_label = QLabel("FEN：-")
        self.bestmove_label = QLabel("bestmove：-")
        self.chinese_label = QLabel("中文走法：-")

        self.image_label = QLabel("尚未生成最佳走法可视化图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(520, 520)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        self.load_saved_config()
        self.build_ui()
        self.connect_signals()
        self.reset_result_display()

    def build_ui(self) -> None:
        central = QWidget()
        root = QGridLayout(central)

        controls = QGroupBox("操作")
        controls_layout = QGridLayout(controls)
        controls_layout.addWidget(self.start_button, 0, 0)
        controls_layout.addWidget(self.stop_button, 0, 1)
        controls_layout.addWidget(self.auto_start_button, 1, 0)
        controls_layout.addWidget(self.auto_stop_button, 1, 1)

        params = QGroupBox("引擎参数")
        params_layout = QFormLayout(params)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.depth_radio)
        mode_layout.addWidget(self.time_radio)

        params_layout.addRow("分析模式", mode_layout)
        params_layout.addRow("深度", self.depth_input)
        params_layout.addRow("时间(ms)", self.movetime_input)
        params_layout.addRow("Hash(MB)", self.hash_input)
        params_layout.addRow("Threads", self.threads_input)

        status = QGroupBox("结果")
        status_layout = QVBoxLayout(status)
        status_layout.addWidget(self.fen_label)
        status_layout.addWidget(self.bestmove_label)
        status_layout.addWidget(self.chinese_label)

        left = QVBoxLayout()
        left.addWidget(controls)
        left.addWidget(params)
        left.addWidget(status)
        left.addStretch(1)

        visual = QGroupBox("最佳走法示意图")
        visual_layout = QVBoxLayout(visual)
        visual_layout.addWidget(self.image_label)

        logs = QGroupBox("日志")
        logs_layout = QVBoxLayout(logs)
        logs_layout.addWidget(self.log_text)

        root.addLayout(left, 0, 0)
        root.addWidget(visual, 0, 1)
        root.addWidget(logs, 1, 0, 1, 2)

        root.setColumnStretch(1, 1)
        root.setRowStretch(1, 1)

        self.setCentralWidget(central)

    def connect_signals(self) -> None:
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.auto_start_button.clicked.connect(self.start_auto_move)
        self.auto_stop_button.clicked.connect(self.stop_auto_move)

    def append_log(self, message: str) -> None:
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def clear_old_outputs(self) -> None:
        old_files = [
            "current_fen.txt",
            "legal_check.json",
            "bestmove.txt",
            "bestmove_chinese.txt",
            "engine_result.json",
            "engine_input_fen.txt",
            "engine_raw_log.txt",
            "bestmove_visualization.png",
        ]

        for name in old_files:
            path = DEBUG_DIR / name
            try:
                if path.exists():
                    path.unlink()
            except Exception as exc:
                self.append_log(f"清理旧文件失败：{path.name}，{exc}")

    def reset_result_display(self) -> None:
        self.fen_label.setText("FEN：-")
        self.bestmove_label.setText("bestmove：-")
        self.chinese_label.setText("中文走法：-")
        self.image_label.setText("尚未生成最佳走法可视化图片")
        self.image_label.setPixmap(QPixmap())

    def load_saved_config(self) -> None:
        if not CONFIG_PATH.exists():
            return

        try:
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return

        analysis_mode = config.get("analysis_mode", "depth")
        if analysis_mode == "movetime":
            self.time_radio.setChecked(True)
        else:
            self.depth_radio.setChecked(True)

        if "depth" in config:
            self.depth_input.setText(str(config.get("depth")))

        if "movetime" in config:
            self.movetime_input.setText(str(config.get("movetime")))

        if "hash" in config:
            self.hash_input.setText(str(config.get("hash")))

        if "threads" in config:
            self.threads_input.setText(str(config.get("threads")))

    def get_window_ignore_rect(self) -> list[dict[str, int]]:
        rect = self.frameGeometry()
        margin = 30

        return [
            {
                "x": max(0, rect.x() - margin),
                "y": max(0, rect.y() - margin),
                "w": rect.width() + margin * 2,
                "h": rect.height() + margin * 2,
            }
        ]

    def start_analysis(self) -> None:
        if self.auto_worker and self.auto_worker.isRunning():
            QMessageBox.warning(self, "提示", "当前正在自动走棋，不能同时启动手动分析。")
            return

        if self.worker and self.worker.isRunning():
            return

        try:
            self.write_config()
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return

        self.log_text.clear()
        self.clear_old_outputs()
        self.reset_result_display()

        self.append_log("开始手动分析。")
        self.append_log("使用窗口忽略区域进行截图识别，不隐藏项目窗口。")

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.auto_start_button.setEnabled(False)
        self.auto_stop_button.setEnabled(False)

        ignore_rects = self.get_window_ignore_rect()
        QApplication.processEvents()

        self._start_analysis_worker(ignore_rects)

    def _start_analysis_worker(self, ignore_rects: list[dict[str, int]]) -> None:
        self.worker = AnalysisWorker(ignore_rects=ignore_rects)
        self.worker.log.connect(self.append_log)
        self.worker.finished_ok.connect(self.analysis_finished)
        self.worker.module_finished.connect(self.on_module_finished)
        self.worker.start()

    def on_module_finished(self, script_name: str) -> None:
        if script_name == "module1_board_detection.py":
            self.restore_main_window()

    def restore_main_window(self) -> None:
        if not self.isVisible():
            self.showNormal()
            self.raise_()
            self.activateWindow()
            QApplication.processEvents()
            self.append_log("项目窗口已恢复显示。")

    def stop_analysis(self) -> None:
        self.append_log("收到停止分析请求。")
        if self.worker:
            try:
                self.worker.request_stop()
            except Exception as exc:
                self.append_log(f"停止分析异常：{exc}")

    def analysis_finished(self, ok: bool, message: str) -> None:
        self.restore_main_window()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.auto_start_button.setEnabled(True)
        self.auto_stop_button.setEnabled(False)

        if ok:
            self.read_final_outputs()
            self.load_visualization()
            self.append_log("分析成功完成。")
        else:
            self.reset_result_display()
            self.append_log(f"分析未完成或已失败：{message}")

    def start_auto_move(self) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "当前正在手动分析，不能同时启动自动走棋。")
            return

        if self.auto_worker and self.auto_worker.isRunning():
            return

        try:
            self.write_config()
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return

        self.log_text.clear()
        self.clear_old_outputs()
        self.reset_result_display()

        self.append_log("使用场景限定：仅用于本地测试、自我对弈或离线研究，不用于联网对局破坏公平性。")

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.auto_start_button.setEnabled(False)
        self.auto_stop_button.setEnabled(True)

        self.auto_worker = AutoMoveWorker(
            project_root=PROJECT_ROOT,
            ignore_rects=[],
            screenshot_delay=1.2,
            retry_delay=0.5,
            after_click_delay=0.5,
            click_gap=0.2,
        )

        self.auto_worker.log.connect(self.append_log)
        self.auto_worker.finished_ok.connect(self.auto_move_finished)

        self.auto_worker.start()

    def stop_auto_move(self) -> None:
        self.append_log("收到停止自动走棋请求。")

        if self.auto_worker and self.auto_worker.isRunning():
            self.auto_worker.request_stop()
        else:
            self.append_log("当前没有正在运行的自动走棋任务。")
            self.auto_start_button.setEnabled(True)
            self.auto_stop_button.setEnabled(False)
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def auto_move_finished(self, ok: bool, message: str) -> None:
        self.restore_main_window()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.auto_start_button.setEnabled(True)
        self.auto_stop_button.setEnabled(False)

        if message:
            self.append_log(message)

        if ok:
            self.read_final_outputs()
            self.load_visualization()
        else:
            if (DEBUG_DIR / "current_fen.txt").exists() or (DEBUG_DIR / "engine_result.json").exists():
                self.read_final_outputs()
                self.load_visualization()

    def write_config(self) -> None:
        depth = self.parse_positive_int(self.depth_input.text(), "深度")
        movetime = self.parse_positive_int(self.movetime_input.text(), "时间")
        hash_size = self.parse_positive_int(self.hash_input.text(), "Hash")
        threads = self.parse_positive_int(self.threads_input.text(), "Threads")

        config = {
            "analysis_mode": "depth" if self.depth_radio.isChecked() else "movetime",
            "depth": depth,
            "movetime": movetime,
            "hash": hash_size,
            "threads": threads,
        }

        CONFIG_PATH.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.append_log(f"已写入 config.json：{config}")

    @staticmethod
    def parse_positive_int(text: str, name: str) -> int:
        try:
            value = int(text.strip())
        except ValueError as exc:
            raise ValueError(f"{name} 必须是整数") from exc

        if value <= 0:
            raise ValueError(f"{name} 必须大于 0")

        return value

    def read_final_outputs(self) -> None:
        fen = self.read_text_file(DEBUG_DIR / "current_fen.txt", "未找到 current_fen.txt")
        bestmove = self.read_text_file(DEBUG_DIR / "bestmove.txt", "未找到 bestmove.txt")
        chinese = self.read_text_file(DEBUG_DIR / "bestmove_chinese.txt", "未找到 bestmove_chinese.txt")

        self.fen_label.setText(f"FEN：{fen or '-'}")
        self.bestmove_label.setText(f"bestmove：{bestmove or '-'}")
        self.chinese_label.setText(f"中文走法：{chinese or '-'}")

        if fen:
            self.append_log(f"识别到的 FEN：{fen}")
        if bestmove:
            self.append_log(f"bestmove：{bestmove}")
        if chinese:
            self.append_log(f"中文四字走法：{chinese}")

        engine_result = DEBUG_DIR / "engine_result.json"
        if engine_result.exists():
            try:
                data = json.loads(engine_result.read_text(encoding="utf-8"))
                if data.get("ok") is True:
                    self.append_log("引擎调用成功")
                else:
                    error = data.get("error") or "未知错误"
                    self.append_log(f"引擎调用失败：{error}")
            except Exception as exc:
                self.append_log(f"读取 engine_result.json 失败：{exc}")
        else:
            self.append_log("未找到 engine_result.json")

    def read_text_file(self, path: Path, missing_message: str) -> str | None:
        if not path.exists():
            self.append_log(missing_message)
            return None

        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            self.append_log(f"读取 {path.name} 失败：{exc}")
            return None

    def load_visualization(self) -> None:
        if not VISUALIZATION_PATH.exists():
            self.image_label.setText("尚未生成最佳走法可视化图片")
            self.image_label.setPixmap(QPixmap())
            self.append_log("未找到 debug_outputs/bestmove_visualization.png")
            return

        pixmap = QPixmap(str(VISUALIZATION_PATH))
        if pixmap.isNull():
            self.image_label.setText("最佳走法可视化图片加载失败")
            return

        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.image_label.setPixmap(scaled)
        self.image_label.setText("")

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if VISUALIZATION_PATH.exists():
            self.load_visualization()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self.write_config()
        except Exception:
            pass

        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(2500)

        if self.auto_worker and self.auto_worker.isRunning():
            self.auto_worker.request_stop()
            self.auto_worker.wait(2500)

        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def run_module_entry(script_name: str) -> int:
    modules = {
        "module1_board_detection.py": "module1_board_detection",
        "module2_point_crops.py": "module2_point_crops",
        "module3_piece_recognition.py": "module3_piece_recognition",
        "module4_fen_generation.py": "module4_fen_generation",
        "module5_engine_analysis.py": "module5_engine_analysis",
    }
    module_name = modules.get(script_name)
    if module_name is None:
        print(f"未知模块：{script_name}")
        return 2

    sys.argv = [script_name, *sys.argv[3:]]
    module = __import__(module_name)
    module.main()
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-module":
        sys.exit(run_module_entry(sys.argv[2]))
    main()
