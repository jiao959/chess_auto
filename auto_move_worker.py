from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    import mss
except Exception:
    mss = None

from PySide6.QtCore import QThread, Signal

from app_paths import module_command

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def decode_process_line(line: bytes) -> str:
    raw = line.rstrip(b"\r\n")
    for encoding in ("utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


AUTO_FULL_MODULES = [
    ("模块一：截图与棋盘识别", "module1_board_detection.py"),
    ("模块二：棋盘交点定位与 90 点裁剪", "module2_point_crops.py"),
    ("模块三：棋子识别", "module3_piece_recognition.py"),
    ("模块四：合法性检查与 FEN 生成", "module4_fen_generation.py"),
    ("模块五：Pikafish 最佳走法分析", "module5_engine_analysis.py"),
]

AUTO_RECOGNITION_MODULES = [
    ("模块一：截图与棋盘识别", "module1_board_detection.py"),
    ("模块二：棋盘交点定位与 90 点裁剪", "module2_point_crops.py"),
    ("模块三：棋子识别", "module3_piece_recognition.py"),
    ("模块四：合法性检查与 FEN 生成", "module4_fen_generation.py"),
]

AUTO_ENGINE_MODULE = ("模块五：Pikafish 最佳走法分析", "module5_engine_analysis.py")


@dataclass(frozen=True)
class ScreenMove:
    bestmove: str
    from_x: int
    from_y: int
    to_x: int
    to_y: int


class AutoMoveWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(bool, str)

    def __init__(
        self,
        project_root: str | Path,
        ignore_rects: list[dict[str, int]] | None = None,
        screenshot_delay: float = 1.2,
        retry_delay: float = 1.5,
        after_click_delay: float = 1.0,
        click_gap: float = 0.2,
        own_move_confirm_timeout: float = 8.0,
    ) -> None:
        super().__init__()
        self.project_root = Path(project_root).resolve()
        self.debug_dir = self.project_root / "debug_outputs"

        self.ignore_rects = ignore_rects or []
        self.screenshot_delay = screenshot_delay
        self.retry_delay = retry_delay
        self.after_click_delay = after_click_delay
        self.click_gap = click_gap
        self.own_move_confirm_timeout = own_move_confirm_timeout

        self.auto_stop_requested = False
        self.current_process: subprocess.Popen[str] | None = None

    def request_stop(self) -> None:
        self.auto_stop_requested = True
        self.terminate_current_process()

    def terminate_current_process(self) -> None:
        proc = self.current_process
        if proc is None:
            return

        try:
            if proc.poll() is None:
                self.log.emit("正在终止当前模块进程...")
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        except Exception as exc:
            self.log.emit(f"终止模块进程异常：{exc}")
        finally:
            self.current_process = None

    def run(self) -> None:
        try:
            if pyautogui is None:
                self.finished_ok.emit(False, "自动走棋错误：未安装 pyautogui")
                return

            self.log.emit("自动走棋启动。")

            while not self.auto_stop_requested:
                self.log.emit("\n=== 自动走棋：识别我方回合并分析 ===")
                ok = self.run_full_analysis_once()

                if not ok:
                    if self.auto_stop_requested:
                        break
                    self.finished_ok.emit(False, "自动走棋错误：分析失败")
                    return

                old_fen = self.read_text("current_fen.txt")
                bestmove = self.read_text("bestmove.txt")
                bestmove_chinese = self.read_text("bestmove_chinese.txt")

                if not old_fen:
                    self.finished_ok.emit(False, "自动走棋错误：找不到 current_fen.txt")
                    return

                if not bestmove:
                    self.finished_ok.emit(False, "自动走棋错误：找不到 bestmove.txt")
                    return

                own_color = self.get_own_color_from_current_position()
                if own_color is None:
                    self.finished_ok.emit(False, "自动走棋错误：无法判断我方颜色")
                    return

                opponent_color = self.opposite_color(own_color)
                own_color_cn = "红方" if own_color == "red" else "黑方"
                opponent_color_cn = "红方" if opponent_color == "red" else "黑方"

                self.log.emit(f"当前 FEN：{old_fen}")
                self.log.emit(f"我方颜色：{own_color_cn}")
                self.log.emit(f"等待对方颜色：{opponent_color_cn}")
                self.log.emit(f"bestmove：{bestmove}")

                if bestmove_chinese:
                    self.log.emit(f"中文四字走法：{bestmove_chinese}")

                screen_move = self.bestmove_to_screen_move(bestmove)
                if screen_move is None:
                    self.finished_ok.emit(False, "自动走棋错误：无法将 bestmove 转换为屏幕坐标")
                    return

                self.log.emit(f"起点屏幕坐标：{screen_move.from_x}, {screen_move.from_y}")
                self.log.emit(f"终点屏幕坐标：{screen_move.to_x}, {screen_move.to_y}")

                if not self.click_move(screen_move):
                    if self.auto_stop_requested:
                        break
                    self.finished_ok.emit(False, "自动走棋错误：鼠标点击失败")
                    return

                self.sleep_with_stop(self.after_click_delay)

                if self.auto_stop_requested:
                    break

                self.log.emit("我方点击完成，等待我方走法落盘。")

                confirmed_fen = self.wait_until_own_move_applied(
                    old_fen=old_fen,
                    bestmove=bestmove,
                )

                if self.auto_stop_requested:
                    break

                if not confirmed_fen:
                    self.finished_ok.emit(False, "自动走棋错误：我方走法未在棋盘上生效")
                    return

                self.log.emit(f"我方走法已确认，FEN：{confirmed_fen}")
                self.log.emit(f"等待对方真实走子：{opponent_color_cn}")

                opponent_fen = self.wait_until_color_move_observed(
                    base_fen=confirmed_fen,
                    mover_color=opponent_color,
                    mover_label="对方",
                )

                if self.auto_stop_requested:
                    break

                if not opponent_fen:
                    self.finished_ok.emit(False, "自动走棋错误：未检测到对方有效走子")
                    return

                self.log.emit(f"检测到对方已走子，识别 FEN：{opponent_fen}")
                self.log.emit("对方走子后重新执行完整识别与分析。")
                self.sleep_with_stop(self.after_click_delay)

            self.log.emit("自动走棋已停止。")
            self.finished_ok.emit(False, "自动走棋已停止。")

        except Exception as exc:
            self.finished_ok.emit(False, f"自动走棋错误：{exc}")
        finally:
            self.terminate_current_process()

    def clear_full_analysis_outputs(self) -> None:
        files = [
            "board_crop.png",
            "board_rect.json",
            "board_detect_overlay.png",
            "board_geometry.json",
            "points.json",
            "points_overlay.png",
            "piece_recognition_results.json",
            "current_fen.txt",
            "legal_check.json",
            "engine_input_fen.txt",
            "engine_raw_log.txt",
            "engine_result.json",
            "bestmove.txt",
            "bestmove_chinese.txt",
            "bestmove_visualization.png",
        ]

        for name in files:
            path = self.debug_dir / name
            try:
                if path.exists():
                    path.unlink()
            except Exception as exc:
                self.log.emit(f"清理旧文件失败：{name}，{exc}")

    def clear_recognition_outputs(self) -> None:
        files = [
            "board_crop.png",
            "board_rect.json",
            "board_detect_overlay.png",
            "board_geometry.json",
            "points.json",
            "points_overlay.png",
            "piece_recognition_results.json",
            "current_fen.txt",
            "legal_check.json",
        ]

        for name in files:
            path = self.debug_dir / name
            try:
                if path.exists():
                    path.unlink()
            except Exception as exc:
                self.log.emit(f"清理旧识别文件失败：{name}，{exc}")

    def clear_engine_outputs(self) -> None:
        files = [
            "engine_input_fen.txt",
            "engine_raw_log.txt",
            "engine_result.json",
            "bestmove.txt",
            "bestmove_chinese.txt",
            "bestmove_visualization.png",
        ]

        for name in files:
            path = self.debug_dir / name
            try:
                if path.exists():
                    path.unlink()
            except Exception as exc:
                self.log.emit(f"清理旧引擎文件失败：{name}，{exc}")

    def run_full_analysis_once(self) -> bool:
        self.clear_full_analysis_outputs()

        for module_name, script_name in AUTO_FULL_MODULES:
            if self.auto_stop_requested:
                return False

            ok = self.run_module_script(module_name, script_name)
            if not ok:
                self.log.emit(f"自动走棋错误：{module_name} 执行失败")
                return False

        engine_result = self.read_json("engine_result.json")
        if not engine_result:
            self.log.emit("自动走棋错误：找不到 engine_result.json")
            return False

        if engine_result.get("ok") is not True:
            error = engine_result.get("error") or "模块五引擎调用失败"
            self.log.emit(f"自动走棋错误：{error}")
            return False

        if not self.read_text("current_fen.txt"):
            self.log.emit("自动走棋错误：模块五完成后找不到 current_fen.txt")
            return False

        if not self.read_text("bestmove.txt"):
            self.log.emit("自动走棋错误：模块五完成后找不到 bestmove.txt")
            return False

        return True

    def run_engine_analysis_once(self) -> bool:
        if not self.read_text("current_fen.txt"):
            self.log.emit("自动走棋错误：复用局面时找不到 current_fen.txt")
            return False

        if not self.read_json("legal_check.json"):
            self.log.emit("自动走棋错误：复用局面时找不到 legal_check.json")
            return False

        if not self.read_json("points.json"):
            self.log.emit("自动走棋错误：复用局面时找不到 points.json")
            return False

        if not self.read_json("board_rect.json"):
            self.log.emit("自动走棋错误：复用局面时找不到 board_rect.json")
            return False

        self.clear_engine_outputs()

        module_name, script_name = AUTO_ENGINE_MODULE
        if not self.run_module_script(module_name, script_name):
            self.log.emit("自动走棋错误：模块五执行失败")
            return False

        engine_result = self.read_json("engine_result.json")
        if not engine_result:
            self.log.emit("自动走棋错误：找不到 engine_result.json")
            return False

        if engine_result.get("ok") is not True:
            error = engine_result.get("error") or "模块五引擎调用失败"
            self.log.emit(f"自动走棋错误：{error}")
            return False

        if not self.read_text("bestmove.txt"):
            self.log.emit("自动走棋错误：模块五完成后找不到 bestmove.txt")
            return False

        return True

    def run_recognition_to_fen_once(self) -> bool:
        if self.fast_recognition_to_fen_once():
            return True

        if self.auto_stop_requested:
            return False

        self.log.emit("快速识别失败，回退完整识别流程。")
        self.clear_recognition_outputs()

        for module_name, script_name in AUTO_RECOGNITION_MODULES:
            if self.auto_stop_requested:
                return False

            ok = self.run_module_script(module_name, script_name)
            if not ok:
                self.log.emit(f"局面重新识别失败：{module_name} 执行失败")
                return False

        fen = self.read_text("current_fen.txt")
        if not fen:
            self.log.emit("局面重新识别失败：未找到 current_fen.txt")
            return False

        return True

    def fast_recognition_to_fen_once(self) -> bool:
        if not self.refresh_board_crop_and_point_crops_fast():
            return False

        for module_name, script_name in AUTO_RECOGNITION_MODULES[2:]:
            if self.auto_stop_requested:
                return False

            ok = self.run_module_script(module_name, script_name)
            if not ok:
                self.log.emit(f"快速识别失败：{module_name} 执行失败")
                return False

        fen = self.read_text("current_fen.txt")
        if not fen:
            self.log.emit("快速识别失败：未找到 current_fen.txt")
            return False

        return True

    def refresh_board_crop_and_point_crops_fast(self) -> bool:
        if mss is None:
            self.log.emit("快速识别不可用：未安装 mss")
            return False

        board_rect = self.read_json("board_rect.json")
        points_payload = self.read_json("points.json")
        if not board_rect or not points_payload:
            self.log.emit("快速识别不可用：缺少 board_rect.json 或 points.json")
            return False

        points = points_payload.get("points")
        if not isinstance(points, list) or len(points) != 90:
            self.log.emit("快速识别不可用：points.json 点位数量不是 90")
            return False

        try:
            rect = {
                "left": int(round(float(board_rect["x"]))),
                "top": int(round(float(board_rect["y"]))),
                "width": int(round(float(board_rect["w"]))),
                "height": int(round(float(board_rect["h"]))),
            }
        except Exception as exc:
            self.log.emit(f"快速识别不可用：board_rect.json 无效，{exc}")
            return False

        try:
            with mss.mss() as sct:
                raw = sct.grab(rect)
            board_bgr = cv2.cvtColor(np.array(raw, dtype=np.uint8), cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            self.log.emit(f"快速识别截图失败：{exc}")
            return False

        board_crop_path = self.debug_dir / "board_crop.png"
        point_crops_dir = self.debug_dir / "point_crops"

        try:
            cv2.imwrite(str(board_crop_path), board_bgr)
        except Exception as exc:
            self.log.emit(f"快速识别保存 board_crop.png 失败：{exc}")
            return False

        try:
            if point_crops_dir.exists():
                for path in point_crops_dir.glob("r*c*.png"):
                    path.unlink()
            else:
                point_crops_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.log.emit(f"快速识别清理 point_crops 失败：{exc}")
            return False

        h, w = board_bgr.shape[:2]

        for point in points:
            try:
                row = int(point["row"])
                col = int(point["col"])
                crop_box = point.get("crop_box") if isinstance(point, dict) else None
                if isinstance(crop_box, dict) and crop_box:
                    left = int(round(float(crop_box["left"])))
                    top = int(round(float(crop_box["top"])))
                    right = int(round(float(crop_box["right"])))
                    bottom = int(round(float(crop_box["bottom"])))
                else:
                    cx = int(round(float(point.get("crop_x", point.get("x")))))
                    cy = int(round(float(point.get("crop_y", point.get("y")))))
                    radius = self.infer_crop_radius(points)
                    left = cx - radius
                    top = cy - radius
                    right = cx + radius
                    bottom = cy + radius
            except Exception as exc:
                self.log.emit(f"快速识别点位数据无效：{exc}")
                return False

            pad_left = max(0, -left)
            pad_top = max(0, -top)
            pad_right = max(0, right - w)
            pad_bottom = max(0, bottom - h)

            safe_left = max(0, left)
            safe_top = max(0, top)
            safe_right = min(w, right)
            safe_bottom = min(h, bottom)

            crop = board_bgr[safe_top:safe_bottom, safe_left:safe_right]
            if crop.size == 0:
                self.log.emit(f"快速识别裁剪为空：r{row}c{col}")
                return False

            if any([pad_left, pad_top, pad_right, pad_bottom]):
                crop = cv2.copyMakeBorder(
                    crop,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_REPLICATE,
                )

            output_path = point_crops_dir / f"r{row}c{col}.png"
            if not cv2.imwrite(str(output_path), crop):
                self.log.emit(f"快速识别保存裁剪图失败：r{row}c{col}")
                return False

        self.log.emit("快速识别：已复用棋盘区域和点位，仅刷新 90 点裁剪。")
        return True

    @staticmethod
    def infer_crop_radius(points: list[Any]) -> int:
        for point in points:
            if not isinstance(point, dict):
                continue
            crop_box = point.get("crop_box")
            if not isinstance(crop_box, dict):
                continue
            try:
                width = float(crop_box["right"]) - float(crop_box["left"])
                if width > 0:
                    return max(18, int(round(width / 2)))
            except Exception:
                continue
        return 41

    def run_module_script(self, module_name: str, script_name: str) -> bool:
        if self.auto_stop_requested:
            return False

        script_path = self.project_root / script_name
        if not getattr(sys, "frozen", False) and not script_path.exists():
            self.log.emit(f"自动走棋错误：未找到模块脚本 {script_path}")
            return False

        self.log.emit(f"\n=== 自动走棋执行：{module_name} ===")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        if script_name == "module1_board_detection.py" and self.ignore_rects:
            env["CHESS_IGNORE_RECTS"] = json.dumps(self.ignore_rects, ensure_ascii=False)

        try:
            proc = subprocess.Popen(
                module_command(script_name),
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                env=env,
                creationflags=CREATE_NO_WINDOW,
            )
        except Exception as exc:
            self.log.emit(f"自动走棋错误：启动 {module_name} 失败，{exc}")
            return False

        self.current_process = proc

        try:
            assert proc.stdout is not None

            for line in iter(proc.stdout.readline, b""):
                if line:
                    self.log.emit(decode_process_line(line))

                if self.auto_stop_requested:
                    self.terminate_current_process()
                    return False

            code = proc.wait()
            self.current_process = None

            if code != 0:
                self.log.emit(f"自动走棋错误：{module_name} 退出码 {code}")
                return False

            if not self.validate_module_outputs(script_name):
                return False

            self.log.emit(f"=== 自动走棋完成：{module_name} ===")
            return True

        finally:
            self.current_process = None

    def validate_module_outputs(self, script_name: str) -> bool:
        if script_name == "module1_board_detection.py":
            if not (self.debug_dir / "board_crop.png").exists():
                self.log.emit("自动走棋错误：模块一未生成 board_crop.png")
                return False

            if not (self.debug_dir / "board_rect.json").exists():
                self.log.emit("自动走棋错误：模块一未生成 board_rect.json")
                return False

        elif script_name == "module2_point_crops.py":
            required = [
                "board_geometry.json",
                "points.json",
                "points_overlay.png",
            ]

            for name in required:
                if not (self.debug_dir / name).exists():
                    self.log.emit(f"自动走棋错误：模块二未生成 {name}")
                    return False

            point_crops = self.debug_dir / "point_crops"
            if not point_crops.exists():
                self.log.emit("自动走棋错误：模块二未生成 point_crops 文件夹")
                return False

            if len(list(point_crops.glob("r*c*.png"))) != 90:
                self.log.emit("自动走棋错误：模块二未生成 90 个点位裁剪图")
                return False

        elif script_name == "module3_piece_recognition.py":
            if not (self.debug_dir / "piece_recognition_results.json").exists():
                self.log.emit("自动走棋错误：模块三未生成 piece_recognition_results.json")
                return False

        elif script_name == "module4_fen_generation.py":
            legal_path = self.debug_dir / "legal_check.json"
            fen_path = self.debug_dir / "current_fen.txt"

            if not legal_path.exists():
                self.log.emit("自动走棋错误：模块四未生成 legal_check.json")
                return False

            try:
                legal = json.loads(legal_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.log.emit(f"自动走棋错误：无法读取 legal_check.json，{exc}")
                return False

            if legal.get("ok") is not True:
                errors = legal.get("errors") or []
                if errors:
                    for error in errors:
                        self.log.emit(f"自动走棋错误：模块四合法性检查失败：{error}")
                else:
                    self.log.emit("自动走棋错误：模块四合法性检查失败")
                return False

            if not fen_path.exists():
                self.log.emit("自动走棋错误：模块四未生成 current_fen.txt")
                return False

        elif script_name == "module5_engine_analysis.py":
            result_path = self.debug_dir / "engine_result.json"

            if not result_path.exists():
                self.log.emit("自动走棋错误：模块五未生成 engine_result.json")
                return False

            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.log.emit(f"自动走棋错误：无法读取 engine_result.json，{exc}")
                return False

            if result.get("ok") is not True:
                error = result.get("error") or "未知错误"
                self.log.emit(f"自动走棋错误：模块五失败：{error}")
                return False

            if not (self.debug_dir / "bestmove.txt").exists():
                self.log.emit("自动走棋错误：模块五未生成 bestmove.txt")
                return False

        return True

    def wait_until_own_move_applied(self, old_fen: str, bestmove: str) -> str | None:
        expected_fen = self.apply_bestmove_to_fen(old_fen, bestmove)
        if expected_fen is None:
            self.log.emit("自动走棋错误：无法生成我方走法后的预期 FEN")
            return None

        old_board = self.fen_board_part(old_fen)
        expected_board = self.fen_board_part(expected_fen)
        deadline = time.time() + self.own_move_confirm_timeout

        self.log.emit(f"我方走法预期 FEN：{expected_fen}")

        while not self.auto_stop_requested and time.time() < deadline:
            ok = self.run_recognition_to_fen_once()

            if self.auto_stop_requested:
                return None

            if not ok:
                self.log.emit("我方走法确认识别失败，0.5 秒后重试。")
                self.sleep_with_stop(0.5)
                continue

            new_fen = self.read_text("current_fen.txt")
            if not new_fen:
                self.log.emit("我方走法确认未找到 FEN，0.5 秒后重试。")
                self.sleep_with_stop(0.5)
                continue

            new_board = self.fen_board_part(new_fen)
            self.log.emit(f"我方走法确认 FEN：{new_fen}")

            if new_board == expected_board:
                return self.write_corrected_current_fen(expected_fen, "我方走法精确匹配预期 FEN")

            if self.own_move_is_visibly_applied(old_fen, new_fen, bestmove):
                self.log.emit("我方走法已从棋盘变化确认，识别 FEN 与预期不完全一致，使用预期 FEN 校正。")
                return self.write_corrected_current_fen(expected_fen, "我方走法容错校正")

            if new_board == old_board:
                self.log.emit("我方走法尚未落盘，继续等待。")
            else:
                self.log.emit("识别到非预期局面，继续等待我方走法稳定。")

            self.sleep_with_stop(0.5)

        return None

    def own_move_is_visibly_applied(self, old_fen: str, new_fen: str, bestmove: str) -> bool:
        old_board = self.parse_fen_board_matrix(old_fen)
        new_board = self.parse_fen_board_matrix(new_fen)
        parsed = self.parse_bestmove(bestmove)

        if old_board is None or new_board is None or parsed is None:
            return False

        from_row, from_col, to_row, to_col = parsed

        try:
            moving_piece = old_board[from_row][from_col]
            old_target = old_board[to_row][to_col]
            new_source = new_board[from_row][from_col]
            new_target = new_board[to_row][to_col]
        except Exception:
            return False

        if not moving_piece:
            return False

        source_cleared = new_source != moving_piece
        target_has_piece = new_target == moving_piece
        was_capture = bool(old_target) and old_target != moving_piece

        if target_has_piece:
            return True

        # Captures are the most likely place for OCR/模板识别 to be unstable:
        # the moving piece lands on top of a previously occupied point, while
        # highlights/animation can make the destination crop look like the old
        # captured piece for a few frames. If the source is definitely clear,
        # trust the engine move and correct the generated FEN.
        if was_capture and source_cleared:
            return True

        return False

    def write_corrected_current_fen(self, fen: str, reason: str) -> str:
        fen_path = self.debug_dir / "current_fen.txt"
        legal_path = self.debug_dir / "legal_check.json"

        try:
            fen_path.write_text(fen + "\n", encoding="utf-8")
        except Exception as exc:
            self.log.emit(f"自动走棋警告：写入校正 FEN 失败，{exc}")
            return fen

        legal = self.read_json("legal_check.json")
        if legal:
            legal["fen"] = fen
            legal["auto_corrected_fen"] = True
            legal["auto_corrected_reason"] = reason
            try:
                legal_path.write_text(json.dumps(legal, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                self.log.emit(f"自动走棋警告：写入校正 legal_check.json 失败，{exc}")

        self.log.emit(f"已校正 current_fen.txt：{fen}")
        return fen

    def wait_until_color_move_observed(self, base_fen: str, mover_color: str, mover_label: str) -> str | None:
        mover_color_cn = "红方" if mover_color == "red" else "黑方"
        base_board = self.fen_board_part(base_fen)

        while not self.auto_stop_requested:
            ok = self.run_recognition_to_fen_once()

            if self.auto_stop_requested:
                return None

            if not ok:
                self.log.emit("局面重新识别失败，0.5 秒后重试。")
                self.sleep_with_stop(self.retry_delay)
                continue

            new_fen = self.read_text("current_fen.txt")
            if not new_fen:
                self.log.emit("局面重新识别后未找到 FEN，0.5 秒后重试。")
                self.sleep_with_stop(self.retry_delay)
                continue

            self.log.emit(f"重新识别 FEN：{new_fen}")

            if self.fen_board_part(new_fen) == base_board:
                self.log.emit(f"局面仍是等待基准，继续等待{mover_label}走动。")
                self.sleep_with_stop(self.retry_delay)
                continue

            if self.color_has_plausible_move(base_fen, new_fen, mover_color):
                return new_fen

            self.log.emit(f"局面有变化，但不像{mover_label}{mover_color_cn}的一步真实走子，继续等待。")
            self.sleep_with_stop(self.retry_delay)

        return None

    def color_has_plausible_move(self, old_fen: str, new_fen: str, color: str) -> bool:
        old_entries = set(self.color_piece_entries(old_fen, color))
        new_entries = set(self.color_piece_entries(new_fen, color))

        removed = old_entries - new_entries
        added = new_entries - old_entries

        if len(removed) != 1 or len(added) != 1:
            return False

        removed_piece = next(iter(removed)).split("@", 1)[0]
        added_piece = next(iter(added)).split("@", 1)[0]
        if removed_piece != added_piece:
            return False

        self.log.emit(
            f"走子形态确认：{next(iter(removed))} -> {next(iter(added))}"
        )
        return True

    def get_own_color_from_current_position(self) -> str | None:
        legal_check = self.read_json("legal_check.json")
        if not legal_check:
            self.log.emit("自动走棋错误：找不到 legal_check.json，无法判断我方颜色")
            return None

        own_color = legal_check.get("own_color")
        if own_color in {"red", "black"}:
            return own_color

        orientation = legal_check.get("orientation")
        if orientation == "red_bottom":
            return "red"
        if orientation == "black_bottom":
            return "black"

        return None

    @staticmethod
    def opposite_color(color: str) -> str:
        return "black" if color == "red" else "red"

    def get_top_color_from_current_position(self) -> str | None:
        legal_check = self.read_json("legal_check.json")
        if not legal_check:
            self.log.emit("自动走棋错误：找不到 legal_check.json，无法判断棋盘方向")
            return None

        orientation = legal_check.get("orientation")
        own_color = legal_check.get("own_color")

        if orientation == "red_bottom":
            return "black"

        if orientation == "black_bottom":
            return "red"

        if own_color == "red":
            return "black"

        if own_color == "black":
            return "red"

        return None

    def expected_top_color_fingerprint_after_own_move(
        self,
        old_fen: str,
        top_color: str,
        bestmove: str,
    ) -> str | None:
        board = self.parse_fen_board_matrix(old_fen)
        parsed = self.parse_bestmove(bestmove)

        if board is None or parsed is None:
            return None

        _, _, to_row, to_col = parsed

        top_entries = self.color_piece_entries(old_fen, top_color)

        try:
            target_piece = board[to_row][to_col]
        except Exception:
            return None

        if self.piece_belongs_to_color(target_piece, top_color):
            captured_entry = f"{target_piece}@{to_row},{to_col}"

            if captured_entry in top_entries:
                top_entries.remove(captured_entry)

            captured_color_cn = "红方" if top_color == "red" else "黑方"
            self.log.emit(
                f"检测到我方 bestmove 会吃掉棋盘上方颜色棋子："
                f"{captured_color_cn} {target_piece}@{to_row},{to_col}，"
                f"已从等待基准中扣除。"
            )

        return "|".join(top_entries)

    def apply_bestmove_to_fen(self, fen: str, bestmove: str) -> str | None:
        board = self.parse_fen_board_matrix(fen)
        parsed = self.parse_bestmove(bestmove)

        if board is None or parsed is None:
            return None

        from_row, from_col, to_row, to_col = parsed

        try:
            piece = board[from_row][from_col]
        except Exception:
            return None

        if not piece:
            return None

        board[from_row][from_col] = ""
        board[to_row][to_col] = piece

        return self.board_matrix_to_fen(board, self.fen_side_to_move(fen))

    @staticmethod
    def board_matrix_to_fen(board: list[list[str]], side_to_move: str) -> str:
        rows: list[str] = []

        for row in board:
            parts: list[str] = []
            empty_count = 0

            for piece in row:
                if not piece:
                    empty_count += 1
                    continue

                if empty_count:
                    parts.append(str(empty_count))
                    empty_count = 0

                parts.append(piece)

            if empty_count:
                parts.append(str(empty_count))

            rows.append("".join(parts) or "9")

        return f"{'/'.join(rows)} {side_to_move} - - 0 1"

    @staticmethod
    def fen_side_to_move(fen: str) -> str:
        parts = fen.strip().split()
        if len(parts) >= 2 and parts[1] in {"w", "b"}:
            return parts[1]
        return "w"

    @staticmethod
    def fen_board_part(fen: str) -> str:
        parts = fen.strip().split()
        if not parts:
            return ""
        return parts[0]

    @staticmethod
    def parse_fen_board_matrix(fen: str) -> list[list[str]] | None:
        board_part = AutoMoveWorker.fen_board_part(fen)
        rows = board_part.split("/")

        if len(rows) != 10:
            return None

        matrix: list[list[str]] = []

        for row_text in rows:
            row: list[str] = []

            for ch in row_text:
                if ch.isdigit():
                    row.extend([""] * int(ch))
                elif ch.isalpha():
                    row.append(ch)
                else:
                    return None

            if len(row) != 9:
                return None

            matrix.append(row)

        return matrix

    @staticmethod
    def piece_belongs_to_color(piece: str, color: str) -> bool:
        if not piece:
            return False

        if color == "red":
            return piece.isupper()

        if color == "black":
            return piece.islower()

        return False

    @staticmethod
    def color_piece_entries(fen: str, color: str) -> list[str]:
        board_part = AutoMoveWorker.fen_board_part(fen)
        rows = board_part.split("/")

        pieces: list[str] = []

        for row_index, row_text in enumerate(rows):
            col_index = 0

            for ch in row_text:
                if ch.isdigit():
                    col_index += int(ch)
                    continue

                is_red_piece = ch.isupper()
                is_black_piece = ch.islower()

                if color == "red" and is_red_piece:
                    pieces.append(f"{ch}@{row_index},{col_index}")
                elif color == "black" and is_black_piece:
                    pieces.append(f"{ch}@{row_index},{col_index}")

                col_index += 1

        return pieces

    @staticmethod
    def color_piece_fingerprint(fen: str, color: str) -> str:
        return "|".join(AutoMoveWorker.color_piece_entries(fen, color))

    def bestmove_to_screen_move(self, bestmove: str) -> ScreenMove | None:
        parsed = self.parse_bestmove(bestmove)
        if parsed is None:
            self.log.emit(f"自动走棋错误：bestmove 无法解析：{bestmove}")
            return None

        from_row, from_col, to_row, to_col = parsed

        legal_check = self.read_json("legal_check.json")
        if not legal_check:
            self.log.emit("自动走棋错误：找不到 legal_check.json")
            return None

        orientation = legal_check.get("orientation")

        if orientation == "black_bottom":
            from_screen_row = 9 - from_row
            from_screen_col = 8 - from_col
            to_screen_row = 9 - to_row
            to_screen_col = 8 - to_col
        else:
            from_screen_row = from_row
            from_screen_col = from_col
            to_screen_row = to_row
            to_screen_col = to_col

        points_payload = self.read_json("points.json")
        if not points_payload:
            self.log.emit("自动走棋错误：找不到 points.json")
            return None

        board_rect = self.read_json("board_rect.json")
        if not board_rect:
            self.log.emit("自动走棋错误：找不到 board_rect.json")
            return None

        from_point = self.find_point(points_payload, from_screen_row, from_screen_col)
        to_point = self.find_point(points_payload, to_screen_row, to_screen_col)

        if from_point is None or to_point is None:
            self.log.emit("自动走棋错误：points.json 中找不到起点或终点")
            return None

        try:
            board_x = float(board_rect["x"])
            board_y = float(board_rect["y"])

            from_x = int(round(board_x + float(from_point["x"])))
            from_y = int(round(board_y + float(from_point["y"])))
            to_x = int(round(board_x + float(to_point["x"])))
            to_y = int(round(board_y + float(to_point["y"])))
        except Exception as exc:
            self.log.emit(f"自动走棋错误：屏幕坐标计算失败，{exc}")
            return None

        return ScreenMove(
            bestmove=bestmove,
            from_x=from_x,
            from_y=from_y,
            to_x=to_x,
            to_y=to_y,
        )

    @staticmethod
    def parse_bestmove(bestmove: str) -> tuple[int, int, int, int] | None:
        bestmove = bestmove.strip()

        if len(bestmove) < 4:
            return None

        files = "abcdefghi"

        try:
            from_col = files.index(bestmove[0])
            from_row = 9 - int(bestmove[1])
            to_col = files.index(bestmove[2])
            to_row = 9 - int(bestmove[3])
        except Exception:
            return None

        if not (0 <= from_row <= 9 and 0 <= to_row <= 9):
            return None

        if not (0 <= from_col <= 8 and 0 <= to_col <= 8):
            return None

        return from_row, from_col, to_row, to_col

    @staticmethod
    def find_point(points_payload: Any, row: int, col: int) -> dict[str, Any] | None:
        if isinstance(points_payload, dict):
            points = points_payload.get("points", [])
        elif isinstance(points_payload, list):
            points = points_payload
        else:
            return None

        if not isinstance(points, list):
            return None

        for point in points:
            if isinstance(point, dict) and point.get("row") == row and point.get("col") == col:
                return point

        return None

    def click_move(self, move: ScreenMove) -> bool:
        if pyautogui is None:
            self.log.emit("自动走棋错误：pyautogui 未安装")
            return False

        try:
            pyautogui.click(move.from_x, move.from_y)
            self.sleep_with_stop(self.click_gap)

            if self.auto_stop_requested:
                return False

            pyautogui.click(move.to_x, move.to_y)
            return True

        except Exception as exc:
            self.log.emit(f"自动走棋错误：点击失败，{exc}")
            return False

    def read_text(self, filename: str) -> str | None:
        path = self.debug_dir / filename

        if not path.exists():
            return None

        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception:
            return None

        return text or None

    def read_json(self, filename: str) -> dict[str, Any] | None:
        path = self.debug_dir / filename

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        return data

    def sleep_with_stop(self, seconds: float) -> None:
        end_time = time.time() + seconds

        while time.time() < end_time:
            if self.auto_stop_requested:
                return
            time.sleep(0.05)

