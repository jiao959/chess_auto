from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app_paths import app_root, resource_path


PROJECT_ROOT = app_root()
DEBUG_DIR = PROJECT_ROOT / "debug_outputs"
CONFIG_PATH = PROJECT_ROOT / "config.json"
DEFAULT_ENGINE_DIR = resource_path("engine")

FEN_PATH = DEBUG_DIR / "current_fen.txt"
LEGAL_CHECK_PATH = DEBUG_DIR / "legal_check.json"
BOARD_CROP_PATH = DEBUG_DIR / "board_crop.png"
BOARD_GEOMETRY_PATH = DEBUG_DIR / "board_geometry.json"

ENGINE_INPUT_FEN_PATH = DEBUG_DIR / "engine_input_fen.txt"
ENGINE_RAW_LOG_PATH = DEBUG_DIR / "engine_raw_log.txt"
BESTMOVE_PATH = DEBUG_DIR / "bestmove.txt"
BESTMOVE_CHINESE_PATH = DEBUG_DIR / "bestmove_chinese.txt"
ENGINE_RESULT_PATH = DEBUG_DIR / "engine_result.json"
VISUALIZATION_PATH = DEBUG_DIR / "bestmove_visualization.png"

ERROR_FEN_MISSING = "模块五错误：未找到 FEN 文件 debug_outputs/current_fen.txt"
ERROR_LEGAL_FAILED = "模块五错误：模块四合法性检查未通过，不能调用引擎"
ERROR_ENGINE_MISSING = "模块五错误：未找到 Pikafish 引擎"
ERROR_ENGINE_START = "模块五错误：Pikafish 引擎调用失败"
ERROR_ENGINE_EXIT = "模块五错误：引擎退出进程"
ERROR_NO_BESTMOVE = "模块五错误：提供的 FEN 错误，引擎无法判断"
ERROR_PARSE_BESTMOVE = "模块五错误：无法解析引擎 bestmove"

FEN_TO_PIECE = {
    "K": "red_king",
    "A": "red_advisor",
    "B": "red_bishop",
    "R": "red_rook",
    "N": "red_knight",
    "C": "red_cannon",
    "P": "red_pawn",
    "k": "black_king",
    "a": "black_advisor",
    "b": "black_bishop",
    "r": "black_rook",
    "n": "black_knight",
    "c": "black_cannon",
    "p": "black_pawn",
}

PIECE_NAME_CN = {
    "red_king": "帅",
    "red_advisor": "仕",
    "red_bishop": "相",
    "red_rook": "车",
    "red_knight": "马",
    "red_cannon": "炮",
    "red_pawn": "兵",
    "black_king": "将",
    "black_advisor": "士",
    "black_bishop": "象",
    "black_rook": "车",
    "black_knight": "马",
    "black_cannon": "炮",
    "black_pawn": "卒",
}

CN_NUM = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
DIRECT_PIECES = {"red_king", "red_rook", "red_cannon", "red_pawn", "black_king", "black_rook", "black_cannon", "black_pawn"}
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class EngineSettings:
    engine_path: str
    analysis_mode: str
    depth: int
    movetime: int
    hash: int
    threads: int


@dataclass
class Move:
    raw: str
    from_col: int
    from_row: int
    to_col: int
    to_row: int


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模块五：Pikafish 引擎分析")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--mode", choices=["depth", "movetime"], default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--movetime", type=int, default=None)
    parser.add_argument("--hash", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def resolve_engine_path(path_value: str | None) -> Path | None:
    path = Path(path_value) if path_value else DEFAULT_ENGINE_DIR
    if path.is_file():
        return path
    if path.is_dir():
        exes = sorted(path.glob("*.exe"))
        if exes:
            return exes[0]
    return None


def build_settings() -> EngineSettings | None:
    config = load_config()
    args = parse_args()
    engine_value = args.engine or config.get("engine_path") or str(DEFAULT_ENGINE_DIR)
    engine_path = resolve_engine_path(engine_value)
    if engine_path is None:
        return None
    mode = args.mode or config.get("analysis_mode", "depth")
    if mode not in {"depth", "movetime"}:
        mode = "depth"
    return EngineSettings(
        engine_path=str(engine_path),
        analysis_mode=mode,
        depth=int(args.depth if args.depth is not None else config.get("depth", 12)),
        movetime=int(args.movetime if args.movetime is not None else config.get("movetime", 1000)),
        hash=int(args.hash if args.hash is not None else config.get("hash", 128)),
        threads=int(args.threads if args.threads is not None else config.get("threads", 2)),
    )


def read_fen() -> str | None:
    if not FEN_PATH.exists():
        print(ERROR_FEN_MISSING)
        return None
    fen = FEN_PATH.read_text(encoding="utf-8").strip()
    if not fen:
        print(ERROR_FEN_MISSING)
        return None
    return fen


def legal_check_ok() -> bool:
    if not LEGAL_CHECK_PATH.exists():
        return True
    try:
        payload = json.loads(LEGAL_CHECK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return True
    if payload.get("ok") is False:
        print(ERROR_LEGAL_FAILED)
        return False
    return True


def send_line(proc: subprocess.Popen[str], line: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def read_until(proc: subprocess.Popen[str], expected: str, raw_log: list[str], timeout: float = 10.0) -> bool:
    assert proc.stdout is not None
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line == "":
            if proc.poll() is not None:
                return False
            continue
        line = line.rstrip("\n")
        raw_log.append(line)
        if expected in line:
            return True
    return False


def run_engine(fen: str, settings: EngineSettings) -> tuple[str | None, list[str], str | None]:
    raw_log: list[str] = []
    try:
        proc = subprocess.Popen(
            [settings.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=CREATE_NO_WINDOW,
        )
    except Exception:
        return None, raw_log, ERROR_ENGINE_START

    try:
        send_line(proc, "uci")
        if not read_until(proc, "uciok", raw_log, timeout=10.0):
            return None, raw_log, ERROR_ENGINE_EXIT if proc.poll() is not None else ERROR_NO_BESTMOVE

        send_line(proc, f"setoption name Hash value {settings.hash}")
        send_line(proc, f"setoption name Threads value {settings.threads}")
        send_line(proc, "isready")
        if not read_until(proc, "readyok", raw_log, timeout=10.0):
            return None, raw_log, ERROR_ENGINE_EXIT if proc.poll() is not None else ERROR_NO_BESTMOVE

        send_line(proc, f"position fen {fen}")
        if settings.analysis_mode == "movetime":
            send_line(proc, f"go movetime {settings.movetime}")
        else:
            send_line(proc, f"go depth {settings.depth}")

        assert proc.stdout is not None
        bestmove: str | None = None
        deadline = time.time() + 120.0
        while time.time() < deadline:
            line = proc.stdout.readline()
            if line == "":
                if proc.poll() is not None:
                    return None, raw_log, ERROR_ENGINE_EXIT
                continue
            line = line.rstrip("\n")
            raw_log.append(line)
            if line.startswith("bestmove "):
                parts = line.split()
                if len(parts) >= 2:
                    bestmove = parts[1]
                break
        if not bestmove or bestmove == "(none)":
            return None, raw_log, ERROR_NO_BESTMOVE
        return bestmove, raw_log, None
    finally:
        try:
            if proc.poll() is None:
                send_line(proc, "quit")
                proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def parse_fen_board(fen: str) -> list[list[str]] | None:
    board_part = fen.split()[0]
    rows = board_part.split("/")
    if len(rows) != 10:
        return None
    matrix: list[list[str]] = []
    for row in rows:
        out: list[str] = []
        for ch in row:
            if ch.isdigit():
                out.extend(["empty"] * int(ch))
            elif ch in FEN_TO_PIECE:
                out.append(FEN_TO_PIECE[ch])
            else:
                return None
        if len(out) != 9:
            return None
        matrix.append(out)
    return matrix


def parse_bestmove(bestmove: str) -> Move | None:
    if len(bestmove) < 4:
        return None
    files = "abcdefghi"
    try:
        from_col = files.index(bestmove[0])
        to_col = files.index(bestmove[2])
        from_row = 9 - int(bestmove[1])
        to_row = 9 - int(bestmove[3])
    except Exception:
        return None
    if not (0 <= from_row <= 9 and 0 <= to_row <= 9):
        return None
    return Move(bestmove[:4], from_col, from_row, to_col, to_row)


def file_num(piece: str, col: int) -> str:
    if piece.startswith("red_"):
        return CN_NUM[8 - col]
    return CN_NUM[col]


def step_num(step: int) -> str:
    if 1 <= step <= 9:
        return CN_NUM[step - 1]
    return str(step)


def move_to_chinese(matrix: list[list[str]], move: Move) -> str | None:
    piece = matrix[move.from_row][move.from_col]
    if piece == "empty":
        return None
    name = PIECE_NAME_CN.get(piece)
    if not name:
        return None
    first_num = file_num(piece, move.from_col)

    if move.to_col != move.from_col:
        action = "平"
        last = file_num(piece, move.to_col)
    else:
        if piece.startswith("red_"):
            action = "进" if move.to_row < move.from_row else "退"
        else:
            action = "进" if move.to_row > move.from_row else "退"
        if piece in DIRECT_PIECES:
            last = step_num(abs(move.to_row - move.from_row))
        else:
            last = file_num(piece, move.to_col)
    if piece not in DIRECT_PIECES and move.to_col != move.from_col:
        action = "进" if (move.to_row < move.from_row if piece.startswith("red_") else move.to_row > move.from_row) else "退"
        last = file_num(piece, move.to_col)
    return f"{name}{first_num}{action}{last}"


def load_board_geometry() -> dict[str, Any] | None:
    if not BOARD_GEOMETRY_PATH.exists():
        return None
    try:
        return json.loads(BOARD_GEOMETRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_orientation() -> str | None:
    if not LEGAL_CHECK_PATH.exists():
        return None
    try:
        return json.loads(LEGAL_CHECK_PATH.read_text(encoding="utf-8")).get("orientation")
    except Exception:
        return None


def standard_to_image_rc(row: int, col: int, orientation: str | None) -> tuple[int, int]:
    if orientation == "black_bottom":
        return 9 - row, 8 - col
    return row, col


def log_move_mapping(bestmove: str, move: Move, matrix: list[list[str]], orientation: str | None) -> None:
    piece = matrix[move.from_row][move.from_col]
    image_from_row, image_from_col = standard_to_image_rc(move.from_row, move.from_col, orientation)
    image_to_row, image_to_col = standard_to_image_rc(move.to_row, move.to_col, orientation)
    print(f"原始 bestmove：{bestmove}")
    print(f"FEN 起点：row={move.from_row}, col={move.from_col}")
    print(f"FEN 终点：row={move.to_row}, col={move.to_col}")
    print(f"FEN 起点棋子类别：{piece}")
    print(f"orientation：{orientation}")
    print(f"截图起点：row={image_from_row}, col={image_from_col}")
    print(f"截图终点：row={image_to_row}, col={image_to_col}")


def draw_visualization(bestmove_chinese: str, move: Move, orientation: str | None) -> bool:
    if not BOARD_CROP_PATH.exists() or not BOARD_GEOMETRY_PATH.exists():
        return False
    geometry = load_board_geometry()
    if geometry is None:
        return False
    board = Image.open(BOARD_CROP_PATH).convert("RGB")
    top_h = 72
    canvas = Image.new("RGB", (board.width, board.height + top_h), "white")
    canvas.paste(board, (0, top_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("simhei.ttf", 34)
    except Exception:
        pass
    text_box = draw.textbbox((0, 0), bestmove_chinese, font=font)
    text_w = text_box[2] - text_box[0]
    draw.text(((board.width - text_w) / 2, 18), bestmove_chinese, fill=(20, 20, 20), font=font)

    from_row, from_col = standard_to_image_rc(move.from_row, move.from_col, orientation)
    to_row, to_col = standard_to_image_rc(move.to_row, move.to_col, orientation)
    left = float(geometry["left"])
    top = float(geometry["top"])
    cell_w = float(geometry["cell_w"])
    cell_h = float(geometry["cell_h"])
    x1 = left + from_col * cell_w
    y1 = top_h + top + from_row * cell_h
    x2 = left + to_col * cell_w
    y2 = top_h + top + to_row * cell_h
    draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=6)
    draw_arrow_head(draw, x1, y1, x2, y2)
    canvas.save(VISUALIZATION_PATH)
    return True


def draw_arrow_head(draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float) -> None:
    angle = np.arctan2(y2 - y1, x2 - x1)
    length = 22
    spread = np.pi / 7
    p1 = (x2 - length * np.cos(angle - spread), y2 - length * np.sin(angle - spread))
    p2 = (x2 - length * np.cos(angle + spread), y2 - length * np.sin(angle + spread))
    draw.polygon([(x2, y2), p1, p2], fill=(255, 0, 0))


def write_result(ok: bool, fen: str | None, settings: EngineSettings | None, bestmove: str | None, bestmove_chinese: str | None, error: str | None) -> None:
    payload = {
        "ok": ok,
        "fen": fen,
        "analysis_mode": settings.analysis_mode if settings else None,
        "depth": settings.depth if settings else None,
        "movetime": settings.movetime if settings else None,
        "hash": settings.hash if settings else None,
        "threads": settings.threads if settings else None,
        "bestmove": bestmove,
        "bestmove_chinese": bestmove_chinese,
        "raw_log_path": str(ENGINE_RAW_LOG_PATH),
        "error": error,
    }
    ENGINE_RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fail(error: str, fen: str | None = None, settings: EngineSettings | None = None, raw_log: list[str] | None = None) -> None:
    if raw_log is not None:
        ENGINE_RAW_LOG_PATH.write_text("\n".join(raw_log) + ("\n" if raw_log else ""), encoding="utf-8")
    write_result(False, fen, settings, None, None, error)
    print(error)


def main() -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    fen = read_fen()
    if fen is None:
        write_result(False, None, None, None, None, ERROR_FEN_MISSING)
        return
    print(f"已读取 FEN：{fen}")
    if not legal_check_ok():
        write_result(False, fen, None, None, None, ERROR_LEGAL_FAILED)
        return
    settings = build_settings()
    if settings is None:
        fail(ERROR_ENGINE_MISSING, fen)
        return
    print(f"引擎路径：{settings.engine_path}")
    print(f"分析模式：{settings.analysis_mode}")
    print(f"Hash：{settings.hash}")
    print(f"Threads：{settings.threads}")
    print(f"depth：{settings.depth}" if settings.analysis_mode == "depth" else f"movetime：{settings.movetime}")

    ENGINE_INPUT_FEN_PATH.write_text(fen + "\n", encoding="utf-8")
    bestmove, raw_log, error = run_engine(fen, settings)
    ENGINE_RAW_LOG_PATH.write_text("\n".join(raw_log) + ("\n" if raw_log else ""), encoding="utf-8")
    if error:
        fail(error, fen, settings, raw_log)
        return
    assert bestmove is not None

    move = parse_bestmove(bestmove)
    if move is None:
        fail(ERROR_PARSE_BESTMOVE, fen, settings, raw_log)
        return
    matrix = parse_fen_board(fen)
    if matrix is None or matrix[move.from_row][move.from_col] == "empty":
        fail(ERROR_NO_BESTMOVE, fen, settings, raw_log)
        return
    orientation = load_orientation()
    log_move_mapping(bestmove, move, matrix, orientation)
    bestmove_chinese = move_to_chinese(matrix, move)
    if bestmove_chinese is None:
        fail(ERROR_PARSE_BESTMOVE, fen, settings, raw_log)
        return

    BESTMOVE_PATH.write_text(bestmove + "\n", encoding="utf-8")
    BESTMOVE_CHINESE_PATH.write_text(bestmove_chinese + "\n", encoding="utf-8")
    draw_visualization(bestmove_chinese, move, orientation)
    write_result(True, fen, settings, bestmove, bestmove_chinese, None)
    print(f"bestmove: {bestmove}")
    print(f"中文走法：{bestmove_chinese}")
    print(f"输出文件：{ENGINE_INPUT_FEN_PATH}")
    print(f"输出文件：{ENGINE_RAW_LOG_PATH}")
    print(f"输出文件：{BESTMOVE_PATH}")
    print(f"输出文件：{BESTMOVE_CHINESE_PATH}")
    print(f"输出文件：{ENGINE_RESULT_PATH}")
    print(f"输出文件：{VISUALIZATION_PATH}")


if __name__ == "__main__":
    main()
