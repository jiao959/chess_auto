from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app_paths import app_root, display_path


PROJECT_ROOT = app_root()
DEBUG_DIR = PROJECT_ROOT / "debug_outputs"
RECOGNITION_PATH = DEBUG_DIR / "piece_recognition_results.json"
FEN_PATH = DEBUG_DIR / "current_fen.txt"
LEGAL_CHECK_PATH = DEBUG_DIR / "legal_check.json"

ROWS = 10
COLS = 9
MIN_PIECE_CONFIDENCE = 0.50
ERROR_PREFIX = "模块四错误"

PIECE_TO_FEN = {
    "red_king": "K",
    "red_advisor": "A",
    "red_bishop": "B",
    "red_rook": "R",
    "red_knight": "N",
    "red_cannon": "C",
    "red_pawn": "P",
    "black_king": "k",
    "black_advisor": "a",
    "black_bishop": "b",
    "black_rook": "r",
    "black_knight": "n",
    "black_cannon": "c",
    "black_pawn": "p",
}

PIECE_LIMITS = {
    "red_king": 1,
    "red_advisor": 2,
    "red_bishop": 2,
    "red_rook": 2,
    "red_knight": 2,
    "red_cannon": 2,
    "red_pawn": 5,
    "black_king": 1,
    "black_advisor": 2,
    "black_bishop": 2,
    "black_rook": 2,
    "black_knight": 2,
    "black_cannon": 2,
    "black_pawn": 5,
}

PIECE_CN = {
    "red_king": "红帅",
    "red_advisor": "红仕",
    "red_bishop": "红相",
    "red_rook": "红车",
    "red_knight": "红马",
    "red_cannon": "红炮",
    "red_pawn": "红兵",
    "black_king": "黑将",
    "black_advisor": "黑士",
    "black_bishop": "黑象",
    "black_rook": "黑车",
    "black_knight": "黑马",
    "black_cannon": "黑炮",
    "black_pawn": "黑卒",
}


@dataclass
class LegalError:
    message: str
    row: int | None = None
    col: int | None = None
    piece: str | None = None


def load_recognition_results(path: Path = RECOGNITION_PATH) -> dict[str, Any] | None:
    if not path.exists():
        print(f"{ERROR_PREFIX}：未找到识别结果 {display_path(path)}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"{ERROR_PREFIX}：无法读取识别结果 {exc}")
        return None


def result_class(item: dict[str, Any]) -> str:
    value = item.get("final_class")
    if isinstance(value, str) and value:
        return value
    value = item.get("predicted_class")
    if isinstance(value, str) and value:
        return value
    return "unknown"


def build_matrix(payload: dict[str, Any]) -> tuple[list[list[str]], list[LegalError], list[LegalError]]:
    matrix = [["empty" for _ in range(COLS)] for _ in range(ROWS)]
    errors: list[LegalError] = []
    warnings: list[LegalError] = []
    seen: set[tuple[int, int]] = set()
    for item in payload.get("results", []):
        row = int(item.get("row", -1))
        col = int(item.get("col", -1))
        if not (0 <= row < ROWS and 0 <= col < COLS):
            errors.append(LegalError("点位坐标越界", row if row >= 0 else None, col if col >= 0 else None, None))
            continue
        if (row, col) in seen:
            errors.append(LegalError("点位重复", row, col, None))
            continue
        seen.add((row, col))

        piece = result_class(item)
        is_empty = bool(item.get("is_empty", piece == "empty"))
        confidence = float(item.get("confidence", 0.0))
        if is_empty or piece == "empty":
            matrix[row][col] = "empty"
            continue
        if piece == "unknown":
            matrix[row][col] = "empty"
            warnings.append(LegalError(f"r{row}c{col} 为 unknown", row, col, piece))
            continue
        if piece not in PIECE_TO_FEN:
            errors.append(LegalError(f"r{row}c{col} 类别非法", row, col, piece))
            continue
        if confidence < MIN_PIECE_CONFIDENCE:
            errors.append(LegalError(f"r{row}c{col} 置信度过低", row, col, piece))
            continue
        matrix[row][col] = piece

    if len(seen) != ROWS * COLS:
        errors.append(LegalError(f"点位数量不是 90，当前 {len(seen)}", None, None, None))
    return matrix, errors, warnings


def find_piece(matrix: list[list[str]], piece: str) -> list[tuple[int, int]]:
    return [(r, c) for r in range(ROWS) for c in range(COLS) if matrix[r][c] == piece]


def determine_orientation(matrix: list[list[str]]) -> tuple[str | None, str | None, str | None, list[LegalError]]:
    errors: list[LegalError] = []
    bottom_kings: list[tuple[str, int, int]] = []
    for row in range(7, 10):
        for col in range(COLS):
            if matrix[row][col] in {"red_king", "black_king"}:
                bottom_kings.append((matrix[row][col], row, col))

    if bottom_kings:
        piece, _, _ = bottom_kings[0]
        if piece == "red_king":
            return "red_bottom", "red", "w", errors
        return "black_bottom", "black", "b", errors

    red_positions = find_piece(matrix, "red_king")
    black_positions = find_piece(matrix, "black_king")
    if red_positions and black_positions:
        red_row = red_positions[0][0]
        black_row = black_positions[0][0]
        if red_row > black_row:
            return "red_bottom", "red", "w", errors
        if black_row > red_row:
            return "black_bottom", "black", "b", errors

    errors.append(LegalError("无法判断棋盘方向，未在棋盘下方找到将或帅", None, None, None))
    return None, None, None, errors


def rotate_180(matrix: list[list[str]]) -> list[list[str]]:
    return [[matrix[ROWS - 1 - row][COLS - 1 - col] for col in range(COLS)] for row in range(ROWS)]


def normalize_matrix(matrix: list[list[str]], orientation: str | None) -> list[list[str]]:
    if orientation == "black_bottom":
        return rotate_180(matrix)
    return [row[:] for row in matrix]


def validate_standard_matrix(matrix: list[list[str]]) -> list[LegalError]:
    errors: list[LegalError] = []
    counts = Counter(piece for row in matrix for piece in row if piece != "empty")

    for piece, limit in PIECE_LIMITS.items():
        count = counts.get(piece, 0)
        if count > limit:
            errors.append(LegalError(f"{PIECE_CN[piece]}出现 {count} 个，超过上限 {limit}", None, None, piece))

    for piece in ("red_king", "black_king"):
        positions = find_piece(matrix, piece)
        if not positions:
            errors.append(LegalError(f"{PIECE_CN[piece]}缺失", None, None, piece))
        elif len(positions) > 1:
            errors.append(LegalError(f"{PIECE_CN[piece]}出现 {len(positions)} 个", None, None, piece))

    total = sum(counts.values())
    if total > 32:
        errors.append(LegalError("棋子总数超过 32", None, None, None))

    for row in range(ROWS):
        for col in range(COLS):
            piece = matrix[row][col]
            if piece == "red_king" and not (7 <= row <= 9 and 3 <= col <= 5):
                errors.append(LegalError("红帅不在九宫内", row, col, piece))
            elif piece == "black_king" and not (0 <= row <= 2 and 3 <= col <= 5):
                errors.append(LegalError("黑将不在九宫内", row, col, piece))
            elif piece == "red_advisor" and not (7 <= row <= 9 and 3 <= col <= 5):
                errors.append(LegalError("红仕不在九宫内", row, col, piece))
            elif piece == "black_advisor" and not (0 <= row <= 2 and 3 <= col <= 5):
                errors.append(LegalError("黑士不在九宫内", row, col, piece))
            elif piece == "red_bishop" and not (5 <= row <= 9):
                errors.append(LegalError("红相过河", row, col, piece))
            elif piece == "black_bishop" and not (0 <= row <= 4):
                errors.append(LegalError("黑象过河", row, col, piece))

    red_king = find_piece(matrix, "red_king")
    black_king = find_piece(matrix, "black_king")
    if len(red_king) == 1 and len(black_king) == 1:
        red_row, red_col = red_king[0]
        black_row, black_col = black_king[0]
        if red_col == black_col:
            low = min(red_row, black_row) + 1
            high = max(red_row, black_row)
            blocked = any(matrix[row][red_col] != "empty" for row in range(low, high))
            if not blocked:
                errors.append(LegalError("将帅照面", None, red_col, None))
    return errors


def matrix_to_fen(matrix: list[list[str]], side_to_move: str) -> str:
    rows: list[str] = []
    for row in matrix:
        parts: list[str] = []
        empty_count = 0
        for piece in row:
            if piece == "empty":
                empty_count += 1
                continue
            if empty_count:
                parts.append(str(empty_count))
                empty_count = 0
            parts.append(PIECE_TO_FEN[piece])
        if empty_count:
            parts.append(str(empty_count))
        rows.append("".join(parts) or "9")
    return f"{'/'.join(rows)} {side_to_move} - - 0 1"


def write_legal_check(
    ok: bool,
    errors: list[LegalError],
    own_color: str | None,
    side_to_move: str | None,
    orientation: str | None,
    fen: str | None,
    warnings: list[LegalError] | None = None,
) -> None:
    payload = {
        "ok": ok,
        "errors": [asdict(error) for error in errors],
        "warnings": [asdict(warning) for warning in (warnings or [])],
        "own_color": own_color,
        "side_to_move": side_to_move,
        "orientation": orientation,
        "fen": fen,
    }
    LEGAL_CHECK_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fail(
    errors: list[LegalError],
    own_color: str | None = None,
    side_to_move: str | None = None,
    orientation: str | None = None,
    warnings: list[LegalError] | None = None,
) -> None:
    if FEN_PATH.exists():
        FEN_PATH.unlink()
    write_legal_check(False, errors, own_color, side_to_move, orientation, None, warnings)
    for error in errors:
        print(f"{ERROR_PREFIX}：{error.message}")


def print_warnings(warnings: list[LegalError]) -> None:
    for warning in warnings:
        print(f"模块四警告：{warning.message}")


def main() -> None:
    payload = load_recognition_results()
    if payload is None:
        if FEN_PATH.exists():
            FEN_PATH.unlink()
        write_legal_check(False, [LegalError("未找到或无法读取识别结果")], None, None, None, None)
        return

    matrix, errors, warnings = build_matrix(payload)
    print_warnings(warnings)
    if errors:
        fail(errors, warnings=warnings)
        return

    orientation, own_color, side_to_move, orientation_errors = determine_orientation(matrix)
    if orientation_errors:
        fail(orientation_errors, warnings=warnings)
        return

    standard_matrix = normalize_matrix(matrix, orientation)
    legal_errors = validate_standard_matrix(standard_matrix)
    if legal_errors:
        fail(legal_errors, own_color, side_to_move, orientation, warnings)
        return

    assert side_to_move is not None
    fen = matrix_to_fen(standard_matrix, side_to_move)
    FEN_PATH.write_text(fen + "\n", encoding="utf-8")
    write_legal_check(True, [], own_color, side_to_move, orientation, fen, warnings)
    print(f"当前 FEN：{fen}")
    print(f"己方颜色：{own_color}")
    print(f"当前走棋方：{side_to_move}")
    print("合法性检查：通过")


if __name__ == "__main__":
    main()
