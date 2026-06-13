from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mss
import numpy as np
from PIL import Image, ImageGrab

from app_paths import app_root


ERROR_MESSAGE = "模块一错误：未识别到棋局 / 未识别到棋盘区域"
PROJECT_ROOT = app_root()
DEBUG_DIR = PROJECT_ROOT / "debug_outputs"
FULL_SCREENSHOT_PATH = DEBUG_DIR / "full_screenshot.png"
BOARD_CROP_PATH = DEBUG_DIR / "board_crop.png"
BOARD_RECT_PATH = DEBUG_DIR / "board_rect.json"
OVERLAY_PATH = DEBUG_DIR / "board_detect_overlay.png"
DEFAULT_SAMPLE_PATH = Path(r"C:\Users\jh\Desktop\棋盘.png")


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h


def ensure_debug_dir() -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def remove_stale_outputs() -> None:
    for path in [BOARD_CROP_PATH, BOARD_RECT_PATH, OVERLAY_PATH]:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def capture_full_screen(output_path: str | Path) -> np.ndarray:
    output_path = Path(output_path)
    last_error: Exception | None = None

    try:
        with mss.MSS() as sct:
            monitor = sct.monitors[1]
            raw = sct.grab(monitor)
            bgra = np.array(raw, dtype=np.uint8)
            bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    except Exception as exc:
        last_error = exc
        print(f"mss 截图失败，改用 Pillow ImageGrab 兜底：{exc}")

        try:
            rgb_image = ImageGrab.grab(all_screens=False)
            rgb = np.array(rgb_image.convert("RGB"), dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as fallback_exc:
            raise RuntimeError(f"全屏截图失败：mss={last_error}; ImageGrab={fallback_exc}") from fallback_exc

    cv2.imwrite(str(output_path), bgr)
    return bgr


def load_reference_board(sample_path: str | Path) -> dict[str, Any]:
    sample_path = Path(sample_path)

    if not sample_path.exists():
        print(f"参考样本不存在，使用默认暖黄色范围：{sample_path}")
        return {
            "source": "fallback",
            "h_low": 8,
            "h_high": 38,
            "s_low": 25,
            "s_high": 230,
            "v_low": 190,
            "v_high": 255,
        }

    sample_bgr = cv2.imdecode(np.fromfile(str(sample_path), dtype=np.uint8), cv2.IMREAD_COLOR)

    if sample_bgr is None:
        print(f"参考样本读取失败，使用默认暖黄色范围：{sample_path}")
        return {
            "source": "fallback",
            "h_low": 8,
            "h_high": 38,
            "s_low": 25,
            "s_high": 230,
            "v_low": 190,
            "v_high": 255,
        }

    return extract_board_color_profile(sample_bgr)


def extract_board_color_profile(sample_bgr: np.ndarray) -> dict[str, Any]:
    hsv = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    valid = (v > 90) & (s > 20)

    if np.count_nonzero(valid) < 100:
        print("参考样本有效背景像素过少，使用默认暖黄色范围")
        return {
            "source": "fallback",
            "h_low": 8,
            "h_high": 38,
            "s_low": 25,
            "s_high": 230,
            "v_low": 190,
            "v_high": 255,
        }

    h_values = h[valid].astype(np.int16)
    s_values = s[valid].astype(np.int16)
    v_values = v[valid].astype(np.int16)

    h_med = int(np.median(h_values))
    s_low = int(max(15, np.percentile(s_values, 5) - 35))
    s_high = int(min(255, np.percentile(s_values, 98) + 55))
    v_low = 190
    v_high = int(min(255, np.percentile(v_values, 99) + 35))

    return {
        "source": "sample",
        "h_low": max(0, h_med - 18),
        "h_high": min(179, h_med + 18),
        "s_low": s_low,
        "s_high": s_high,
        "v_low": v_low,
        "v_high": v_high,
        "h_median": h_med,
    }


def build_board_color_mask(screen_bgr: np.ndarray, color_profile: dict[str, Any]) -> np.ndarray:
    hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(
        [color_profile["h_low"], color_profile["s_low"], color_profile["v_low"]],
        dtype=np.uint8,
    )
    upper = np.array(
        [color_profile["h_high"], color_profile["s_high"], color_profile["v_high"]],
        dtype=np.uint8,
    )

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)), iterations=1)

    return mask


def load_ignore_rects_from_env() -> list[Rect]:
    raw = os.environ.get("CHESS_IGNORE_RECTS", "").strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except Exception:
        return []

    rects: list[Rect] = []

    if not isinstance(data, list):
        return rects

    for item in data:
        try:
            x = int(item["x"])
            y = int(item["y"])
            w = int(item["w"])
            h = int(item["h"])

            if w > 0 and h > 0:
                rects.append(Rect(x, y, w, h))
        except Exception:
            continue

    return rects


def rect_intersection_area(a: Rect, b: Rect) -> int:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)

    if right <= left or bottom <= top:
        return 0

    return (right - left) * (bottom - top)


def should_ignore_candidate(rect: Rect, ignore_rects: list[Rect]) -> bool:
    for ignore in ignore_rects:
        inter = rect_intersection_area(rect, ignore)
        if inter <= 0:
            continue

        overlap_candidate = inter / max(rect.area, 1)

        if overlap_candidate >= 0.50:
            return True

        cx = rect.x + rect.w / 2
        cy = rect.y + rect.h / 2

        if ignore.x <= cx <= ignore.right and ignore.y <= cy <= ignore.bottom:
            return True

    return False


def find_color_candidates(screen_bgr: np.ndarray, mask: np.ndarray) -> list[dict[str, Any]]:
    screen_h, screen_w = screen_bgr.shape[:2]
    ignore_rects = load_ignore_rects_from_env()

    if ignore_rects:
        print(f"模块一：将忽略项目窗口区域数量：{len(ignore_rects)}")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rect = Rect(x, y, w, h)

        if should_ignore_candidate(rect, ignore_rects):
            print(f"忽略项目窗口内候选：x={rect.x}, y={rect.y}, w={rect.w}, h={rect.h}")
            continue

        if w < 200 or h < 200:
            continue

        area_ratio = rect.area / float(screen_w * screen_h)

        if area_ratio < 0.012:
            continue

        aspect = w / max(h, 1)

        if not 0.65 <= aspect <= 1.15:
            continue

        local_mask = mask[y : y + h, x : x + w]
        coverage = float(np.count_nonzero(local_mask) / max(local_mask.size, 1))

        if coverage < 0.18:
            continue

        candidates.append(
            {
                "rect": rect,
                "coverage": coverage,
                "area_ratio": area_ratio,
            }
        )

    candidates.sort(key=lambda item: item["rect"].area, reverse=True)
    return candidates


def detect_grid_lines(crop_bgr: np.ndarray) -> dict[str, Any]:
    h, w = crop_bgr.shape[:2]

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 55, 145)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(45, min(w, h) // 12),
        minLineLength=max(45, int(min(w, h) * 0.22)),
        maxLineGap=max(10, int(min(w, h) * 0.035)),
    )

    x_positions: list[float] = []
    y_positions: list[float] = []

    if lines is not None:
        for raw in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in raw]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dx <= max(6, w * 0.015) and dy >= h * 0.16:
                x_positions.append((x1 + x2) / 2)
            elif dy <= max(6, h * 0.015) and dx >= w * 0.25:
                y_positions.append((y1 + y2) / 2)

    x_lines = cluster_positions(x_positions, tolerance=max(6, w * 0.018))
    y_lines = cluster_positions(y_positions, tolerance=max(6, h * 0.018))

    return {
        "x_lines": x_lines,
        "y_lines": y_lines,
        "edge_count": 0 if lines is None else len(lines),
    }


def cluster_positions(values: list[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []

    for value in sorted(values):
        if not groups or abs(value - np.mean(groups[-1])) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)

    return [float(np.mean(group)) for group in groups]


def score_board_candidate(crop_bgr: np.ndarray) -> dict[str, Any]:
    h, w = crop_bgr.shape[:2]

    grid = detect_grid_lines(crop_bgr)
    x_lines = grid["x_lines"]
    y_lines = grid["y_lines"]

    x_count_score = count_score(len(x_lines), ideal=9, low=6, high=14)
    y_count_score = count_score(len(y_lines), ideal=10, low=7, high=16)

    x_spacing_score = spacing_score(x_lines, expected_count=9)
    y_spacing_score = spacing_score(y_lines, expected_count=10)

    aspect = w / max(h, 1)
    aspect_score = max(0.0, 1.0 - abs(aspect - 0.89) / 0.35)

    score = (
        x_count_score * 0.22
        + y_count_score * 0.22
        + x_spacing_score * 0.23
        + y_spacing_score * 0.23
        + aspect_score * 0.10
    )

    reason = (
        f"x_lines={len(x_lines)}, y_lines={len(y_lines)}, "
        f"x_spacing={x_spacing_score:.3f}, y_spacing={y_spacing_score:.3f}"
    )

    return {
        "score": float(score),
        "x_lines": x_lines,
        "y_lines": y_lines,
        "reason": reason,
        "edge_count": grid["edge_count"],
    }


def count_score(count: int, ideal: int, low: int, high: int) -> float:
    if count < low or count > high:
        return 0.0

    return max(0.0, 1.0 - abs(count - ideal) / max(ideal, 1))


def spacing_score(lines: list[float], expected_count: int) -> float:
    if len(lines) < max(4, expected_count // 2):
        return 0.0

    ordered = sorted(lines)
    best = 0.0

    for start_index in range(len(ordered)):
        for end_index in range(start_index + 1, len(ordered)):
            start = ordered[start_index]
            end = ordered[end_index]
            span = end - start

            if span <= 0:
                continue

            step = span / (expected_count - 1)

            if step < 18:
                continue

            model = [start + i * step for i in range(expected_count)]
            distances = [min(abs(line - candidate) for candidate in ordered) for line in model]

            hit_ratio = sum(distance <= step * 0.18 for distance in distances) / expected_count
            regularity = max(0.0, 1.0 - float(np.mean(distances)) / max(step * 0.35, 1.0))

            best = max(best, hit_ratio * 0.65 + regularity * 0.35)

    return best


def locate_board(screen_bgr: np.ndarray, sample_path: str | Path = DEFAULT_SAMPLE_PATH) -> dict[str, Any]:
    color_profile = load_reference_board(sample_path)
    print(f"颜色配置来源：{color_profile['source']}")

    mask = build_board_color_mask(screen_bgr, color_profile)
    candidates = find_color_candidates(screen_bgr, mask)

    print(f"检测到候选区域数量：{len(candidates)}")

    scored: list[dict[str, Any]] = []

    for index, item in enumerate(candidates):
        rect: Rect = item["rect"]
        crop = screen_bgr[rect.y : rect.bottom, rect.x : rect.right]
        score_data = score_board_candidate(crop)
        final_score = score_data["score"] * 0.82 + min(item["coverage"], 1.0) * 0.18
        refined = refine_rect_by_grid_lines(rect, score_data, screen_bgr.shape)

        scored_item = {
            **item,
            **score_data,
            "score": float(final_score),
            "refined_rect": refined,
        }

        scored.append(scored_item)

        print(
            f"候选 {index + 1}: x={rect.x}, y={rect.y}, w={rect.w}, h={rect.h}, "
            f"score={final_score:.3f}, 横线={len(score_data['y_lines'])}, 纵线={len(score_data['x_lines'])}, "
            f"{score_data['reason']}"
        )

    if not scored:
        return {
            "ok": False,
            "error": ERROR_MESSAGE,
        }

    best = max(scored, key=lambda item: item["score"])
    print(f"最佳候选得分：{best['score']:.3f}")

    if best["score"] < 0.48:
        return {
            "ok": False,
            "error": ERROR_MESSAGE,
            "best": best,
        }

    selected_rect = best["refined_rect"] or best["rect"]

    print(
        f"最终选择的棋盘区域坐标：x={selected_rect.x}, y={selected_rect.y}, "
        f"w={selected_rect.w}, h={selected_rect.h}"
    )

    return {
        "ok": True,
        "rect": selected_rect,
        "score": best["score"],
        "best": best,
        "candidates": scored,
    }


def refine_rect_by_grid_lines(candidate_rect: Rect, score_data: dict[str, Any], screen_shape: tuple[int, ...]) -> Rect | None:
    x_lines = sorted(score_data["x_lines"])
    y_lines = sorted(score_data["y_lines"])

    if len(x_lines) < 6 or len(y_lines) < 7:
        return None

    screen_h, screen_w = screen_shape[:2]
    margin = 8

    left = max(0, candidate_rect.x + int(round(min(x_lines))) - margin)
    right = min(screen_w, candidate_rect.x + int(round(max(x_lines))) + margin)
    top = max(0, candidate_rect.y + int(round(min(y_lines))) - margin)
    bottom = min(screen_h, candidate_rect.y + int(round(max(y_lines))) + margin)

    w = right - left
    h = bottom - top

    if w < 180 or h < 180:
        return None

    aspect = w / max(h, 1)

    if not 0.65 <= aspect <= 1.15:
        return None

    return Rect(left, top, w, h)


def save_board_outputs(screen_bgr: np.ndarray, result: dict[str, Any]) -> None:
    remove_stale_outputs()

    if not result.get("ok"):
        return

    rect: Rect = result["rect"]

    crop = screen_bgr[rect.y : rect.bottom, rect.x : rect.right]
    cv2.imwrite(str(BOARD_CROP_PATH), crop)

    BOARD_RECT_PATH.write_text(
        json.dumps(
            {
                "x": int(rect.x),
                "y": int(rect.y),
                "w": int(rect.w),
                "h": int(rect.h),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    overlay = screen_bgr.copy()
    cv2.rectangle(overlay, (rect.x, rect.y), (rect.right, rect.bottom), (0, 0, 255), 3)
    cv2.putText(
        overlay,
        f"score={result['score']:.3f}",
        (rect.x, max(25, rect.y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(OVERLAY_PATH), overlay)

    print(f"已保存棋盘裁剪图：{BOARD_CROP_PATH}")
    print(f"已保存棋盘裁剪框：{BOARD_RECT_PATH}")
    print(f"已保存识别叠加图：{OVERLAY_PATH}")


def main() -> None:
    ensure_debug_dir()
    print("开始模块一：全屏截图与棋盘区域自动识别")

    try:
        screen_bgr = capture_full_screen(FULL_SCREENSHOT_PATH)
    except Exception as exc:
        remove_stale_outputs()
        print(str(exc))
        print(ERROR_MESSAGE)
        return

    print(f"已保存全屏截图：{FULL_SCREENSHOT_PATH}")

    result = locate_board(screen_bgr, DEFAULT_SAMPLE_PATH)

    if not result.get("ok"):
        remove_stale_outputs()
        print(result.get("error", ERROR_MESSAGE))
        return

    save_board_outputs(screen_bgr, result)


if __name__ == "__main__":
    main()
