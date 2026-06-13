from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app_paths import app_root, display_path


PROJECT_ROOT = app_root()
DEBUG_DIR = PROJECT_ROOT / "debug_outputs"
BOARD_CROP_PATH = DEBUG_DIR / "board_crop.png"
POINT_CROPS_DIR = DEBUG_DIR / "point_crops"
PREVIEW_PATH = DEBUG_DIR / "point_crops_preview.png"
OVERLAY_PATH = DEBUG_DIR / "points_overlay.png"
GEOMETRY_PATH = DEBUG_DIR / "board_geometry.json"
POINTS_JSON_PATH = DEBUG_DIR / "points.json"

ROWS = 10
COLS = 9


@dataclass(frozen=True)
class AxisFit:
    positions: list[float]
    step: float
    score: float
    hits: int
    mean_error: float


@dataclass(frozen=True)
class BoardGeometry:
    source: str
    method: str
    left: float
    top: float
    right: float
    bottom: float
    cell_w: float
    cell_h: float
    crop_radius: int
    rows: int
    cols: int
    score: float


@dataclass(frozen=True)
class PointInfo:
    row: int
    col: int

    # 理论棋盘交点：用于 FEN、走法箭头、自动点击。
    grid_x: float
    grid_y: float

    # 识别裁剪中心：当前不做棋子圆心修正，默认等于理论交点。
    crop_x: float
    crop_y: float

    crop_path: str
    crop_box: dict[str, int]

    @property
    def x(self) -> float:
        # 兼容旧模块：x 表示理论棋盘交点。
        return self.grid_x

    @property
    def y(self) -> float:
        # 兼容旧模块：y 表示理论棋盘交点。
        return self.grid_y


@dataclass(frozen=True)
class ControlPoint:
    row: int
    col: int
    x: float
    y: float
    score: float


def ensure_output_dirs() -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    if POINT_CROPS_DIR.exists():
        shutil.rmtree(POINT_CROPS_DIR)
    POINT_CROPS_DIR.mkdir(parents=True, exist_ok=True)


def read_board_crop(path: Path = BOARD_CROP_PATH) -> np.ndarray | None:
    if not path.exists():
        print(f"模块二错误：未找到棋盘裁剪图 {display_path(path)}")
        return None

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        print("模块二错误：无法读取棋盘裁剪图")
        return None

    print(f"成功读取棋盘裁剪图：{path}，尺寸：{image.shape[1]}x{image.shape[0]}")
    return image


def write_image_unicode(path: str | Path, image_bgr: np.ndarray) -> bool:
    """
    避免 Windows 中文路径下 cv2.imwrite 偶发失败。
    """
    path = Path(path)
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def detect_hough_segments(board_bgr: np.ndarray) -> dict[str, Any]:
    h, w = board_bgr.shape[:2]

    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 55, 145)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(45, min(w, h) // 13),
        minLineLength=max(35, int(min(w, h) * 0.08)),
        maxLineGap=max(10, int(min(w, h) * 0.035)),
    )

    horizontal: list[tuple[float, float]] = []
    vertical: list[tuple[float, float]] = []
    total = 0 if lines is None else len(lines)

    if lines is not None:
        for raw in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in raw]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = float(np.hypot(dx, dy))

            if dy <= max(5, h * 0.010) and dx >= max(35, w * 0.07):
                horizontal.append(((y1 + y2) / 2.0, length))
            elif dx <= max(5, w * 0.010) and dy >= max(35, h * 0.07):
                vertical.append(((x1 + x2) / 2.0, length))

    print(f"检测到 Hough 线段：{total}")
    print(f"接近水平线段数量：{len(horizontal)}")
    print(f"接近垂直线段数量：{len(vertical)}")

    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "total": total,
    }


def cluster_line_positions(
    lines: list[tuple[float, float]],
    tolerance: float,
    max_count: int,
    axis: str,
    extent: int,
) -> list[float]:
    if not lines:
        return []

    ordered = sorted(lines, key=lambda item: item[0])
    groups: list[list[tuple[float, float]]] = []

    for pos, length in ordered:
        if not groups:
            groups.append([(pos, length)])
            continue

        group_mean = np.average(
            [item[0] for item in groups[-1]],
            weights=[item[1] for item in groups[-1]],
        )

        if abs(pos - group_mean) <= tolerance:
            groups[-1].append((pos, length))
        else:
            groups.append([(pos, length)])

    clusters: list[tuple[float, float]] = []

    for group in groups:
        positions = [item[0] for item in group]
        weights = [item[1] for item in group]

        weighted_pos = float(np.average(positions, weights=weights))
        total_weight = float(sum(weights))

        group_min = float(min(positions))
        group_max = float(max(positions))
        group_center = (group_min + group_max) / 2.0

        near_start_edge = group_center < extent * 0.18
        near_end_edge = group_center > extent * 0.82

        if near_start_edge and len(group) >= 2:
            cluster_pos = group_max
        elif near_end_edge and len(group) >= 2:
            cluster_pos = group_min
        else:
            cluster_pos = weighted_pos

        clusters.append((cluster_pos, total_weight))

    if len(clusters) > max_count:
        clusters = sorted(clusters, key=lambda item: item[1], reverse=True)[:max_count]

    return sorted(position for position, _ in clusters)


def fit_even_axis(detected: list[float], count: int, extent: int) -> AxisFit | None:
    detected = sorted(float(v) for v in detected if 0 <= v <= extent)

    if len(detected) < 2:
        return None

    min_step = extent * 0.065
    max_step = extent * 0.145
    best: AxisFit | None = None
    visited: set[tuple[int, int]] = set()

    for i, a in enumerate(detected):
        for b in detected[i + 1:]:
            gap = b - a

            for k in range(1, count):
                step = gap / k

                if min_step <= step <= max_step:
                    for offset in range(count):
                        start = a - offset * step
                        key = (round(start * 10), round(step * 10))

                        if key in visited:
                            continue

                        visited.add(key)

                        fit = score_axis_candidate(start, step, detected, count, extent)

                        if fit is not None and (best is None or fit.score > best.score):
                            best = fit

    return best


def fit_axis_with_fixed_step(
    detected: list[float],
    count: int,
    extent: int,
    fixed_step: float,
) -> AxisFit | None:
    detected = sorted(float(v) for v in detected if 0 <= v <= extent)

    if not detected:
        return None

    full_length = (count - 1) * fixed_step

    if full_length <= 0 or full_length > extent * 1.10:
        return None

    starts: list[float] = []

    for line in detected:
        for offset in range(count):
            starts.append(line - offset * fixed_step)

    expected_start = extent * 0.06
    starts.append(expected_start)
    starts.append((extent - full_length) / 2.0)
    starts.append(extent * 0.064)

    best: AxisFit | None = None
    visited: set[int] = set()

    for start in starts:
        key = round(start * 10)

        if key in visited:
            continue

        visited.add(key)

        fit = score_axis_candidate(start, fixed_step, detected, count, extent)

        if fit is not None and (best is None or fit.score > best.score):
            best = fit

    return best


def score_axis_candidate(
    start: float,
    step: float,
    detected: list[float],
    count: int,
    extent: int,
) -> AxisFit | None:
    end = start + (count - 1) * step

    if start < -extent * 0.04 or end > extent * 1.04:
        return None

    positions = [start + idx * step for idx in range(count)]

    if detected:
        errors = [min(abs(pos - line) for line in detected) for pos in positions]
    else:
        errors = [step] * count

    hit_limit = max(4.0, step * 0.16)
    hits = sum(error <= hit_limit for error in errors)
    mean_error = float(np.mean([min(error, hit_limit * 2) for error in errors]))

    hit_score = hits / count
    error_score = max(0.0, 1.0 - mean_error / max(hit_limit, 1.0))
    bounds_score = 1.0 if 0 <= start <= extent and 0 <= end <= extent else 0.65

    expected_start = extent * 0.06
    edge_bias = 1.0 - min(abs(start - expected_start) / max(extent * 0.22, 1.0), 1.0) * 0.08

    score = hit_score * 0.62 + error_score * 0.28 + bounds_score * 0.08 + edge_bias * 0.02

    return AxisFit(
        positions=positions,
        step=step,
        score=float(score),
        hits=hits,
        mean_error=mean_error,
    )


def choose_equal_cell_geometry(
    x_positions: list[float],
    y_positions: list[float],
    x_fit: AxisFit,
    y_fit: AxisFit,
    w: int,
    h: int,
) -> BoardGeometry | None:
    candidate_steps: list[float] = [
        x_fit.step,
        y_fit.step,
        (x_fit.step + y_fit.step) / 2.0,
    ]

    if x_fit.score >= y_fit.score or x_fit.hits >= y_fit.hits:
        candidate_steps.append(x_fit.step)
    else:
        candidate_steps.append(y_fit.step)

    best: tuple[float, AxisFit, AxisFit, float] | None = None
    visited: set[int] = set()

    for base_step in candidate_steps:
        for scale in np.linspace(0.985, 1.015, 7):
            step = float(base_step * scale)
            key = round(step * 20)

            if key in visited:
                continue

            visited.add(key)

            x_fixed = fit_axis_with_fixed_step(x_positions, COLS, w, step)
            y_fixed = fit_axis_with_fixed_step(y_positions, ROWS, h, step)

            if x_fixed is None or y_fixed is None:
                continue

            board_ratio = (x_fixed.positions[-1] - x_fixed.positions[0]) / max(
                y_fixed.positions[-1] - y_fixed.positions[0],
                1.0,
            )
            ratio_score = max(0.0, 1.0 - abs(board_ratio - (8 / 9)) / 0.18)

            x_edge_ok = 0 <= x_fixed.positions[0] <= w and x_fixed.positions[-1] <= w
            y_edge_ok = 0 <= y_fixed.positions[0] <= h and y_fixed.positions[-1] <= h
            edge_score = 1.0 if x_edge_ok and y_edge_ok else 0.65

            combined = (
                x_fixed.score * 0.38
                + y_fixed.score * 0.38
                + ratio_score * 0.16
                + edge_score * 0.08
            )

            if best is None or combined > best[0]:
                best = (combined, x_fixed, y_fixed, step)

    if best is None:
        return None

    combined, best_x, best_y, common_step = best

    if combined < 0.44:
        return None

    crop_radius = int(round(common_step * 0.48))

    return BoardGeometry(
        source=str(BOARD_CROP_PATH),
        method="hough_even_axis_equal_cell",
        left=best_x.positions[0],
        top=best_y.positions[0],
        right=best_x.positions[-1],
        bottom=best_y.positions[-1],
        cell_w=common_step,
        cell_h=common_step,
        crop_radius=max(18, crop_radius),
        rows=ROWS,
        cols=COLS,
        score=float(combined),
    )


def get_rough_geometry_from_hough(
    board_bgr: np.ndarray,
    x_positions: list[float],
    y_positions: list[float],
) -> BoardGeometry:
    h, w = board_bgr.shape[:2]

    x_fit = fit_even_axis(x_positions, COLS, w)
    y_fit = fit_even_axis(y_positions, ROWS, h)

    if x_fit is not None and y_fit is not None:
        print(
            f"x_fit step={x_fit.step:.2f}, score={x_fit.score:.3f}, "
            f"hits={x_fit.hits}, mean_error={x_fit.mean_error:.2f}"
        )
        print(
            f"y_fit step={y_fit.step:.2f}, score={y_fit.score:.3f}, "
            f"hits={y_fit.hits}, mean_error={y_fit.mean_error:.2f}"
        )

        geometry = choose_equal_cell_geometry(x_positions, y_positions, x_fit, y_fit, w, h)

        if geometry is not None:
            print("粗定位：已启用等格距约束")
            return geometry

        cell_ratio = min(x_fit.step, y_fit.step) / max(x_fit.step, y_fit.step)
        board_ratio = (x_fit.positions[-1] - x_fit.positions[0]) / max(
            y_fit.positions[-1] - y_fit.positions[0],
            1.0,
        )
        ratio_score = max(0.0, 1.0 - abs(board_ratio - (8 / 9)) / 0.22)
        combined = x_fit.score * 0.34 + y_fit.score * 0.34 + cell_ratio * 0.17 + ratio_score * 0.15

        if combined >= 0.46 and cell_ratio >= 0.78:
            crop_radius = int(round(min(x_fit.step, y_fit.step) * 0.48))

            return BoardGeometry(
                source=str(BOARD_CROP_PATH),
                method="hough_even_axis",
                left=x_fit.positions[0],
                top=y_fit.positions[0],
                right=x_fit.positions[-1],
                bottom=y_fit.positions[-1],
                cell_w=x_fit.step,
                cell_h=y_fit.step,
                crop_radius=max(18, crop_radius),
                rows=ROWS,
                cols=COLS,
                score=float(combined),
            )

    print("粗定位：Hough 等间距拟合失败，使用 fallback_margin")
    return fallback_geometry(board_bgr)


def weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        raise ValueError("weighted_median received empty values")

    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    total = float(sum(max(0.0, w) for _, w in pairs))

    if total <= 0:
        return float(np.median(values))

    acc = 0.0
    half = total / 2.0

    for value, weight in pairs:
        acc += max(0.0, weight)
        if acc >= half:
            return float(value)

    return float(pairs[-1][0])


def crop_array_safe(image: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray | None:
    h, w = image.shape[:2]
    left = cx - radius
    top = cy - radius
    right = cx + radius + 1
    bottom = cy + radius + 1

    if left < 0 or top < 0 or right > w or bottom > h:
        return None

    return image[top:bottom, left:right]


def score_cross_at(gray: np.ndarray, edges: np.ndarray, cx: int, cy: int, cell: float) -> float:
    """
    判断某个局部位置是否像“清晰棋盘交叉点”。

    特征：
    1. 中心水平细条和垂直细条比背景更暗。
    2. 水平、垂直方向都有边缘响应。
    3. 整个局部区域不能过于复杂，避免棋子文字、棋子阴影、楚河汉界文字误当控制点。
    """
    h, w = gray.shape[:2]

    radius = max(10, int(round(cell * 0.30)))
    arm = max(8, int(round(cell * 0.22)))
    line_half = max(1, int(round(cell * 0.018)))

    if cx - radius < 0 or cy - radius < 0 or cx + radius >= w or cy + radius >= h:
        return -1.0

    patch = gray[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
    edge_patch = edges[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]

    center = radius

    h_strip = patch[
        center - line_half:center + line_half + 1,
        center - arm:center + arm + 1,
    ]
    v_strip = patch[
        center - arm:center + arm + 1,
        center - line_half:center + line_half + 1,
    ]

    h_edge = edge_patch[
        center - line_half:center + line_half + 1,
        center - arm:center + arm + 1,
    ]
    v_edge = edge_patch[
        center - arm:center + arm + 1,
        center - line_half:center + line_half + 1,
    ]

    mask = np.ones(patch.shape, dtype=bool)
    mask[
        center - line_half:center + line_half + 1,
        center - arm:center + arm + 1,
    ] = False
    mask[
        center - arm:center + arm + 1,
        center - line_half:center + line_half + 1,
    ] = False

    background = patch[mask]

    if background.size == 0:
        return -1.0

    bg_mean = float(np.mean(background))
    h_mean = float(np.mean(h_strip))
    v_mean = float(np.mean(v_strip))

    h_contrast = max(0.0, (bg_mean - h_mean) / 55.0)
    v_contrast = max(0.0, (bg_mean - v_mean) / 55.0)

    h_edge_score = float(np.mean(h_edge > 0))
    v_edge_score = float(np.mean(v_edge > 0))
    full_edge_score = float(np.mean(edge_patch > 0))

    # 棋子、文字、阴影会让整块 patch 变复杂。
    dark_relative_ratio = float(np.mean(patch < bg_mean - 28.0))
    dark_absolute_ratio = float(np.mean(patch < 105))

    complexity_penalty = 0.0
    complexity_penalty += max(0.0, dark_relative_ratio - 0.24) * 1.8
    complexity_penalty += max(0.0, dark_absolute_ratio - 0.10) * 1.2
    complexity_penalty += max(0.0, full_edge_score - 0.18) * 0.9

    # 两个方向都要有响应，避免单线或文字笔画误检。
    balance = min(h_contrast, v_contrast) / max(max(h_contrast, v_contrast), 1e-6)
    edge_balance = min(h_edge_score, v_edge_score) / max(max(h_edge_score, v_edge_score), 1e-6)

    score = (
        0.36 * h_contrast
        + 0.36 * v_contrast
        + 0.10 * h_edge_score
        + 0.10 * v_edge_score
        + 0.05 * balance
        + 0.03 * edge_balance
        - complexity_penalty
    )

    return float(score)


def search_clear_cross_near(
    gray: np.ndarray,
    edges: np.ndarray,
    expected_x: float,
    expected_y: float,
    cell: float,
) -> tuple[float, float, float] | None:
    """
    在粗网格交点附近小范围搜索最像清晰交叉点的位置。
    """
    h, w = gray.shape[:2]

    search_radius = max(4, int(round(cell * 0.12)))
    step = max(1, int(round(cell * 0.025)))

    base_x = int(round(expected_x))
    base_y = int(round(expected_y))

    if not (0 <= base_x < w and 0 <= base_y < h):
        return None

    best_score = -1.0
    best_x = base_x
    best_y = base_y

    for dy in range(-search_radius, search_radius + 1, step):
        y = base_y + dy

        if y < 0 or y >= h:
            continue

        for dx in range(-search_radius, search_radius + 1, step):
            x = base_x + dx

            if x < 0 or x >= w:
                continue

            score = score_cross_at(gray, edges, x, y, cell)

            if score > best_score:
                best_score = score
                best_x = x
                best_y = y

    # 阈值不能太高，否则空棋盘纹理较浅时会检测不到。
    if best_score < 0.16:
        return None

    return float(best_x), float(best_y), float(best_score)


def detect_clear_control_points(
    board_bgr: np.ndarray,
    rough_geometry: BoardGeometry,
) -> list[ControlPoint]:
    """
    真正的控制点：
    以粗网格为索引，只在理论交点附近寻找“清晰棋盘十字交叉点”。

    不是对 Hough 线位置两两相减。
    每个控制点都带 row/col，因此后面可以用：
    cell = abs(x1 - x2) / abs(col1 - col2)
    cell = abs(y1 - y2) / abs(row1 - row2)
    """
    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 135)

    cell = float((rough_geometry.cell_w + rough_geometry.cell_h) / 2.0)

    controls: list[ControlPoint] = []

    for row in range(ROWS):
        for col in range(COLS):
            expected_x = rough_geometry.left + col * rough_geometry.cell_w
            expected_y = rough_geometry.top + row * rough_geometry.cell_h

            found = search_clear_cross_near(
                gray=gray,
                edges=edges,
                expected_x=expected_x,
                expected_y=expected_y,
                cell=cell,
            )

            if found is None:
                continue

            x, y, score = found

            controls.append(
                ControlPoint(
                    row=row,
                    col=col,
                    x=x,
                    y=y,
                    score=score,
                )
            )

    print(f"清晰棋盘交叉控制点数量：{len(controls)}")
    return controls


def estimate_geometry_from_control_points(
    controls: list[ControlPoint],
    rough_geometry: BoardGeometry,
    w: int,
    h: int,
) -> BoardGeometry | None:
    """
    用带 row/col 的真实交叉控制点估计网格。

    核心：
    1. 方格边长 cell 来自控制点之间的真实距离 / 行列跨度。
    2. left/top 来自控制点反推：
       left = x - col * cell
       top  = y - row * cell
    """
    if len(controls) < 5:
        print("控制点校正失败：可用控制点少于 5 个")
        return None

    rough_cell = float((rough_geometry.cell_w + rough_geometry.cell_h) / 2.0)
    min_cell = rough_cell * 0.88
    max_cell = rough_cell * 1.12

    cell_values: list[float] = []
    cell_weights: list[float] = []

    for i, a in enumerate(controls):
        for b in controls[i + 1:]:
            if a.row == b.row:
                dc = abs(a.col - b.col)
                if dc >= 2:
                    value = abs(a.x - b.x) / dc
                    if min_cell <= value <= max_cell:
                        weight = ((a.score + b.score) / 2.0) * min(dc, 6)
                        cell_values.append(float(value))
                        cell_weights.append(float(weight))

            if a.col == b.col:
                dr = abs(a.row - b.row)
                if dr >= 2:
                    value = abs(a.y - b.y) / dr
                    if min_cell <= value <= max_cell:
                        weight = ((a.score + b.score) / 2.0) * min(dr, 7)
                        cell_values.append(float(value))
                        cell_weights.append(float(weight))

    if len(cell_values) < 4:
        print("控制点校正失败：有效 cell 候选少于 4 个")
        return None

    cell = weighted_median(cell_values, cell_weights)

    left_values: list[float] = []
    left_weights: list[float] = []
    top_values: list[float] = []
    top_weights: list[float] = []

    for p in controls:
        left_values.append(p.x - p.col * cell)
        left_weights.append(max(0.01, p.score))

        top_values.append(p.y - p.row * cell)
        top_weights.append(max(0.01, p.score))

    left = weighted_median(left_values, left_weights)
    top = weighted_median(top_values, top_weights)
    right = left + (COLS - 1) * cell
    bottom = top + (ROWS - 1) * cell

    if left < -cell * 0.40 or top < -cell * 0.40 or right > w + cell * 0.40 or bottom > h + cell * 0.40:
        print("控制点校正失败：校正后的棋盘边界越界")
        return None

    residuals: list[float] = []

    for p in controls:
        expected_x = left + p.col * cell
        expected_y = top + p.row * cell
        residual = float(np.hypot(expected_x - p.x, expected_y - p.y))
        residuals.append(residual)

    mean_residual = float(np.mean(residuals)) if residuals else cell
    residual_score = max(0.0, 1.0 - mean_residual / max(cell * 0.12, 1.0))
    support_score = min(1.0, len(controls) / 18.0)

    score = residual_score * 0.70 + support_score * 0.30

    if score < 0.42:
        print(
            f"控制点校正失败：score={score:.3f}, "
            f"mean_residual={mean_residual:.2f}, cell={cell:.2f}"
        )
        return None

    print(
        f"控制点校正成功：cell={cell:.2f}, "
        f"mean_residual={mean_residual:.2f}, score={score:.3f}"
    )

    crop_radius = int(round(cell * 0.48))

    return BoardGeometry(
        source=str(BOARD_CROP_PATH),
        method="clear_intersection_control_points",
        left=float(left),
        top=float(top),
        right=float(right),
        bottom=float(bottom),
        cell_w=float(cell),
        cell_h=float(cell),
        crop_radius=max(18, crop_radius),
        rows=ROWS,
        cols=COLS,
        score=float(score),
    )


def refine_geometry_by_control_points(
    board_bgr: np.ndarray,
    rough_geometry: BoardGeometry,
) -> BoardGeometry | None:
    h, w = board_bgr.shape[:2]

    controls = detect_clear_control_points(board_bgr, rough_geometry)
    refined = estimate_geometry_from_control_points(controls, rough_geometry, w, h)

    return refined


def locate_geometry(board_bgr: np.ndarray) -> BoardGeometry:
    h, w = board_bgr.shape[:2]

    segments = detect_hough_segments(board_bgr)

    x_positions = cluster_line_positions(
        segments["vertical"],
        tolerance=max(5, w * 0.014),
        max_count=16,
        axis="x",
        extent=w,
    )

    y_positions = cluster_line_positions(
        segments["horizontal"],
        tolerance=max(5, h * 0.014),
        max_count=18,
        axis="y",
        extent=h,
    )

    print(f"聚类后的 x 位置数量：{len(x_positions)}")
    print(f"聚类后的 y 位置数量：{len(y_positions)}")

    rough_geometry = get_rough_geometry_from_hough(board_bgr, x_positions, y_positions)

    refined_geometry = refine_geometry_by_control_points(board_bgr, rough_geometry)

    if refined_geometry is not None:
        print("最终网格：已启用清晰交叉控制点校正")
        return refined_geometry

    print("最终网格：控制点校正失败，回退粗定位")
    return rough_geometry


def fallback_geometry(board_bgr: np.ndarray) -> BoardGeometry:
    h, w = board_bgr.shape[:2]

    left = w * 0.067
    right = w * 0.933
    top = h * 0.064

    cell = (right - left) / (COLS - 1)
    bottom = top + cell * (ROWS - 1)

    if bottom > h * 0.965:
        bottom = h * 0.936
        cell = (bottom - top) / (ROWS - 1)
        width = cell * (COLS - 1)
        left = (w - width) / 2.0
        right = left + width

    crop_radius = int(round(cell * 0.48))

    return BoardGeometry(
        source=str(BOARD_CROP_PATH),
        method="fallback_margin_equal_cell",
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        cell_w=cell,
        cell_h=cell,
        crop_radius=max(18, crop_radius),
        rows=ROWS,
        cols=COLS,
        score=0.0,
    )


def generate_points(geometry: BoardGeometry) -> list[PointInfo]:
    points: list[PointInfo] = []

    for row in range(ROWS):
        for col in range(COLS):
            grid_x = geometry.left + col * geometry.cell_w
            grid_y = geometry.top + row * geometry.cell_h
            crop_path = POINT_CROPS_DIR / f"r{row}c{col}.png"

            points.append(
                PointInfo(
                    row=row,
                    col=col,
                    grid_x=float(grid_x),
                    grid_y=float(grid_y),
                    crop_x=float(grid_x),
                    crop_y=float(grid_y),
                    crop_path=str(crop_path),
                    crop_box={},
                )
            )

    return points


def shifted_geometry(geometry: BoardGeometry, row_shift: int) -> BoardGeometry:
    if row_shift == 0:
        return geometry

    shift_y = row_shift * geometry.cell_h

    return BoardGeometry(
        source=geometry.source,
        method=f"{geometry.method}_row_shift_{row_shift:+d}",
        left=geometry.left,
        top=geometry.top + shift_y,
        right=geometry.right,
        bottom=geometry.bottom + shift_y,
        cell_w=geometry.cell_w,
        cell_h=geometry.cell_h,
        crop_radius=geometry.crop_radius,
        rows=geometry.rows,
        cols=geometry.cols,
        score=geometry.score,
    )


def score_piece_center(crop_bgr: np.ndarray) -> float:
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(16, min(h, w) // 2),
        param1=80,
        param2=18,
        minRadius=max(10, int(min(h, w) * 0.24)),
        maxRadius=max(12, int(min(h, w) * 0.48)),
    )

    if circles is None:
        return 0.0

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    best = 0.0

    for circle in np.round(circles[0, :]).astype(int):
        x, y, radius = int(circle[0]), int(circle[1]), int(circle[2])
        center_error = float(np.hypot(x - cx, y - cy)) / max(min(h, w) * 0.24, 1.0)
        center_score = max(0.0, 1.0 - center_error)
        radius_score = max(0.0, 1.0 - abs(radius - min(h, w) * 0.34) / max(min(h, w) * 0.20, 1.0))
        best = max(best, center_score * 0.72 + radius_score * 0.28)

    return float(best)


def score_board_surface(crop_bgr: np.ndarray) -> float:
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return 0.0

    center = crop_bgr[h // 4:h * 3 // 4, w // 4:w * 3 // 4]
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    warm_board = ((h_ch >= 8) & (h_ch <= 38) & (s_ch >= 20) & (v_ch >= 120))
    green_frame = ((h_ch >= 35) & (h_ch <= 90) & (s_ch >= 35) & (v_ch >= 65))
    valid = warm_board | green_frame

    return float(np.count_nonzero(valid) / max(valid.size, 1))


def score_row_shift_candidate(board_bgr: np.ndarray, geometry: BoardGeometry) -> float:
    h, w = board_bgr.shape[:2]
    radius = int(geometry.crop_radius)
    piece_scores: list[float] = []
    penalty = 0.0

    def crop_for_score(cx: int, cy: int) -> tuple[np.ndarray | None, int]:
        left = cx - radius
        top = cy - radius
        right = cx + radius
        bottom = cy + radius
        overflow = max(0, -left) + max(0, -top) + max(0, right - w) + max(0, bottom - h)

        if overflow > radius:
            return None, overflow

        safe_left = max(0, left)
        safe_top = max(0, top)
        safe_right = min(w, right)
        safe_bottom = min(h, bottom)
        crop = board_bgr[safe_top:safe_bottom, safe_left:safe_right]
        if overflow > 0:
            crop = cv2.copyMakeBorder(
                crop,
                max(0, -top),
                max(0, bottom - h),
                max(0, -left),
                max(0, right - w),
                cv2.BORDER_REPLICATE,
            )
        return crop, overflow

    for row in range(ROWS):
        for col in range(COLS):
            cx = int(round(geometry.left + col * geometry.cell_w))
            cy = int(round(geometry.top + row * geometry.cell_h))

            crop, overflow = crop_for_score(cx, cy)
            if crop is None:
                penalty += 0.35
                continue

            if overflow > 0:
                penalty += overflow / max(radius * 12.0, 1.0)

            score = score_piece_center(crop)
            surface = score_board_surface(crop)
            penalty += max(0.0, 0.45 - surface) * 0.20
            if score >= 0.35:
                piece_scores.append(score)

    strong_piece_score = sum(piece_scores)
    edge_support = 0.0

    for row in (0, ROWS - 1):
        row_best = 0.0
        for col in range(COLS):
            cx = int(round(geometry.left + col * geometry.cell_w))
            cy = int(round(geometry.top + row * geometry.cell_h))
            crop, _ = crop_for_score(cx, cy)
            if crop is None:
                continue
            row_best = max(row_best, score_piece_center(crop))
        edge_support += row_best

    return float(strong_piece_score + edge_support * 2.0 - penalty)


def select_best_row_shift(board_bgr: np.ndarray, geometry: BoardGeometry) -> BoardGeometry:
    candidates = [shifted_geometry(geometry, shift) for shift in (-1, 0, 1)]
    scored = [(score_row_shift_candidate(board_bgr, candidate), candidate) for candidate in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)

    best_score, best_geometry = scored[0]
    current_score = next(score for score, candidate in scored if candidate is geometry)

    print(
        "行偏移自检评分："
        + ", ".join(f"{candidate.method}={score:.2f}" for score, candidate in scored)
    )

    if best_geometry is not geometry and best_score >= current_score + 1.25:
        print(f"行偏移自检：采用 {best_geometry.method}")
        return best_geometry

    print("行偏移自检：保持当前网格")
    return geometry


def crop_points(
    board_bgr: np.ndarray,
    geometry: BoardGeometry,
    points: list[PointInfo],
) -> list[PointInfo]:
    h, w = board_bgr.shape[:2]
    radius = int(geometry.crop_radius)
    updated: list[PointInfo] = []

    for point in points:
        cx = int(round(point.crop_x))
        cy = int(round(point.crop_y))

        left = cx - radius
        top = cy - radius
        right = cx + radius
        bottom = cy + radius

        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - w)
        pad_bottom = max(0, bottom - h)

        safe_left = max(0, left)
        safe_top = max(0, top)
        safe_right = min(w, right)
        safe_bottom = min(h, bottom)

        crop = board_bgr[safe_top:safe_bottom, safe_left:safe_right]

        if any([pad_left, pad_top, pad_right, pad_bottom]):
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REPLICATE,
            )

        ok = write_image_unicode(point.crop_path, crop)

        if not ok:
            print(f"模块二警告：保存裁剪图失败：{point.crop_path}")

        updated.append(
            PointInfo(
                row=point.row,
                col=point.col,
                grid_x=point.grid_x,
                grid_y=point.grid_y,
                crop_x=point.crop_x,
                crop_y=point.crop_y,
                crop_path=point.crop_path,
                crop_box={
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                },
            )
        )

    return updated


def save_preview(points: list[PointInfo]) -> None:
    tile = 88
    label_h = 34
    preview = Image.new("RGB", (COLS * tile, ROWS * (tile + label_h)), "white")
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()

    for point in points:
        image = Image.open(point.crop_path).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        x = point.col * tile
        y = point.row * (tile + label_h)

        preview.paste(image, (x, y))
        draw.text(
            (x + 3, y + tile + 2),
            f"r{point.row}c{point.col}\nunknown 0.00",
            fill=(20, 20, 20),
            font=font,
        )
        draw.rectangle(
            (x, y, x + tile - 1, y + tile + label_h - 1),
            outline=(210, 210, 210),
        )

    preview.save(PREVIEW_PATH)


def save_overlay(
    board_bgr: np.ndarray,
    geometry: BoardGeometry,
    points: list[PointInfo],
) -> None:
    rgb = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    x_lines = [geometry.left + col * geometry.cell_w for col in range(COLS)]
    y_lines = [geometry.top + row * geometry.cell_h for row in range(ROWS)]

    for x in x_lines:
        draw.line((x, geometry.top, x, geometry.bottom), fill=(255, 0, 0), width=1)

    for y in y_lines:
        draw.line((geometry.left, y, geometry.right, y), fill=(255, 0, 0), width=1)

    radius = max(4, int(min(geometry.cell_w, geometry.cell_h) * 0.055))

    for point in points:
        draw.ellipse(
            (
                point.grid_x - radius,
                point.grid_y - radius,
                point.grid_x + radius,
                point.grid_y + radius,
            ),
            outline=(0, 255, 0),
            width=2,
        )

        if point.row in {0, 3, 6, 9} or point.col in {0, 4, 8}:
            draw.text(
                (point.grid_x + radius + 2, point.grid_y - radius - 2),
                f"r{point.row}c{point.col}",
                fill=(255, 0, 0),
                font=font,
            )

    image.save(OVERLAY_PATH)


def save_json_outputs(
    geometry: BoardGeometry,
    points: list[PointInfo],
) -> None:
    GEOMETRY_PATH.write_text(
        json.dumps(asdict(geometry), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    serialized_points: list[dict[str, Any]] = []

    for point in points:
        item = asdict(point)

        # 兼容旧模块：x/y 仍然表示理论棋盘交点。
        item["x"] = point.grid_x
        item["y"] = point.grid_y

        serialized_points.append(item)

    payload = {
        "points": serialized_points,
    }

    POINTS_JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ensure_output_dirs()

    board_bgr = read_board_crop()

    if board_bgr is None:
        return

    geometry = locate_geometry(board_bgr)
    geometry = select_best_row_shift(board_bgr, geometry)

    print(f"最终使用的方法：{geometry.method}")
    print(
        f"left={geometry.left:.2f}, top={geometry.top:.2f}, "
        f"right={geometry.right:.2f}, bottom={geometry.bottom:.2f}"
    )
    print(f"cell_w={geometry.cell_w:.2f}, cell_h={geometry.cell_h:.2f}")
    print(f"crop_radius={geometry.crop_radius}")

    points = generate_points(geometry)

    if len(points) != 90:
        print("模块二错误：交点数量不是 90")
        return

    points = crop_points(board_bgr, geometry, points)

    crop_count = len(list(POINT_CROPS_DIR.glob("r*c*.png")))

    if crop_count != 90:
        print(f"模块二错误：交点裁剪图数量不是 90，当前数量：{crop_count}")
        return

    print("已成功输出 90 个裁剪图")

    try:
        save_json_outputs(geometry, points)
        print(f"已成功输出几何参数：{GEOMETRY_PATH}")
        print(f"已成功输出点位坐标：{POINTS_JSON_PATH}")
    except Exception as exc:
        print(f"模块二错误：保存核心 JSON 输出失败：{exc}")
        return

    try:
        save_preview(points)
        print(f"已成功输出预览图：{PREVIEW_PATH}")
    except Exception as exc:
        print(f"警告：保存点位预览图失败，不影响自动走棋：{exc}")

    try:
        save_overlay(board_bgr, geometry, points)
        print(f"已成功输出覆盖图：{OVERLAY_PATH}")
    except Exception as exc:
        print(f"警告：保存点位覆盖图失败，不影响自动走棋：{exc}")


if __name__ == "__main__":
    main()
