from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app_paths import app_root, resource_path


PROJECT_ROOT = app_root()
TEMPLATE_DIR = resource_path("template")
DEBUG_DIR = PROJECT_ROOT / "debug_outputs"
POINT_CROPS_DIR = DEBUG_DIR / "point_crops"
RESULTS_PATH = DEBUG_DIR / "piece_recognition_results.json"
PREVIEW_PATH = DEBUG_DIR / "point_recognition_preview.png"

ROWS = 10
COLS = 9
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PIECE_CLASSES = [
    "black_advisor",
    "black_bishop",
    "black_cannon",
    "black_king",
    "black_knight",
    "black_pawn",
    "black_rook",
    "red_advisor",
    "red_bishop",
    "red_cannon",
    "red_king",
    "red_knight",
    "red_pawn",
    "red_rook",
]


@dataclass(frozen=True)
class TemplateSample:
    class_name: str
    path: str
    feature: np.ndarray
    variants: list[np.ndarray]


@dataclass(frozen=True)
class RecognitionResult:
    row: int
    col: int
    crop_path: str
    predicted_class: str
    confidence: float
    best_template_path: str | None
    is_empty: bool
    presence_score: float
    color: str
    top1_class: str | None
    top1_score: float
    top2_class: str | None
    top2_score: float
    score_margin: float


def read_image_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_templates() -> tuple[list[TemplateSample], dict[str, int]]:
    if not TEMPLATE_DIR.exists():
        print("模块三错误：未找到棋子样本库 template/")
        return [], {}

    samples: list[TemplateSample] = []
    counts: dict[str, int] = {}
    for class_name in PIECE_CLASSES:
        class_dir = TEMPLATE_DIR / class_name
        image_paths = []
        if class_dir.exists():
            image_paths = sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        counts[class_name] = len(image_paths)
        if not image_paths:
            print(f"警告：类别文件夹为空：template/{class_name}")
            continue
        for path in image_paths:
            image = read_image_bgr(path)
            if image is None:
                print(f"警告：无法读取样本：{path}")
                continue
            color = "red" if class_name.startswith("red_") else "black"
            feature = extract_text_mask_feature(image, color)
            samples.append(TemplateSample(class_name=class_name, path=str(path), feature=feature, variants=augmented_features(feature)))
    print(f"已加载样本数量：{len(samples)}")
    return samples, counts


def load_point_crops() -> list[tuple[int, int, Path]]:
    if not POINT_CROPS_DIR.exists():
        print("模块三错误：未找到点位裁剪图 debug_outputs/point_crops/")
        return []

    crops: list[tuple[int, int, Path]] = []
    pattern = re.compile(r"^r(\d+)c(\d+)\.png$", re.IGNORECASE)
    for path in sorted(POINT_CROPS_DIR.glob("r*c*.png")):
        match = pattern.match(path.name)
        if not match:
            continue
        row = int(match.group(1))
        col = int(match.group(2))
        if 0 <= row < ROWS and 0 <= col < COLS:
            crops.append((row, col, path))
    print(f"读取点位裁剪图数量：{len(crops)}")
    return crops


def center_mask(shape: tuple[int, int], radius_ratio: float = 0.47) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    radius = min(h, w) * radius_ratio
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2).astype(np.uint8)


def locate_piece_circle(image_bgr: np.ndarray) -> tuple[float, float, float]:
    h, w = image_bgr.shape[:2]
    # Module 2 already crops every point around the board intersection, so the
    # piece center should be the crop center.  Hough circles are easily pulled
    # toward rim shadows and produced worse text alignment on real crops.
    return (w - 1) / 2.0, (h - 1) / 2.0, min(w, h) * 0.42


def extract_text_mask_feature(image_bgr: np.ndarray, color: str, size: int = 72) -> np.ndarray:
    cx, cy, radius = locate_piece_circle(image_bgr)
    scale = size / max(radius * 2.0, 1.0)
    matrix = np.array(
        [
            [scale, 0.0, size / 2.0 - cx * scale],
            [0.0, scale, size / 2.0 - cy * scale],
        ],
        dtype=np.float32,
    )
    normalized = cv2.warpAffine(
        image_bgr,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return extract_text_mask_from_normalized_piece(normalized, color).reshape(-1)


def extract_text_mask_from_normalized_piece(piece_bgr: np.ndarray, color: str) -> np.ndarray:
    size = piece_bgr.shape[0]
    hsv = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2GRAY)
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2.0
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    inner = dist <= size * 0.32

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    if color == "red":
        mask = inner & (((h <= 13) | (h >= 168)) & (s >= 45) & (v >= 55))
    elif color == "black":
        mask = inner & (((v <= 112) & (s <= 170)) | (gray <= 86))
    else:
        mask = inner & (gray <= 115)

    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask_f = mask_u8.astype(np.float32) / 255.0
    norm = float(np.linalg.norm(mask_f))
    if norm <= 1e-6:
        return mask_f
    return mask_f / norm


def transform_feature(feature: np.ndarray, angle: float, scale: float, dx: float, dy: float, size: int = 72) -> np.ndarray:
    image = feature.reshape(size, size).astype(np.float32)
    center = (size / 2.0, size / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    warped = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    norm = float(np.linalg.norm(warped))
    if norm <= 1e-6:
        return warped.reshape(-1)
    return (warped / norm).reshape(-1)


def augmented_features(feature: np.ndarray) -> list[np.ndarray]:
    variants = [feature]
    for angle in (-6.0, 6.0):
        variants.append(transform_feature(feature, angle=angle, scale=1.0, dx=0.0, dy=0.0))
    for scale in (0.94, 1.06):
        variants.append(transform_feature(feature, angle=0.0, scale=scale, dx=0.0, dy=0.0))
    for dx, dy in ((-3.0, 0.0), (3.0, 0.0), (0.0, -3.0), (0.0, 3.0)):
        variants.append(transform_feature(feature, angle=0.0, scale=1.0, dx=dx, dy=dy))
    return variants


def extract_piece_feature(image_bgr: np.ndarray, size: int = 72) -> np.ndarray:
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 45, 135).astype(np.float32) / 255.0
    text = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5,
    ).astype(np.float32) / 255.0
    mask = center_mask((size, size), 0.43).astype(np.float32)
    feature = (edges * 0.42 + text * 0.58) * mask
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-6:
        return feature.reshape(-1)
    return (feature / norm).reshape(-1)


def estimate_presence(image_bgr: np.ndarray) -> float:
    resized = cv2.resize(image_bgr, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 135)

    mask_center = center_mask(gray.shape, 0.43).astype(bool)
    yy, xx = np.ogrid[:96, :96]
    dist = np.sqrt((xx - 47.5) ** 2 + (yy - 47.5) ** 2)
    mask_ring = (dist >= 25) & (dist <= 45)
    corner_mask = np.zeros(gray.shape, dtype=bool)
    corner = 16
    corner_mask[:corner, :corner] = True
    corner_mask[:corner, -corner:] = True
    corner_mask[-corner:, :corner] = True
    corner_mask[-corner:, -corner:] = True

    center_gray = gray[mask_center]
    corner_gray = gray[corner_mask]
    center_hsv = hsv[mask_center]
    corner_hsv = hsv[corner_mask]
    edge_density = float(np.count_nonzero(edges[mask_center]) / max(np.count_nonzero(mask_center), 1))
    ring_edge_density = float(np.count_nonzero(edges[mask_ring]) / max(np.count_nonzero(mask_ring), 1))
    dark_ratio = float(np.count_nonzero(center_gray < 115) / max(center_gray.size, 1))
    color_delta = float(np.linalg.norm(np.median(center_hsv, axis=0).astype(np.float32) - np.median(corner_hsv, axis=0).astype(np.float32)) / 255.0)
    gray_delta = float(abs(float(np.median(center_gray)) - float(np.median(corner_gray))) / 255.0)

    score = (
        min(edge_density / 0.16, 1.0) * 0.28
        + min(ring_edge_density / 0.14, 1.0) * 0.24
        + min(dark_ratio / 0.18, 1.0) * 0.24
        + min(color_delta / 0.30, 1.0) * 0.14
        + min(gray_delta / 0.22, 1.0) * 0.10
    )
    return float(max(0.0, min(score, 1.0)))


def detect_piece_color(image_bgr: np.ndarray) -> str:
    resized = cv2.resize(image_bgr, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    mask = center_mask(gray.shape, 0.34).astype(bool)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    red_pixels = mask & (((h <= 10) | (h >= 170)) & (s >= 55) & (v >= 70))

    # Black text is dark and concentrated in the central character region.
    # The center mask avoids board lines, piece rim shadows, and outside background.
    black_pixels = mask & (v <= 105) & (s <= 150)
    strong_black_pixels = mask & (gray <= 80)

    mask_area = max(int(np.count_nonzero(mask)), 1)
    red_ratio = float(np.count_nonzero(red_pixels) / mask_area)
    black_ratio = float((np.count_nonzero(black_pixels) + np.count_nonzero(strong_black_pixels)) / mask_area)

    if red_ratio >= 0.030 and red_ratio >= black_ratio * 0.75:
        return "red"
    if black_ratio >= 0.035 and black_ratio > red_ratio * 1.15:
        return "black"
    return "unknown"


def classify_crop(image_bgr: np.ndarray, templates: list[TemplateSample]) -> tuple[str, float, str | None, str, str | None, float, str | None, float, float]:
    color = detect_piece_color(image_bgr)
    if color not in {"red", "black"}:
        return "unknown", 0.0, None, color, None, 0.0, None, 0.0, 0.0
    templates = [sample for sample in templates if sample.class_name.startswith(f"{color}_")]
    if not templates:
        return "unknown", 0.0, None, color, None, 0.0, None, 0.0, 0.0
    feature = extract_text_mask_feature(image_bgr, color)
    query_features = augmented_features(feature)
    class_scores: dict[str, tuple[float, str]] = {}
    for sample in templates:
        score = max(float(np.dot(query_feature, template_feature)) for query_feature in query_features for template_feature in sample.variants)
        current = class_scores.get(sample.class_name)
        if current is None or score > current[0]:
            class_scores[sample.class_name] = (score, sample.path)

    ranked = sorted(class_scores.items(), key=lambda item: item[1][0], reverse=True)
    if not ranked:
        return "unknown", 0.0, None, color, None, 0.0, None, 0.0, 0.0
    top1_class, (top1_score, top1_path) = ranked[0]
    if len(ranked) > 1:
        top2_class, (top2_score, _) = ranked[1]
    else:
        top2_class, top2_score = None, 0.0

    margin = float(top1_score - top2_score)
    score_conf = max(0.0, min((top1_score - 0.18) / 0.62, 1.0))
    margin_conf = max(0.0, min(margin / 0.16, 1.0))
    confidence = score_conf * 0.58 + margin_conf * 0.42
    if top1_score < 0.34 or margin < 0.035 or confidence < 0.7:
        return "unknown", float(confidence), top1_path, color, top1_class, float(top1_score), top2_class, float(top2_score), margin
    return top1_class, float(confidence), top1_path, color, top1_class, float(top1_score), top2_class, float(top2_score), margin


def recognize_all(crops: list[tuple[int, int, Path]], templates: list[TemplateSample]) -> list[RecognitionResult]:
    results: list[RecognitionResult] = []
    for row, col, path in crops:
        image = read_image_bgr(path)
        if image is None:
            print(f"警告：无法读取点位图 r{row}c{col}，标为 unknown")
            results.append(
                RecognitionResult(row, col, str(path), "unknown", 0.0, None, False, 0.0, "unknown", None, 0.0, None, 0.0, 0.0)
            )
            continue

        presence_score = estimate_presence(image)
        if presence_score < 0.50:
            results.append(
                RecognitionResult(row, col, str(path), "empty", 1.0 - presence_score, None, True, presence_score, "none", None, 0.0, None, 0.0, 0.0)
            )
            continue

        predicted, confidence, best_path, color, top1_class, top1_score, top2_class, top2_score, score_margin = classify_crop(image, templates)
        if predicted == "unknown":
            print(f"警告：点位 r{row}c{col} 无法可靠分类，color={color}, presence={presence_score:.2f}, confidence={confidence:.2f}")
        results.append(
            RecognitionResult(
                row,
                col,
                str(path),
                predicted,
                confidence,
                best_path,
                False,
                presence_score,
                color,
                top1_class,
                top1_score,
                top2_class,
                top2_score,
                score_margin,
            )
        )
    return sorted(results, key=lambda item: (item.row, item.col))


def save_results_json(results: list[RecognitionResult], template_counts: dict[str, int]) -> None:
    payload: dict[str, Any] = {
        "template_dir": str(TEMPLATE_DIR),
        "template_counts": template_counts,
        "total_points": len(results),
        "results": [asdict(result) for result in results],
    }
    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_preview(results: list[RecognitionResult]) -> None:
    tile = 88
    label_h = 42
    preview = Image.new("RGB", (COLS * tile, ROWS * (tile + label_h)), "white")
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    by_key = {(item.row, item.col): item for item in results}
    for row in range(ROWS):
        for col in range(COLS):
            item = by_key.get((row, col))
            x = col * tile
            y = row * (tile + label_h)
            if item and Path(item.crop_path).exists():
                crop = Image.open(item.crop_path).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            else:
                crop = Image.new("RGB", (tile, tile), (235, 235, 235))
            preview.paste(crop, (x, y))
            label = "unknown"
            confidence = 0.0
            if item is not None:
                label = item.predicted_class
                confidence = item.confidence
            draw.text((x + 3, y + tile + 2), f"r{row}c{col}\n{label[:14]}\n{confidence:.2f}", fill=(20, 20, 20), font=font)
            draw.rectangle((x, y, x + tile - 1, y + tile + label_h - 1), outline=(210, 210, 210))
    preview.save(PREVIEW_PATH)


def main() -> None:
    templates, template_counts = load_templates()

    if not TEMPLATE_DIR.exists():
        return

    crops = load_point_crops()

    if not crops:
        return

    if len(crops) != 90:
        print(f"警告：点位裁剪图数量不是 90，当前数量：{len(crops)}")

    results = recognize_all(crops, templates)

    try:
        save_results_json(results, template_counts)
        print(f"已保存棋子识别结果：{RESULTS_PATH}")
    except Exception as exc:
        print(f"模块三错误：保存棋子识别结果失败：{exc}")
        return

    try:
        save_preview(results)
        print(f"已保存带识别标签预览图：{PREVIEW_PATH}")
    except Exception as exc:
        print(f"警告：保存棋子识别预览图失败，不影响自动走棋：{exc}")


if __name__ == "__main__":
    main()
