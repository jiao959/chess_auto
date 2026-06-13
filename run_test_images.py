from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import module1_board_detection as module1
import module2_point_crops as module2
import module3_piece_recognition as module3


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "test"
DEBUG_DIR = PROJECT_ROOT / "debug_outputs"
TEST_RESULTS_DIR = DEBUG_DIR / "test_results"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def read_image_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def copy_artifacts(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "full_screenshot.png",
        "board_crop.png",
        "board_detect_overlay.png",
        "points_overlay.png",
        "point_crops_preview.png",
        "board_geometry.json",
        "points.json",
        "piece_recognition_results.json",
    ]:
        source = DEBUG_DIR / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)
    crops_source = DEBUG_DIR / "point_crops"
    if crops_source.exists():
        crops_target = target_dir / "point_crops"
        if crops_target.exists():
            shutil.rmtree(crops_target)
        shutil.copytree(crops_source, crops_target)


def run_one(image_path: Path) -> dict:
    print(f"\n=== 测试图片：{image_path.name} ===")
    image_bgr = read_image_bgr(image_path)
    if image_bgr is None:
        print(f"无法读取测试图片：{image_path}")
        return {"image": str(image_path), "ok": False, "error": "无法读取测试图片"}

    module1.ensure_debug_dir()
    cv2.imwrite(str(module1.FULL_SCREENSHOT_PATH), image_bgr)
    result = module1.locate_board(image_bgr, module1.DEFAULT_SAMPLE_PATH)
    if not result.get("ok"):
        print(result.get("error", module1.ERROR_MESSAGE))
        return {"image": str(image_path), "ok": False, "error": result.get("error", module1.ERROR_MESSAGE)}
    module1.save_board_outputs(image_bgr, result)

    module2.main()
    module3.main()

    recognition_path = DEBUG_DIR / "piece_recognition_results.json"
    counts = {}
    total_points = 0
    if recognition_path.exists():
        payload = json.loads(recognition_path.read_text(encoding="utf-8"))
        total_points = int(payload.get("total_points", 0))
        for item in payload.get("results", []):
            label = item.get("predicted_class", "unknown")
            counts[label] = counts.get(label, 0) + 1

    target_dir = TEST_RESULTS_DIR / image_path.stem
    copy_artifacts(target_dir)
    print(f"已保存该测试图片的输出：{target_dir}")
    print(f"识别数量：{total_points}，类别统计：{counts}")
    return {"image": str(image_path), "ok": True, "output_dir": str(target_dir), "total_points": total_points, "counts": counts}


def main() -> None:
    if not TEST_DIR.exists():
        print("未找到 test/ 文件夹")
        return
    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in TEST_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        print("test/ 文件夹中没有图片")
        return
    summary = [run_one(path) for path in image_paths]
    summary_path = TEST_RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已完成 {len(image_paths)} 张测试图片，汇总：{summary_path}")


if __name__ == "__main__":
    main()
