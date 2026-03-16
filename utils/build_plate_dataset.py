"""
Build/expand a plate detection dataset from traffic videos using pseudo-labeling.

Why:
- Real-world performance depends on real-world data variety.
- This script extracts diverse frames and auto-labels plate boxes from a seed model.
- You can then quickly review only hard/unlabeled frames.

Example:
  python utils/build_plate_dataset.py \
    --input "traffic_video_modified.mp4" \
    --model "license_plate_detector.pt" \
    --sample-fps 2 \
    --conf-thresh 0.45 \
    --max-frames 1200 \
    --save-unlabeled-hard
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("ultralytics is required. Install with: pip install ultralytics") from e


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
CLASS_ID = 0  # license_plate
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck (COCO ids in YOLOv8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build plate dataset from video using pseudo labels")
    p.add_argument("--input", nargs="+", required=True, help="Video files and/or folders")
    p.add_argument("--model", default="license_plate_detector.pt", help="Seed plate model path")
    p.add_argument("--out-root", default="datasets/plates", help="Output YOLO dataset root")
    p.add_argument("--sample-fps", type=float, default=2.0, help="Frames per second to sample")
    p.add_argument("--max-frames", type=int, default=1500, help="Max sampled frames to process")
    p.add_argument("--conf-thresh", type=float, default=0.45, help="Pseudo-label confidence threshold")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (0..1)")
    p.add_argument("--min-scene-diff", type=float, default=6.0, help="Minimum scene-change score")
    p.add_argument("--force-every", type=int, default=10, help="Force keep every N sampled frames")
    p.add_argument("--save-unlabeled-hard", action="store_true", help="Save high-detail frames with no label")
    p.add_argument("--clean", action="store_true", help="Clean generated dataset dirs before writing")
    p.add_argument("--use-ocr-fallback", action="store_true", help="Use vehicle+OCR fallback when plate model misses")
    p.add_argument("--ocr-gpu", action="store_true", help="Use GPU for EasyOCR fallback")
    p.add_argument("--ocr-min-conf", type=float, default=0.2, help="EasyOCR min confidence")
    return p.parse_args()


def iter_videos(inputs: Iterable[str]) -> Iterable[Path]:
    for item in inputs:
        p = Path(item)
        if not p.exists():
            print(f"[WARN] Missing path: {p}")
            continue
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p
            continue
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                    yield f


def ensure_dirs(base: Path):
    (base / "images" / "train").mkdir(parents=True, exist_ok=True)
    (base / "images" / "val").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (base / "review" / "unlabeled").mkdir(parents=True, exist_ok=True)


def write_data_yaml(base: Path):
    yaml_text = (
        f"path: ./{base.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "nc: 1\n"
        "names: ['license_plate']\n"
    )
    (base / "data.yaml").write_text(yaml_text, encoding="utf-8")


def maybe_clean(base: Path):
    import shutil
    targets = [
        base / "images" / "train",
        base / "images" / "val",
        base / "labels" / "train",
        base / "labels" / "val",
        base / "review" / "unlabeled",
    ]
    for t in targets:
        if t.exists():
            shutil.rmtree(t)


def yolo_line(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    bw = max(1e-6, min(1.0, (x2 - x1) / w))
    bh = max(1e-6, min(1.0, (y2 - y1) / h))
    cx = max(0.0, min(1.0, ((x1 + x2) / 2) / w))
    cy = max(0.0, min(1.0, ((y1 + y2) / 2) / h))
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def split_name(stem: str, val_ratio: float) -> str:
    h = hashlib.sha1(stem.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def scene_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    diff = cv2.absdiff(prev_gray, gray)
    return float(np.mean(diff))


def high_detail(gray: np.ndarray) -> bool:
    # Keep likely hard samples for manual review (many edges but no pseudo-labels).
    edges = cv2.Canny(gray, 100, 200)
    density = float(np.count_nonzero(edges)) / float(edges.size)
    return density > 0.06


def _is_plate_like_text(text: str) -> bool:
    t = "".join(ch for ch in text.upper() if ch.isalnum())
    return len(t) >= 4 and any(ch.isdigit() for ch in t)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def ocr_plate_boxes_in_crop(ocr_reader, crop_bgr: np.ndarray, min_conf: float) -> list[tuple[float, float, float, float]]:
    h, w = crop_bgr.shape[:2]
    out = []
    ocr = ocr_reader.readtext(crop_bgr, detail=1, paragraph=False)
    for pts, text, conf in ocr:
        if conf < min_conf:
            continue
        if not _is_plate_like_text(text):
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if (x2 - x1) < max(16, 0.04 * w) or (y2 - y1) < max(8, 0.02 * h):
            continue
        pad_x = 0.08 * (x2 - x1)
        pad_y = 0.25 * (y2 - y1)
        x1 = _clip(x1 - pad_x, 0, w - 1)
        y1 = _clip(y1 - pad_y, 0, h - 1)
        x2 = _clip(x2 + pad_x, 1, w)
        y2 = _clip(y2 + pad_y, 1, h)
        out.append((x1, y1, x2, y2))
    return out


def ocr_fallback_boxes(frame: np.ndarray, vehicle_model: YOLO, ocr_reader, min_conf: float) -> list[tuple[float, float, float, float]]:
    result = vehicle_model(frame, verbose=False)[0]
    boxes = []
    for vb in result.boxes:
        cls_id = int(vb.cls[0])
        if cls_id not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = map(int, vb.xyxy[0])
        if x2 <= x1 or y2 <= y1:
            continue
        # Plate tends to be in lower-front region.
        vh = y2 - y1
        sub_y1 = y1 + int(0.55 * vh)
        sub = frame[sub_y1:y2, x1:x2]
        if sub.size == 0:
            continue
        local_boxes = ocr_plate_boxes_in_crop(ocr_reader, sub, min_conf=min_conf)
        for lx1, ly1, lx2, ly2 in local_boxes:
            boxes.append((x1 + lx1, sub_y1 + ly1, x1 + lx2, sub_y1 + ly2))
    return boxes


def process_video(
    video_path: Path,
    model: YOLO,
    vehicle_model: YOLO | None,
    ocr_reader,
    out_root: Path,
    sample_fps: float,
    max_frames: int,
    conf_thresh: float,
    val_ratio: float,
    min_scene_diff: float,
    force_every: int,
    save_unlabeled_hard: bool,
    ocr_min_conf: float,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return 0, 0, 0

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(native_fps / max(sample_fps, 0.1))))
    print(f"[INFO] {video_path.name}: native_fps={native_fps:.2f}, sample_step={step}")

    sampled = 0
    saved_labeled = 0
    saved_review = 0
    frame_idx = 0
    kept_idx = 0
    prev_gray = None

    while sampled < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue

        sampled += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keep = False
        if prev_gray is None:
            keep = True
        else:
            score = scene_score(prev_gray, gray)
            keep = score >= min_scene_diff
        if sampled % max(1, force_every) == 0:
            keep = True
        if not keep:
            continue

        prev_gray = gray
        kept_idx += 1
        h, w = frame.shape[:2]
        result = model(frame, verbose=False)[0]
        lines = []
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            if (x2 - x1) < 10 or (y2 - y1) < 6:
                continue
            lines.append(yolo_line(x1, y1, x2, y2, w, h))

        if not lines and vehicle_model is not None and ocr_reader is not None:
            for x1, y1, x2, y2 in ocr_fallback_boxes(
                frame, vehicle_model=vehicle_model, ocr_reader=ocr_reader, min_conf=ocr_min_conf
            ):
                lines.append(yolo_line(x1, y1, x2, y2, w, h))

        base_stem = f"{video_path.stem}_f{frame_idx:06d}"
        split = split_name(base_stem, val_ratio)

        if lines:
            img_out = out_root / "images" / split / f"{base_stem}.jpg"
            lbl_out = out_root / "labels" / split / f"{base_stem}.txt"
            cv2.imwrite(str(img_out), frame)
            lbl_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            saved_labeled += 1
        elif save_unlabeled_hard and high_detail(gray):
            rev_out = out_root / "review" / "unlabeled" / f"{base_stem}.jpg"
            cv2.imwrite(str(rev_out), frame)
            saved_review += 1

        if sampled % 100 == 0:
            print(
                f"[INFO] {video_path.name}: sampled={sampled}, labeled={saved_labeled}, review={saved_review}"
            )

    cap.release()
    return sampled, saved_labeled, saved_review


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    if args.clean:
        maybe_clean(out_root)
    ensure_dirs(out_root)
    write_data_yaml(out_root)

    videos = list(iter_videos(args.input))
    if not videos:
        raise SystemExit("No valid videos found in --input")

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model}")
    model = YOLO(args.model)
    vehicle_model = None
    ocr_reader = None
    if args.use_ocr_fallback:
        try:
            import easyocr
            vehicle_model = YOLO("yolov8n.pt")
            ocr_reader = easyocr.Reader(["en"], gpu=args.ocr_gpu)
            print("[INFO] OCR fallback enabled (vehicle + EasyOCR)")
        except Exception as e:
            print(f"[WARN] Could not enable OCR fallback: {e}")
            vehicle_model = None
            ocr_reader = None

    total_sampled = 0
    total_labeled = 0
    total_review = 0

    for v in videos:
        sampled, labeled, review = process_video(
            video_path=v,
            model=model,
            vehicle_model=vehicle_model,
            ocr_reader=ocr_reader,
            out_root=out_root,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            conf_thresh=args.conf_thresh,
            val_ratio=args.val_ratio,
            min_scene_diff=args.min_scene_diff,
            force_every=args.force_every,
            save_unlabeled_hard=args.save_unlabeled_hard,
            ocr_min_conf=args.ocr_min_conf,
        )
        total_sampled += sampled
        total_labeled += labeled
        total_review += review

    print("[DONE] Dataset build complete")
    print(f"[DONE] Sampled frames: {total_sampled}")
    print(f"[DONE] Labeled frames: {total_labeled}")
    print(f"[DONE] Review frames: {total_review}")
    print(f"[DONE] data.yaml: {(out_root / 'data.yaml').as_posix()}")
    print("[NEXT] Manually review a subset, then run: python train_model.py --task plate")


if __name__ == "__main__":
    main()
