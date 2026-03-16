"""
Auto-generate YOLO labels for license plate images using EasyOCR text boxes.

Usage:
    python utils/autolabel_plates.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import easyocr


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATASET_ROOT = Path("datasets/plates")
CLASS_ID = 0  # license_plate


def _iter_images(folder: Path) -> Iterable[Path]:
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _to_yolo_line(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    bw = _clip((x2 - x1) / w, 1e-6, 1.0)
    bh = _clip((y2 - y1) / h, 1e-6, 1.0)
    cx = _clip(((x1 + x2) / 2) / w, 0.0, 1.0)
    cy = _clip(((y1 + y2) / 2) / h, 0.0, 1.0)
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _is_plate_like_text(text: str) -> bool:
    t = "".join(ch for ch in text.upper() if ch.isalnum())
    return len(t) >= 4 and any(ch.isdigit() for ch in t)


def _ocr_union_bbox(reader: easyocr.Reader, image_bgr) -> tuple[float, float, float, float] | None:
    h, w = image_bgr.shape[:2]
    ocr = reader.readtext(image_bgr, detail=1, paragraph=False)
    boxes = []
    for pts, text, conf in ocr:
        if conf < 0.15:
            continue
        if not _is_plate_like_text(text):
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        # Filter extreme tiny boxes.
        if (x2 - x1) < (0.05 * w) or (y2 - y1) < (0.03 * h):
            continue
        boxes.append((x1, y1, x2, y2))

    if not boxes:
        return None

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    # Add padding around text box to cover full plate region.
    pad_x = 0.05 * w
    pad_y = 0.04 * h
    x1 = _clip(x1 - pad_x, 0, w - 1)
    y1 = _clip(y1 - pad_y, 0, h - 1)
    x2 = _clip(x2 + pad_x, 1, w)
    y2 = _clip(y2 + pad_y, 1, h)
    return x1, y1, x2, y2


def generate_labels(split: str, reader: easyocr.Reader) -> tuple[int, int]:
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    labeled = 0
    for image_path in _iter_images(image_dir):
        total += 1
        if total % 5 == 0:
            print(f"[INFO] {split}: processed {total} images...")
        out_txt = label_dir / f"{image_path.stem}.txt"
        img = cv2.imread(str(image_path))
        if img is None:
            out_txt.write_text("", encoding="utf-8")
            continue
        h, w = img.shape[:2]
        bbox = _ocr_union_bbox(reader, img)
        if bbox is None:
            out_txt.write_text("", encoding="utf-8")
            continue
        line = _to_yolo_line(*bbox, w, h)
        out_txt.write_text(line + "\n", encoding="utf-8")
        labeled += 1
    return total, labeled


def _parse_args():
    p = argparse.ArgumentParser(description="Auto-generate YOLO labels for plate images")
    p.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR if available")
    return p.parse_args()


def main():
    args = _parse_args()
    print(f"[INFO] Loading EasyOCR (gpu={args.gpu})...")
    reader = easyocr.Reader(["en"], gpu=args.gpu)
    grand_total = 0
    grand_labeled = 0
    for split in ("train", "val"):
        print(f"[INFO] Processing split: {split}")
        total, labeled = generate_labels(split, reader)
        grand_total += total
        grand_labeled += labeled
        print(f"[INFO] {split}: labeled {labeled}/{total} images")

    print(f"[DONE] Generated labels for {grand_labeled}/{grand_total} images.")
    print("[NOTE] These are auto-labels; manually review for best training quality.")


if __name__ == "__main__":
    main()
