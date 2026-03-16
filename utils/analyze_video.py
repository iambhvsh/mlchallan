"""
Analyze a traffic video with the existing detector pipeline and save an annotated output video.

Usage:
  python utils/analyze_video.py --input "D:\\Bhavesh\\challan\\traffic_video_modified.mp4"
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import cv2

# Allow running as: python utils/analyze_video.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.detector import TrafficViolationDetector
from utils.challan import ChallanManager


def parse_args():
    p = argparse.ArgumentParser(description="Analyze traffic video and save annotated output")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output", default="outputs/traffic_video_annotated.mp4", help="Output video path")
    p.add_argument("--violation-type", default="signal", help="Violation type")
    p.add_argument("--stop-line", type=int, default=35, help="Stop line y %% from top")
    p.add_argument("--sample-every", type=int, default=3, help="Run detection every N frames")
    p.add_argument("--cooldown-seconds", type=int, default=30, help="Cooldown per plate+violation for challan")
    p.add_argument("--auto-stop-line", action="store_true", help="Automatically estimate stop line per frame")
    return p.parse_args()


def normalize_plate(text: str) -> str:
    return (text or "").upper().replace(" ", "").replace("-", "")


def draw_overlay(frame, detections, violations, stop_line_y):
    h, w = frame.shape[:2]
    cv2.line(frame, (0, stop_line_y), (w, stop_line_y), (0, 255, 255), 2)
    cv2.putText(frame, "STOP LINE", (10, max(20, stop_line_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    violation_boxes = {tuple(v["box"]) for v in violations}
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        is_violation = tuple(d["box"]) in violation_boxes
        color = (0, 0, 255) if is_violation else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d.get('vehicle_type','vehicle')} {d.get('confidence',0):.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for v in violations:
        x1, y1, x2, y2 = v["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"VIOLATION: {v.get('violation_type','').upper()}",
                    (x1, max(18, y1 - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        plate = v.get("plate_text", "")
        if plate:
            cv2.putText(frame, f"PLATE: {plate}", (x1, min(h - 10, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    detector = TrafficViolationDetector(custom_model_path="license_plate_detector.pt")
    challan_mgr = ChallanManager()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    print(f"[INFO] Processing video: {args.input}")
    print(f"[INFO] Resolution: {width}x{height}  FPS: {fps:.2f}  Frames: {total}")
    print(f"[INFO] Output: {args.output}")

    last_result = {"detections": [], "violations": []}
    last_stop_line_y = int(height * args.stop_line / 100)
    recent = {}
    challans_issued = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % max(1, args.sample_every) == 0:
            detections = detector._detect_vehicles(frame)
            if args.auto_stop_line:
                estimated = detector.estimate_stop_line_y(frame, default_pct=args.stop_line)
                stop_y = int(0.7 * last_stop_line_y + 0.3 * estimated)
            else:
                stop_y = int(frame.shape[0] * args.stop_line / 100)
            last_stop_line_y = stop_y
            violations = detector._find_violations(detections, stop_y, frame.shape[0], args.violation_type)

            for v in violations:
                plate_text, plate_conf, _ = detector._extract_plate(frame, v["box"])
                v["plate_text"] = plate_text
                v["plate_confidence"] = plate_conf

                plate = normalize_plate(plate_text)
                if not plate:
                    continue
                key = (plate, args.violation_type)
                now = datetime.now()
                last_time = recent.get(key)
                if last_time and (now - last_time).total_seconds() < args.cooldown_seconds:
                    continue
                recent[key] = now

                challan_mgr.create_challan(
                    plate_text=plate_text,
                    violation_type=args.violation_type,
                    image_path=os.path.basename(args.input),
                    confidence=plate_conf,
                    vehicle_type=v.get("vehicle_type", "Unknown"),
                    annotated_image=os.path.basename(args.output),
                )
                challans_issued += 1

            last_result = {"detections": detections, "violations": violations}

        draw_overlay(frame, last_result["detections"], last_result["violations"], last_stop_line_y)
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[INFO] Processed {frame_idx}/{total if total > 0 else '?'} frames")

    cap.release()
    writer.release()

    print(f"[DONE] Video saved: {args.output}")
    print(f"[DONE] Challans issued from video: {challans_issued}")


if __name__ == "__main__":
    main()
