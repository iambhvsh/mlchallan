"""
Traffic Violation Detector
Uses YOLOv8 for vehicle + license plate detection, EasyOCR for plate reading.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import uuid
import re
import time

# ── graceful imports ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed – using mock detector")

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[WARN] easyocr not installed – OCR will be mocked")

# ── constants ─────────────────────────────────────────────────────────────────
VEHICLE_CLASSES = {
    2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
    1: 'bicycle', 6: 'train'
}

FINE_AMOUNTS = {
    'signal': 1000,
    'stop_line': 500,
    'no_helmet': 1000,
    'wrong_lane': 500,
    'speeding': 2000,
}

ANNOTATED_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads')
# ── Enhanced model parameters for better accuracy ──────────────────────────────
MIN_VEHICLE_CONFIDENCE = 0.50  # Increased from 0.35 to reduce false positives
MIN_PLATE_CONFIDENCE = 0.55   # Added plate-specific threshold
NMS_THRESHOLD = 0.35          # More aggressive NMS to reduce overlaps
LINE_MARGIN_PX = 8
TRACK_IOU_THRESHOLD = 0.30    # Slightly more relaxed for better tracking
TRACK_MAX_MISSES = 8          # Reduced for quicker stale track removal
TRACK_MIN_DOWNWARD_PX = 2
LIVE_STATE_TTL_SECONDS = 600
STOP_LINE_MIN_ANGLE_DEG = 12
STOP_LINE_MIN_LEN_RATIO = 0.25
MULTI_SCALE_ENABLED = True    # Enable multi-scale detection
OCR_UPSCALE_FACTOR = 3        # Increased from 2 for better OCR accuracy


class TrafficViolationDetector:
    def __init__(self, custom_model_path=None):
        self.vehicle_model = None
        self.plate_model = None
        self.ocr_reader = None
        self.live_states = {}
        self._load_models(custom_model_path)

    def _load_models(self, custom_model_path):
        # Vehicle detection – YOLOv8m (better accuracy than yolov8n, still fast)
        if YOLO_AVAILABLE:
            try:
                # Using YOLOv8m instead of YOLOv8n for significantly better accuracy
                self.vehicle_model = YOLO('yolov8m.pt')
                print("[INFO] YOLOv8m vehicle model loaded (improved from yolov8n)")
            except Exception as e:
                print(f"[WARN] Could not load YOLOv8m: {e}")
                try:
                    # Fallback to yolov8n if medium fails
                    self.vehicle_model = YOLO('yolov8n.pt')
                    print("[INFO] Fallback to YOLOv8n vehicle model")
                except Exception as e2:
                    print(f"[WARN] Could not load YOLOv8n either: {e2}")

            # License plate model – use a specialized plate detector if available
            # Falls back to using the main model's bounding boxes
            plate_model_path = custom_model_path or 'license_plate_detector.pt'
            if os.path.exists(plate_model_path):
                try:
                    self.plate_model = YOLO(plate_model_path)
                    print("[INFO] License plate model loaded")
                except Exception as e:
                    print(f"[WARN] Plate model not loaded: {e}")

        # OCR
        if OCR_AVAILABLE:
            try:
                # Use GPU if available for faster processing
                gpu_available = True
                try:
                    import torch
                    gpu_available = torch.cuda.is_available()
                except:
                    gpu_available = False
                
                self.ocr_reader = easyocr.Reader(['en'], gpu=gpu_available)
                gpu_str = " (GPU enabled)" if gpu_available else " (CPU only)"
                print(f"[INFO] EasyOCR loaded{gpu_str}")
            except Exception as e:
                print(f"[WARN] EasyOCR failed: {e}")

    # ── public API ────────────────────────────────────────────────────────────

    def analyze_image(self, image_path: str, violation_type: str, stop_line_y_pct: int = 50,
                      auto_stop_line: bool = False) -> dict:
        """
        Main entry point.
        Returns detections, violations, and an annotated image path.
        """
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Could not read image', 'violations': [], 'detections': []}

        h, w = img.shape[:2]
        if auto_stop_line:
            stop_line_y = self.estimate_stop_line_y(img, default_pct=stop_line_y_pct)
        else:
            stop_line_y = int(h * stop_line_y_pct / 100)

        # 1. Detect vehicles
        detections = self._detect_vehicles(img)

        # 2. Find violations (crossed stop line / in intersection)
        violations = self._find_violations(detections, stop_line_y, h, violation_type)

        # 3. For each violation, detect + read number plate
        for v in violations:
            box = v['box']
            plate_text, plate_conf, plate_box = self._extract_plate(img, box)
            v['plate_text'] = plate_text
            v['plate_confidence'] = plate_conf
            v['plate_box'] = plate_box

        # 4. Annotate image
        annotated_path = self._annotate(img, detections, violations, stop_line_y, image_path)

        return {
            'detections': detections,
            'violations': violations,
            'annotated_image': os.path.basename(annotated_path),
            'summary': self._build_summary(detections, violations),
            'stop_line_y': stop_line_y
        }

    def analyze_live_frame(self, img: np.ndarray, violation_type: str, stop_line_y_pct: int = 50,
                           stream_id: str = "default", auto_stop_line: bool = False) -> dict:
        """
        Live analysis with temporal tracking for robust line-cross violation detection.
        Returns detections/violations and summary. Annotation is handled by client overlay.
        """
        if img is None or img.size == 0:
            return {'error': 'Could not read frame', 'violations': [], 'detections': []}

        h = img.shape[0]
        if auto_stop_line:
            raw_stop_y = self.estimate_stop_line_y(img, default_pct=stop_line_y_pct)
            state = self.live_states.setdefault(stream_id, {
                "tracks": {},
                "next_id": 1,
                "frame_idx": 0,
                "last_seen_ts": time.time(),
            })
            prev_stop = state.get("stop_line_y")
            stop_line_y = int(0.7 * prev_stop + 0.3 * raw_stop_y) if prev_stop else raw_stop_y
            state["stop_line_y"] = stop_line_y
        else:
            stop_line_y = int(h * stop_line_y_pct / 100)
        detections = self._detect_vehicles(img)
        detections = [d for d in detections if d.get('confidence', 0.0) >= MIN_VEHICLE_CONFIDENCE]

        violations = self._find_violations_temporal(
            detections=detections,
            stop_line_y=stop_line_y,
            violation_type=violation_type,
            stream_id=stream_id
        )

        for v in violations:
            plate_text, plate_conf, plate_box = self._extract_plate(img, v['box'])
            v['plate_text'] = plate_text
            v['plate_confidence'] = plate_conf
            v['plate_box'] = plate_box

        return {
            'detections': detections,
            'violations': violations,
            'summary': self._build_summary(detections, violations),
            'stop_line_y': stop_line_y,
            'live': True,
        }

    # ── detection ─────────────────────────────────────────────────────────────

    def _detect_vehicles(self, img: np.ndarray) -> list:
        """
        Enhanced vehicle detection with multi-scale processing and better filtering.
        """
        if self.vehicle_model is None:
            return self._mock_detections(img)

        detections = []
        
        # Primary detection at full resolution
        results = self.vehicle_model(img, verbose=False, conf=MIN_VEHICLE_CONFIDENCE, iou=NMS_THRESHOLD)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Apply size filtering – ignore very small/large detections
            box_width = x2 - x1
            box_height = y2 - y1
            img_h, img_w = img.shape[:2]
            area_ratio = (box_width * box_height) / (img_h * img_w)
            
            if 0.002 < area_ratio < 0.8:  # Reasonable size constraints
                detections.append({
                    'vehicle_type': VEHICLE_CLASSES[cls_id],
                    'confidence': conf,
                    'box': [x1, y1, x2, y2],
                    'center_y': (y1 + y2) // 2,
                    'scale': 'primary'
                })
        
        # Multi-scale detection for small objects (if enabled)
        if MULTI_SCALE_ENABLED and len(detections) < 5:
            # Check at 1.5x upscaling for small vehicles
            h, w = img.shape[:2]
            scale_factor = 1.3
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            if new_h < 1280 and new_w < 1280:  # Limit upscaling to avoid excessive computation
                img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                results_scaled = self.vehicle_model(img_scaled, verbose=False, conf=MIN_VEHICLE_CONFIDENCE, iou=NMS_THRESHOLD)[0]
                
                for box in results_scaled.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Scale back to original coordinates
                    x1, y1, x2, y2 = int(x1/scale_factor), int(y1/scale_factor), int(x2/scale_factor), int(y2/scale_factor)
                    conf = float(box.conf[0])
                    
                    # Check if this is a new detection (avoid duplicates)
                    is_duplicate = False
                    for existing in detections:
                        iou = self._iou_xyxy([x1, y1, x2, y2], existing['box'])
                        if iou > 0.5:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        detections.append({
                            'vehicle_type': VEHICLE_CLASSES[cls_id],
                            'confidence': conf * 0.95,  # Slight confidence penalty for scaled detection
                            'box': [x1, y1, x2, y2],
                            'center_y': (y1 + y2) // 2,
                            'scale': 'upscaled'
                        })
        
        # Sort by confidence (descending)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections

    def _find_violations(self, detections, stop_line_y, img_h, violation_type):
        """
        Enhanced violation detection with better filtering.
        A vehicle is in violation if its bottom edge is beyond the stop line.
        """
        violations = []
        for d in detections:
            x1, y1, x2, y2 = d['box']
            bottom_y = y2  # bottom of bounding box = front of vehicle
            margin = max(LINE_MARGIN_PX, int(img_h * 0.015))
            conf = d.get('confidence', 0.0)
            
            # Only report violations for high-confidence detections
            if conf >= MIN_VEHICLE_CONFIDENCE and bottom_y > (stop_line_y + margin):
                # Additional filter: ignore detections at image boundaries (often unreliable)
                box_width = x2 - x1
                if x1 > 10 and x2 < 1270 and box_width > 20:
                    violations.append({
                        **d,
                        'violation_type': violation_type,
                        'crossed_by_px': bottom_y - stop_line_y,
                    })
        return violations

    def _find_violations_temporal(self, detections, stop_line_y, violation_type, stream_id):
        self._cleanup_live_states()
        state = self.live_states.setdefault(stream_id, {
            "tracks": {},
            "next_id": 1,
            "frame_idx": 0,
            "last_seen_ts": time.time(),
        })
        state["frame_idx"] += 1
        state["last_seen_ts"] = time.time()

        tracks = state["tracks"]
        track_ids = list(tracks.keys())
        matched_track_ids = set()
        violations = []

        # Greedy IoU matching to keep lightweight temporal continuity.
        for d in detections:
            best_tid = None
            best_iou = 0.0
            for tid in track_ids:
                t = tracks[tid]
                if t.get("vehicle_type") != d.get("vehicle_type"):
                    continue
                iou = self._iou_xyxy(t["box"], d["box"])
                if iou >= TRACK_IOU_THRESHOLD and iou > best_iou and tid not in matched_track_ids:
                    best_iou = iou
                    best_tid = tid

            if best_tid is None:
                tid = state["next_id"]
                state["next_id"] += 1
                tracks[tid] = {
                    "box": d["box"],
                    "bottom_y": d["box"][3],
                    "misses": 0,
                    "vehicle_type": d.get("vehicle_type", "vehicle"),
                    "last_cross_frame": -10**9,
                    "seen_count": 1,
                }
                d["track_id"] = tid
                matched_track_ids.add(tid)
                continue

            t = tracks[best_tid]
            prev_bottom = t["bottom_y"]
            curr_bottom = d["box"][3]
            moved_down = (curr_bottom - prev_bottom) >= TRACK_MIN_DOWNWARD_PX
            crossed_line = (
                prev_bottom <= (stop_line_y - LINE_MARGIN_PX) and
                curr_bottom >= (stop_line_y + LINE_MARGIN_PX) and
                moved_down
            )

            t["box"] = d["box"]
            t["bottom_y"] = curr_bottom
            t["misses"] = 0
            t["seen_count"] += 1
            d["track_id"] = best_tid
            matched_track_ids.add(best_tid)

            # Trigger once per track for a short interval to avoid duplicate spikes.
            if crossed_line and (state["frame_idx"] - t["last_cross_frame"] > 15) and t["seen_count"] >= 2:
                t["last_cross_frame"] = state["frame_idx"]
                violations.append({
                    **d,
                    "violation_type": violation_type,
                    "crossed_by_px": max(0, curr_bottom - stop_line_y),
                })

        # Age out stale tracks.
        for tid in list(track_ids):
            if tid in matched_track_ids:
                continue
            tracks[tid]["misses"] += 1
            if tracks[tid]["misses"] > TRACK_MAX_MISSES:
                del tracks[tid]

        return violations

    # ── plate extraction ──────────────────────────────────────────────────────

    def _extract_plate(self, img, vehicle_box):
        """
        Enhanced plate extraction with better region detection.
        """
        x1, y1, x2, y2 = vehicle_box
        box_height = y2 - y1
        box_width = x2 - x1
        
        # For different vehicle types, plate location varies
        # Cars/motorcycles/buses: typically in lower 30-40% of vehicle bbox
        plate_region_start_ratio = 0.60
        plate_region_height_ratio = 0.35
        
        plate_region_y1 = y1 + int(box_height * plate_region_start_ratio)
        plate_region_y2 = min(y2, y1 + int(box_height * (plate_region_start_ratio + plate_region_height_ratio)))
        
        crop = img[plate_region_y1:plate_region_y2, x1:x2]

        if crop.size == 0:
            return '', 0.0, None

        plate_box_abs = None
        plate_crop = crop

        # Try dedicated plate model first with higher confidence threshold
        if self.plate_model:
            try:
                res = self.plate_model(crop, verbose=False, conf=MIN_PLATE_CONFIDENCE, iou=NMS_THRESHOLD)[0]
                if res.boxes:
                    pb = res.boxes[0]
                    px1, py1, px2, py2 = map(int, pb.xyxy[0])
                    
                    # Ensure valid crop
                    py1 = max(0, py1)
                    px1 = max(0, px1)
                    py2 = min(crop.shape[0], py2)
                    px2 = min(crop.shape[1], px2)
                    
                    if py2 > py1 and px2 > px1:
                        plate_crop = crop[py1:py2, px1:px2]
                        plate_box_abs = [x1 + px1, plate_region_y1 + py1,
                                        x1 + px2, plate_region_y1 + py2]
            except Exception:
                pass

        # OCR
        text, conf = self._run_ocr(plate_crop)
        return text, conf, plate_box_abs

    def _run_ocr(self, img_crop):
        """
        Enhanced OCR with advanced preprocessing for Indian license plates.
        """
        if self.ocr_reader is None:
            return self._mock_ocr(), 0.85
        
        if img_crop.size == 0:
            return '', 0.0

        try:
            # 1. Enhanced preprocessing for license plates
            gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            
            # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # 3. Bilateral filtering to preserve edges while reducing noise
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # 4. Upscale for better character recognition
            upscaled = cv2.resize(filtered, None, fx=OCR_UPSCALE_FACTOR, fy=OCR_UPSCALE_FACTOR, 
                                  interpolation=cv2.INTER_CUBIC)
            
            # 5. Adaptive thresholding for better character segmentation
            thresh = cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            
            # 6. Morphological operations to clean up the image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # 7. Run OCR on preprocessed image
            results = self.ocr_reader.readtext(cleaned)
            
            if results:
                # Combine all text fragments and filter invalid characters
                texts = [r[1] for r in results]
                confs = [r[2] for r in results]
                combined = ''.join(texts).upper().replace(" ", "").replace("-", "")
                
                # Indian plate pattern: 2 letters (state) + 2 digits + 2 letters + 4 digits
                # Clean up non-alphanumeric characters
                combined = re.sub(r'[^A-Z0-9]', '', combined)
                
                # Filter out unlikely plates (too short or too long)
                if 6 <= len(combined) <= 14:
                    avg_conf = float(np.mean(confs)) if confs else 0.0
                    return combined, min(avg_conf, 0.95)
                    
        except Exception as e:
            print(f"[WARN] OCR error: {e}")

        return '', 0.0

    # ── annotation ────────────────────────────────────────────────────────────

    def _annotate(self, img, detections, violations, stop_line_y, original_path):
        out = img.copy()
        h, w = out.shape[:2]

        # Draw stop line
        cv2.line(out, (0, stop_line_y), (w, stop_line_y), (0, 255, 255), 3)
        cv2.putText(out, 'STOP LINE', (10, stop_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        violation_boxes = {tuple(v['box']) for v in violations}

        for d in detections:
            box = d['box']
            is_violation = tuple(box) in violation_boxes
            color = (0, 0, 255) if is_violation else (0, 200, 0)
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d['vehicle_type']} {d['confidence']:.2f}"
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for v in violations:
            x1, y1, x2, y2 = v['box']
            plate = v.get('plate_text', '')
            if plate:
                cv2.putText(out, f"PLATE: {plate}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            pb = v.get('plate_box')
            if pb:
                cv2.rectangle(out, (pb[0], pb[1]), (pb[2], pb[3]), (0, 165, 255), 2)

            # Violation banner
            cv2.rectangle(out, (x1, y1 - 30), (x2, y1), (0, 0, 200), -1)
            cv2.putText(out, f"VIOLATION: {v['violation_type'].upper()}",
                        (x1 + 4, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        out_name = 'annotated_' + os.path.basename(original_path)
        out_path = os.path.join(ANNOTATED_DIR, out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, out)
        return out_path

    # ── helpers / mocks ───────────────────────────────────────────────────────

    def _mock_detections(self, img):
        h, w = img.shape[:2]
        return [
            {'vehicle_type': 'car', 'confidence': 0.91, 'box': [50, 100, 300, 350], 'center_y': 225},
            {'vehicle_type': 'motorcycle', 'confidence': 0.87, 'box': [350, 200, 500, 420], 'center_y': 310},
        ]

    def _mock_ocr(self):
        import random, string
        state = random.choice(['MH', 'DL', 'KA', 'TN', 'GJ'])
        dist = random.randint(1, 99)
        alpha = ''.join(random.choices(string.ascii_uppercase, k=2))
        num = random.randint(1000, 9999)
        return f"{state}{dist:02d}{alpha}{num}"

    def _build_summary(self, detections, violations):
        return (f"Detected {len(detections)} vehicle(s). "
                f"{len(violations)} violation(s) found. "
                f"Number plates read: {sum(1 for v in violations if v.get('plate_text'))}.")

    @staticmethod
    def _iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        b_area = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(a_area + b_area - inter)

    def reset_live_state(self, stream_id: str | None = None):
        if stream_id:
            self.live_states.pop(stream_id, None)
        else:
            self.live_states.clear()

    def _cleanup_live_states(self):
        now = time.time()
        stale = [
            sid for sid, s in self.live_states.items()
            if now - s.get("last_seen_ts", now) > LIVE_STATE_TTL_SECONDS
        ]
        for sid in stale:
            del self.live_states[sid]

    def estimate_stop_line_y(self, img: np.ndarray, default_pct: int = 35) -> int:
        """
        Estimate stop-line y-position from near-horizontal road markings.
        Falls back to default percentage if confidence is low.
        """
        h, w = img.shape[:2]
        default_y = int(h * default_pct / 100)
        if h < 50 or w < 50:
            return default_y

        roi_y1 = int(h * 0.45)
        roi_y2 = int(h * 0.95)
        roi = img[roi_y1:roi_y2, :]
        if roi.size == 0:
            return default_y

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 70, 180)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=max(40, int(w * STOP_LINE_MIN_LEN_RATIO)),
            maxLineGap=25
        )
        if lines is None:
            return default_y

        candidates = []
        roi_h = max(1, roi_y2 - roi_y1)
        target = 0.62 * roi_h

        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            ang = abs(np.degrees(np.arctan2(dy, dx)))
            if ang > STOP_LINE_MIN_ANGLE_DEG:
                continue

            length = float(np.hypot(dx, dy))
            y_mid = (y1 + y2) / 2.0
            dist_penalty = 1.0 / (1.0 + abs(y_mid - target) / max(1.0, 0.2 * roi_h))
            score = length * dist_penalty
            candidates.append((y_mid, score))

        if not candidates:
            return default_y

        weighted_y = sum(y * s for y, s in candidates) / max(1e-6, sum(s for _, s in candidates))
        y_abs = int(roi_y1 + weighted_y)
        return max(int(h * 0.20), min(int(h * 0.90), y_abs))
