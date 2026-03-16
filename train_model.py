"""
train_model.py
──────────────
Fine-tunes two YOLOv8 models:
  1. Vehicle detector  – COCO-pretrained yolov8n, fine-tuned on vehicle classes
  2. License plate detector – trained from scratch / fine-tuned on an Indian plate dataset

Run this ONCE before starting the Flask app if you want custom models.
Otherwise, the app falls back to the pretrained yolov8n weights automatically.

Usage:
    python train_model.py --task vehicle   # fine-tune vehicle detector
    python train_model.py --task plate     # train plate detector
    python train_model.py --task both      # both (default)
"""

import argparse, os
from pathlib import Path

# ── check ultralytics ─────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip install ultralytics")
    exit(1)

# ── Dataset helpers ───────────────────────────────────────────────────────────

def download_vehicle_dataset():
    """
    Use Roboflow public dataset or convert COCO.
    Here we generate a minimal data.yaml that points YOLOv8 to the COCO classes
    we care about (vehicles only), using the COCO-pretrained weights.
    For production, replace with your own annotated dataset.
    """
    yaml_content = """
path: ./datasets/vehicles
train: images/train
val: images/val

nc: 6
names: ['bicycle', 'car', 'motorcycle', 'bus', 'truck', 'train']
"""
    os.makedirs('datasets/vehicles/images/train', exist_ok=True)
    os.makedirs('datasets/vehicles/images/val', exist_ok=True)
    os.makedirs('datasets/vehicles/labels/train', exist_ok=True)
    os.makedirs('datasets/vehicles/labels/val', exist_ok=True)
    with open('datasets/vehicles/data.yaml', 'w') as f:
        f.write(yaml_content)
    print("[INFO] Created vehicle dataset scaffold at datasets/vehicles/")
    print("[INFO] → Place your training images in datasets/vehicles/images/train/")
    print("[INFO] → Place YOLO-format label .txt files in datasets/vehicles/labels/train/")


def download_plate_dataset():
    """
    Downloads the Indian number plate dataset via kagglehub and creates a YOLO yaml.
    Note: the Kaggle dataset is image-classification, not detection.
    For detection, you need annotated bounding boxes. 
    This function sets up the structure and prints guidance.
    """
    try:
        import kagglehub
        import pandas as pd
        print("[INFO] Downloading Indian Number Plates dataset from Kaggle …")
        dataset_dir = Path(kagglehub.dataset_download("dataclusterlabs/indian-number-plates-dataset"))
        tabular_exts = {'.csv', '.tsv', '.json', '.jsonl', '.parquet', '.feather', '.xlsx', '.xls'}
        tabular_files = [p for p in dataset_dir.rglob('*') if p.is_file() and p.suffix.lower() in tabular_exts]

        if tabular_files:
            source_file = sorted(tabular_files)[0]
            if source_file.suffix.lower() == '.csv':
                df = pd.read_csv(source_file)
            elif source_file.suffix.lower() == '.tsv':
                df = pd.read_csv(source_file, sep='\t')
            elif source_file.suffix.lower() in {'.json', '.jsonl'}:
                df = pd.read_json(source_file, lines=(source_file.suffix.lower() == '.jsonl'))
            else:
                # Fallback for parquet/feather/excel files.
                if source_file.suffix.lower() == '.parquet':
                    df = pd.read_parquet(source_file)
                elif source_file.suffix.lower() == '.feather':
                    df = pd.read_feather(source_file)
                else:
                    df = pd.read_excel(source_file)

            os.makedirs('data', exist_ok=True)
            df.to_csv('data/plate_db.csv', index=False)
            print(f"[INFO] Saved {len(df)} plate records to data/plate_db.csv (source: {source_file.name})")
        else:
            print(f"[WARN] Kaggle dataset downloaded to {dataset_dir}, but no tabular metadata file was found.")
            print("[INFO] You can still use the images for training; add your own annotations for detection.")
    except Exception as e:
        print(f"[WARN] Kaggle download failed: {e}")

    yaml_content = """
path: ./datasets/plates
train: images/train
val: images/val

nc: 1
names: ['license_plate']
"""
    os.makedirs('datasets/plates/images/train', exist_ok=True)
    os.makedirs('datasets/plates/images/val', exist_ok=True)
    os.makedirs('datasets/plates/labels/train', exist_ok=True)
    os.makedirs('datasets/plates/labels/val', exist_ok=True)
    with open('datasets/plates/data.yaml', 'w') as f:
        f.write(yaml_content)
    print("[INFO] Created plate dataset scaffold at datasets/plates/")
    print("[INFO] → Add annotated plate images + YOLO-format labels to train/val dirs")
    print("[INFO] → Recommended public dataset: https://universe.roboflow.com (search 'license plate')")


# ── Training ──────────────────────────────────────────────────────────────────

def train_vehicle_model(epochs=50, imgsz=640, batch=16, model_size='m'):
    """
    Fine-tune YOLOv8 for vehicle detection.
    
    Args:
        epochs: Number of training epochs
        imgsz: Image size for training (default 640)
        batch: Batch size (default 16, reduce if OOM)
        model_size: Model architecture - 'n' (nano), 's' (small), 'm' (medium, recommended), 'l' (large)
    """
    _assert_dataset_ready('datasets/vehicles', 'vehicle')
    print(f"\n[TRAIN] Starting vehicle model training with YOLOv8{model_size} …")
    
    # Use medium model for better accuracy (trade-off: slower inference)
    model_variants = {'n': 'yolov8n.pt', 's': 'yolov8s.pt', 'm': 'yolov8m.pt', 'l': 'yolov8l.pt', 'x': 'yolov8x.pt'}
    base_model = model_variants.get(model_size, 'yolov8m.pt')
    
    print(f"[INFO] Using {base_model} as base model")
    model = YOLO(base_model)

    # Only keep vehicle classes from COCO during transfer
    results = model.train(
        data='datasets/vehicles/data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='vehicle_detector',
        project='runs/train',
        patience=10,
        save=True,
        device='0' if _gpu_available() else 'cpu',
        # freeze initial layers for better transfer learning
        freeze=10 if model_size in ('n', 's') else 15,
        # augmentation
        mosaic=1.0,
        mixup=0.1,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    best_path = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f"[TRAIN] Vehicle model saved to: {best_path}")
    return str(best_path)


def train_plate_model(epochs=80, imgsz=640, batch=16, model_size='s'):
    """
    Train license plate detector.
    
    Args:
        epochs: Number of training epochs
        imgsz: Image size for training (default 640)
        batch: Batch size (default 16, reduce if OOM)
        model_size: Model architecture - 'n' (nano), 's' (small, recommended for plates), 'm' (medium)
    """
    _assert_dataset_ready('datasets/plates', 'plate')
    print(f"\n[TRAIN] Starting license plate model training with YOLOv8{model_size} …")
    
    # Use small/medium model for better accuracy on small objects like plates
    model_variants = {'n': 'yolov8n.pt', 's': 'yolov8s.pt', 'm': 'yolov8m.pt', 'l': 'yolov8l.pt'}
    base_model = model_variants.get(model_size, 'yolov8s.pt')
    
    print(f"[INFO] Using {base_model} as base model for plate detection")
    model = YOLO(base_model)  # pretrained backbone

    results = model.train(
        data='datasets/plates/data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='plate_detector',
        project='runs/train',
        patience=15,
        save=True,
        device='0' if _gpu_available() else 'cpu',
        # plates are small objects – optimize for small object detection
        close_mosaic=10,
        hsv_h=0.01,
        hsv_s=0.3,
        perspective=0.001,
        # Better augmentation for small objects
        scale=0.5,
        translate=0.1,
        degrees=5,
    )
    best_path = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f"[TRAIN] Plate model saved to: {best_path}")
    # Copy to project root for easy loading
    import shutil
    shutil.copy(str(best_path), 'license_plate_detector.pt')
    print("[TRAIN] Copied to license_plate_detector.pt")
    return str(best_path)


def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _count_images(dir_path: str) -> int:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    p = Path(dir_path)
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob('*') if f.is_file() and f.suffix.lower() in exts)


def _count_labels(dir_path: str) -> int:
    p = Path(dir_path)
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob('*.txt') if f.is_file())


def _dataset_issues(base_dir: str) -> list[str]:
    train_images = _count_images(os.path.join(base_dir, 'images', 'train'))
    val_images = _count_images(os.path.join(base_dir, 'images', 'val'))
    train_labels = _count_labels(os.path.join(base_dir, 'labels', 'train'))
    val_labels = _count_labels(os.path.join(base_dir, 'labels', 'val'))

    issues = []
    if train_images == 0:
        issues.append(f"- No training images found in {base_dir}/images/train")
    if val_images == 0:
        issues.append(f"- No validation images found in {base_dir}/images/val")
    if train_labels == 0:
        issues.append(f"- No training label files found in {base_dir}/labels/train")
    if val_labels == 0:
        issues.append(f"- No validation label files found in {base_dir}/labels/val")
    return issues


def _assert_dataset_ready(base_dir: str, name: str):
    issues = _dataset_issues(base_dir)
    if issues:
        msg = "\n".join([
            f"[ERROR] {name.capitalize()} dataset is not ready:",
            *issues,
            "[HINT] Add images + YOLO .txt labels, then retry training.",
        ])
        raise RuntimeError(msg)


# ── Validate / test ───────────────────────────────────────────────────────────

def validate_model(model_path, data_yaml):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    print(f"[VAL] mAP50: {metrics.box.map50:.4f}")
    print(f"[VAL] mAP50-95: {metrics.box.map:.4f}")
    return metrics


def test_on_image(model_path, image_path):
    model = YOLO(model_path)
    results = model(image_path)
    results[0].show()
    results[0].save(filename='test_result.jpg')
    print("[TEST] Result saved to test_result.jpg")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train traffic detection models')
    parser.add_argument('--task', choices=['vehicle','plate','both','setup'], default='both')
    parser.add_argument('--epochs-vehicle', type=int, default=50)
    parser.add_argument('--epochs-plate',   type=int, default=80)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--model-vehicle', choices=['n','s','m','l','x'], default='m', help='Vehicle model size')
    parser.add_argument('--model-plate', choices=['n','s','m','l'], default='s', help='Plate model size')
    parser.add_argument('--validate', type=str, default=None, help='Path to model to validate')
    parser.add_argument('--test-image', type=str, default=None)
    args = parser.parse_args()

    if args.validate:
        yaml = 'datasets/vehicles/data.yaml' if 'vehicle' in args.validate else 'datasets/plates/data.yaml'
        validate_model(args.validate, yaml)

    elif args.test_image:
        model_path = 'runs/train/vehicle_detector/weights/best.pt'
        test_on_image(model_path, args.test_image)

    elif args.task == 'setup':
        download_vehicle_dataset()
        download_plate_dataset()

    elif args.task == 'vehicle':
        download_vehicle_dataset()
        train_vehicle_model(epochs=args.epochs_vehicle, imgsz=args.imgsz, batch=args.batch, model_size=args.model_vehicle)

    elif args.task == 'plate':
        download_plate_dataset()
        train_plate_model(epochs=args.epochs_plate, imgsz=args.imgsz, batch=args.batch, model_size=args.model_plate)

    else:  # both
        download_vehicle_dataset()
        download_plate_dataset()
        trained_any = False

        vehicle_issues = _dataset_issues('datasets/vehicles')
        if vehicle_issues:
            print("[WARN] Skipping vehicle training because dataset is incomplete:")
            print("\n".join(vehicle_issues))
        else:
            train_vehicle_model(epochs=args.epochs_vehicle, imgsz=args.imgsz, batch=args.batch)
            trained_any = True

        plate_issues = _dataset_issues('datasets/plates')
        if plate_issues:
            print("[WARN] Skipping plate training because dataset is incomplete:")
            print("\n".join(plate_issues))
        else:
            train_plate_model(epochs=args.epochs_plate, imgsz=args.imgsz, batch=args.batch)
            trained_any = True

        if trained_any:
            print("\n[DONE] Training finished. Start the app with: python app.py")
        else:
            print("\n[INFO] No model was trained. Run --task setup, add images + labels, then retry.")
