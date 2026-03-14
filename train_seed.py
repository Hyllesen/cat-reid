"""
train_seed.py
Fine-tunes yolo26n-cls.pt on the seed cat-identity dataset.
"""

import random
import shutil
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH   = "yolo26n-cls.pt"
DATA_DIR     = Path("seed_data")
OUTPUT_NAME  = "seed_model"
DEVICE       = "mps"
EPOCHS       = 30
IMG_SIZE     = 224
VAL_SPLIT    = 0.2   # fraction of each class held out for val

# High LR suits a small dataset where we want fast feature adaptation.
# Cosine decay brings it down smoothly over 30 epochs.
LR0          = 0.01
LRF          = 0.1   # final lr = LR0 * LRF

# Light augmentation — avoids overfitting on 193 samples
AUGMENT_KWARGS = dict(
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    flipud=0.0,
    fliplr=0.5,
    degrees=10,
    translate=0.1,
    scale=0.3,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _create_val_split():
    """
    Copies ~VAL_SPLIT fraction of each class from train/ into val/.
    Returns True if val/ was created by this run (so we can clean up later).
    """
    val_dir = DATA_DIR / "val"
    if val_dir.exists():
        print("val/ already exists — skipping split creation.")
        return False

    train_dir = DATA_DIR / "train"
    random.seed(42)
    total_moved = 0

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        n_val = max(1, int(len(images) * VAL_SPLIT))
        val_images = random.sample(images, n_val)

        dest_class = val_dir / class_dir.name
        dest_class.mkdir(parents=True, exist_ok=True)
        for img in val_images:
            shutil.copy(img, dest_class / img.name)
        print(f"  val split — {class_dir.name}: {n_val}/{len(images)} images")
        total_moved += n_val

    print(f"Created val/ with {total_moved} images total.\n")
    return True


def train():
    val_created_here = _create_val_split()

    model = YOLO(MODEL_PATH)

    project_dir = Path.cwd() / "runs" / "classify"

    results = model.train(
        data=str(DATA_DIR.resolve()),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=DEVICE,
        lr0=LR0,
        lrf=LRF,
        cos_lr=True,
        project=str(project_dir),
        name=OUTPUT_NAME,
        exist_ok=True,
        **AUGMENT_KWARGS,
    )

    # Copy the best weights to the repo root for easy access
    best_weights = project_dir / OUTPUT_NAME / "weights" / "best.pt"
    dest = Path("seed_model.pt")
    if best_weights.exists():
        shutil.copy(best_weights, dest)
        print(f"\nSaved: {dest.resolve()}")
    else:
        print(f"\n[warn] best.pt not found at {best_weights}; check runs/ folder.")

    if val_created_here:
        shutil.rmtree(DATA_DIR / "val")
        print("Cleaned up temporary val/ split.")

    return results


if __name__ == "__main__":
    train()
