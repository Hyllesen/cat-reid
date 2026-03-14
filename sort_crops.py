"""
sort_crops.py
Classifies images in dataset/raw_crops/ using seed_model.pt and moves
each image into dataset/sorted/[class]/ or dataset/sorted/unsure/.
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH        = "seed_model.pt"
INPUT_DIR         = Path("dataset/raw_crops")
OUTPUT_DIR        = Path("dataset/sorted")
DEVICE            = "mps"
CONFIDENCE_THRESH = 0.6
BATCH_SIZE        = 64
IMAGE_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sort_crops():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    images = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in '{INPUT_DIR}'")
        return

    print(f"Found {len(images)} images — loading model...")
    model = YOLO(MODEL_PATH)

    counts = {}
    # Process in batches for throughput; pass plain str paths to avoid MPS quirks
    for batch_start in range(0, len(images), BATCH_SIZE):
        batch = images[batch_start : batch_start + BATCH_SIZE]
        results = model.predict(
            source=[str(p) for p in batch],
            device=DEVICE,
            verbose=False,
        )

        for img_path, result in zip(batch, results):
            probs = result.probs
            conf  = float(probs.top1conf)
            label = result.names[probs.top1]

            dest_class = label if conf >= CONFIDENCE_THRESH else "unsure"
            dest_dir   = OUTPUT_DIR / dest_class
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_file = dest_dir / img_path.name
            # Avoid clobbering if a name collision exists
            if dest_file.exists():
                dest_file = dest_dir / f"{img_path.stem}_dup{img_path.suffix}"

            shutil.move(str(img_path), dest_file)
            counts[dest_class] = counts.get(dest_class, 0) + 1

        done = min(batch_start + BATCH_SIZE, len(images))
        print(f"  {done}/{len(images)} processed...", end="\r")

    print()
    print("\n── Summary ──────────────────────────")
    for cls in sorted(counts):
        print(f"  {cls:20s}: {counts[cls]:>5} images")
    print(f"  {'TOTAL':20s}: {sum(counts.values()):>5} images")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    sort_crops()
