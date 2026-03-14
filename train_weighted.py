"""
train_weighted.py

Trains yolo26n-cls.pt on the sorted cat dataset using class-weighted
CrossEntropyLoss to handle the imbalance (orange >> black_white >> grey).

Pipeline:
  1. Build stratified train/val split from dataset/sorted/
  2. Compute inverse-frequency class weights
  3. Train with a custom weighted loss trainer (50 epochs, mps)
  4. Print confusion matrix on the val split
  5. Export best weights to CoreML
"""

import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer

# ── Config ────────────────────────────────────────────────────────────────────
SORTED_DIR   = Path("/Volumes/external-nvme256gb/cat-reid/dataset/sorted")
MODEL_PATH   = "yolo26n-cls.pt"
PROJECT      = str(Path.cwd() / "runs" / "classify")
RUN_NAME     = "weighted_model"
DEVICE       = "mps"
EPOCHS       = 50
IMG_SIZE     = 224
BATCH        = 128
VAL_FRACTION = 0.15   # 15 % of each class held out for val
CONF_THRESH  = 0.0    # evaluate all val samples
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Classes to train on (exclude 'unsure')
TRAIN_CLASSES = {"orange", "black_white", "grey"}

# ── Weighted loss ─────────────────────────────────────────────────────────────

class WeightedClassificationLoss:
    """Drop-in replacement for v8ClassificationLoss with class weights."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(
            preds, batch["cls"], weight=self.weight, reduction="mean"
        )
        return loss, loss.detach()


class WeightedClassificationTrainer(ClassificationTrainer):
    """ClassificationTrainer that injects inverse-frequency class weights."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        self._class_weights = class_weights
        super().__init__(**kwargs)

    def set_model_attributes(self):
        super().set_model_attributes()
        w = self._class_weights.to(self.device)
        self.model.criterion = WeightedClassificationLoss(weight=w)
        # Log weights so they're visible in the run summary
        names = self.data.get("names", {})
        print("\n── Class weights ─────────────────────────────────────")
        for idx, name in sorted(names.items()):
            print(f"  [{idx}] {name:20s}: {w[idx]:.4f}")
        print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_class_weights(counts: dict[str, int]) -> tuple[list[str], torch.Tensor]:
    """
    Inverse-frequency weighting, normalised so the mean weight == 1.
    Returns (sorted class names, weight tensor in that order).
    """
    classes = sorted(counts.keys())
    freqs   = torch.tensor([counts[c] for c in classes], dtype=torch.float32)
    weights = 1.0 / freqs
    weights = weights / weights.mean()   # normalise
    return classes, weights


def build_split(source_dir: Path, tmp_dir: Path, val_fraction: float, seed: int = 42):
    """
    Copies images from source_dir/<class>/ into tmp_dir/train/<class>/
    and tmp_dir/val/<class>/.  Returns per-class sample counts.
    """
    random.seed(seed)
    counts: dict[str, int] = {}

    for cls_dir in sorted(source_dir.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in TRAIN_CLASSES:
            continue

        images = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        random.shuffle(images)
        n_val   = max(1, int(len(images) * val_fraction))
        val_set = set(images[:n_val])
        counts[cls_dir.name] = len(images)

        for split in ("train", "val"):
            (tmp_dir / split / cls_dir.name).mkdir(parents=True, exist_ok=True)

        for img in images:
            dest_split = "val" if img in val_set else "train"
            shutil.copy(img, tmp_dir / dest_split / cls_dir.name / img.name)

        print(f"  {cls_dir.name:20s}: {len(images) - n_val} train | {n_val} val")

    return counts


def confusion_matrix_report(model: YOLO, val_dir: Path, class_names: list[str]):
    """Run inference on val set and print a confusion matrix."""
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    for true_idx, cls_name in enumerate(class_names):
        cls_val_dir = val_dir / cls_name
        if not cls_val_dir.exists():
            continue
        images = [p for p in cls_val_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if not images:
            continue
        results = model.predict(
            source=[str(p) for p in images],
            device=DEVICE,
            verbose=False,
            stream=True,
        )
        for res in results:
            pred_name = res.names[res.probs.top1]
            pred_idx  = name_to_idx.get(pred_name, true_idx)
            matrix[true_idx][pred_idx] += 1

    col_w = 14
    header = f"{'':20s}" + "".join(f"{n:>{col_w}}" for n in class_names)
    print("\n── Confusion Matrix (rows=actual, cols=predicted) ────────")
    print(header)
    print("-" * len(header))
    for i, row_name in enumerate(class_names):
        row_str = f"{row_name:20s}" + "".join(f"{matrix[i][j]:>{col_w}}" for j in range(n))
        print(row_str)
    print()

    # Per-class recall
    print("── Per-class recall ──────────────────────────────────────")
    for i, name in enumerate(class_names):
        total  = matrix[i].sum()
        recall = matrix[i][i] / total if total else 0.0
        print(f"  {name:20s}: {recall:.1%}  ({matrix[i][i]}/{total})")
    print()


# ── Checkpoint sanitiser ──────────────────────────────────────────────────────

def _strip_criterion_from_checkpoint(pt_path: Path):
    """
    Removes any custom criterion object from a saved Ultralytics checkpoint.

    Ultralytics pickles the full model object (including model.criterion) into
    the .pt file.  When WeightedClassificationLoss is defined in __main__ the
    pickle stream records it as 'c__main__\\nWeightedClassificationLoss', which
    causes an AttributeError when the checkpoint is loaded in any other script.

    This function loads the file (with WeightedClassificationLoss in scope so
    unpickling succeeds), sets criterion=None on every top-level model object,
    then re-saves — producing a portable checkpoint.
    """
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    changed = False
    for val in ckpt.values():
        if hasattr(val, "criterion") and val.criterion is not None:
            val.criterion = None
            changed = True
    if changed:
        torch.save(ckpt, str(pt_path))
        print(f"  Stripped criterion from {pt_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Build temporary train/val split
    tmp_dir = Path(tempfile.mkdtemp(prefix="cat_reid_split_"))
    print("── Building stratified train/val split ───────────────────")
    counts = build_split(SORTED_DIR, tmp_dir, VAL_FRACTION)

    # 2. Compute class weights
    class_names, weights = compute_class_weights(counts)
    print(f"\nClasses : {class_names}")
    print(f"Counts  : {[counts[c] for c in class_names]}")
    print(f"Weights : {weights.tolist()}\n")

    # 3. Train
    trainer = WeightedClassificationTrainer(
        class_weights=weights,
        overrides=dict(
            model=MODEL_PATH,
            data=str(tmp_dir),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            project=PROJECT,
            name=RUN_NAME,
            exist_ok=True,
            cos_lr=True,
            lr0=0.001,
            lrf=0.1,
            dropout=0.2,       # regularise — dataset is medium-sized
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.4,
            fliplr=0.5,
            degrees=10,
            scale=0.3,
            verbose=True,
        ),
    )
    trainer.train()

    # Strip the custom criterion from every saved checkpoint so the .pt files
    # can be loaded anywhere without needing WeightedClassificationLoss in scope.
    run_weights_dir = Path(PROJECT) / RUN_NAME / "weights"
    for pt_path in run_weights_dir.glob("*.pt"):
        _strip_criterion_from_checkpoint(pt_path)

    # 4. Load best weights and run confusion matrix on val split
    best_pt = run_weights_dir / "best.pt"
    print(f"\nLoading best weights: {best_pt}")
    best_model = YOLO(str(best_pt))

    confusion_matrix_report(best_model, tmp_dir / "val", class_names)

    # 5. Export to CoreML
    print("── Exporting to CoreML ───────────────────────────────────")
    best_model.export(format="coreml", imgsz=IMG_SIZE, nms=False)
    coreml_path = best_pt.with_suffix(".mlpackage")
    if coreml_path.exists():
        print(f"CoreML model saved: {coreml_path}")
    else:
        # Ultralytics may emit .mlpackage next to the .pt
        candidates = list(best_pt.parent.glob("*.mlpackage"))
        if candidates:
            print(f"CoreML model saved: {candidates[0]}")
        else:
            print("[warn] .mlpackage not found — check the run directory.")

    # 6. Clean up temp split
    shutil.rmtree(tmp_dir)
    print("Temporary split removed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
