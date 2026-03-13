"""
extract_features.py — Phase 2: Feature Extraction

Loads all cat-crop JPEGs from /data/raw_crops/, runs them through a
pretrained MobileNetV3-Large backbone (timm), and saves L2-normalized
1D embeddings to embeddings.pkl.

Usage:
    python extract_features.py

Output:
    embeddings.pkl  — dict with keys:
        "embeddings"  : np.ndarray  shape (N, 960), float32, L2-normalized
        "paths"       : list[str]   len N, absolute paths to source images
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CROPS_DIR = Path("./data/raw_crops/")
OUTPUT_PATH = Path("embeddings.pkl")
MODEL_NAME = "mobilenetv3_large_100"
BATCH_SIZE = 64

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CropDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform):
        self.paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), str(self.paths[idx])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Device selection: prefer MPS (Apple Silicon), fall back to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log.info("MPS not available — using CPU")

    # Load model without classifier head to get raw embeddings
    log.info("Loading model: %s", MODEL_NAME)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)

    # Build preprocessing pipeline matching model's training config
    data_config = resolve_data_config({}, model=model)
    transform = create_transform(**data_config)

    # Gather all JPEG crops
    image_paths = sorted(CROPS_DIR.glob("*.jpg"))
    if not image_paths:
        log.error("No .jpg files found in %s", CROPS_DIR)
        return
    log.info("Found %d crop images", len(image_paths))

    dataset = CropDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)

    all_embeddings: list[np.ndarray] = []
    all_paths: list[str] = []

    log.info("Extracting embeddings (batch_size=%d) …", BATCH_SIZE)
    with torch.no_grad():
        for batch_idx, (images, paths) in enumerate(loader):
            images = images.to(device)
            feats = model(images)                          # (B, 960)
            feats = F.normalize(feats, p=2, dim=1)        # L2-normalize
            all_embeddings.append(feats.cpu().numpy())
            all_paths.extend(paths)

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
                log.info("  processed %d / %d images", len(all_paths), len(image_paths))

    embeddings = np.vstack(all_embeddings).astype(np.float32)  # (N, 960)
    log.info("Embedding matrix shape: %s", embeddings.shape)

    payload = {"embeddings": embeddings, "paths": all_paths}
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    log.info("Saved embeddings to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
