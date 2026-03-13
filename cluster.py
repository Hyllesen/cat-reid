"""
cluster.py — Phase 3: DBSCAN Clustering & Organisation

Loads embeddings from embeddings.pkl, clusters them with DBSCAN, and:
  - Moves source images into /data/organized/cluster_N/ (or cluster_noise/)
  - Writes cat_map.json  — one entry per resident cluster (>5 images) with
    a default name and the cluster centroid embedding
  - Writes centroids.pkl — dict mapping cluster_id -> centroid np.ndarray
    for use in Phase 4 real-time matching

Usage:
    python cluster.py

Tune via env vars (optional):
    DBSCAN_EPS          float, default 0.20
    DBSCAN_MIN_SAMPLES  int,   default 5
    RESIDENT_MIN_IMAGES int,   default 5  (clusters with > this many images
                                           are tagged as resident cats)
"""

import json
import logging
import os
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDINGS_PATH = Path("embeddings.pkl")
ORGANIZED_DIR   = Path("data/organized/")
CAT_MAP_PATH    = Path("cat_map.json")
CENTROIDS_PATH  = Path("centroids.pkl")

DBSCAN_EPS          = float(os.getenv("DBSCAN_EPS",          "0.20"))
DBSCAN_MIN_SAMPLES  = int(os.getenv("DBSCAN_MIN_SAMPLES",   "5"))
RESIDENT_MIN_IMAGES = int(os.getenv("RESIDENT_MIN_IMAGES",  "5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Suppress sklearn cosine-metric precision warnings on float32 data
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cluster_dir(label: int) -> Path:
    name = f"cluster_{label}" if label != -1 else "cluster_noise"
    return ORGANIZED_DIR / name


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Mean of embeddings, L2-normalised to a unit vector."""
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load embeddings
    log.info("Loading embeddings from %s", EMBEDDINGS_PATH)
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)

    embeddings: np.ndarray = data["embeddings"].astype(np.float64)  # float64 for sklearn cosine
    paths: list[str]       = data["paths"]
    n = len(paths)
    log.info("Loaded %d embeddings, shape %s", n, embeddings.shape)

    # DBSCAN
    log.info("Running DBSCAN (eps=%.2f, min_samples=%d, metric=cosine) …",
             DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                metric="cosine", n_jobs=-1)
    labels: np.ndarray = db.fit_predict(embeddings)

    unique_labels = sorted(set(labels))
    n_clusters = sum(1 for l in unique_labels if l != -1)
    n_noise    = int((labels == -1).sum())
    log.info("Found %d clusters + %d noise points", n_clusters, n_noise)

    # Create output directories and move images
    ORGANIZED_DIR.mkdir(parents=True, exist_ok=True)
    for label in unique_labels:
        cluster_dir(label).mkdir(parents=True, exist_ok=True)

    moved = 0
    for src_str, label in zip(paths, labels):
        src = Path(src_str)
        if not src.exists():
            log.warning("Source file missing, skipping: %s", src)
            continue
        dst = cluster_dir(label) / src.name
        shutil.move(str(src), str(dst))
        moved += 1

    log.info("Moved %d / %d images into %s", moved, n, ORGANIZED_DIR)

    # Build cat_map.json and centroids
    cat_map: dict   = {}
    centroids: dict = {}

    for label in unique_labels:
        if label == -1:
            continue  # noise is not a resident cat

        mask = labels == label
        cluster_size = int(mask.sum())
        cluster_embs = embeddings[mask].astype(np.float32)
        centroid     = compute_centroid(cluster_embs)

        centroids[int(label)] = centroid

        is_resident = cluster_size > RESIDENT_MIN_IMAGES
        entry = {
            "cluster_id":  int(label),
            "name":        f"Unknown Cat {label}",
            "image_count": cluster_size,
            "is_resident": is_resident,
            "directory":   str(cluster_dir(label)),
            "centroid":    centroid.tolist(),
        }
        cat_map[str(label)] = entry

        status = "resident" if is_resident else "transient"
        log.info("  cluster_%d: %d images (%s)", label, cluster_size, status)

    # Noise summary
    noise_count = int((labels == -1).sum())
    if noise_count:
        log.info("  cluster_noise: %d images (unassigned)", noise_count)

    # Save cat_map.json
    with open(CAT_MAP_PATH, "w") as f:
        json.dump(cat_map, f, indent=2)
    log.info("Saved cat_map.json (%d cluster entries)", len(cat_map))

    # Save centroids.pkl
    with open(CENTROIDS_PATH, "wb") as f:
        pickle.dump(centroids, f)
    log.info("Saved centroids.pkl (%d centroids)", len(centroids))

    # Summary
    resident_count = sum(1 for v in cat_map.values() if v["is_resident"])
    log.info("Done. %d resident cats identified.", resident_count)


if __name__ == "__main__":
    main()
