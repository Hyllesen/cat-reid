# Copilot Instructions

## Project Overview

This is a **Cat Re-Identification pipeline** for a home camera system. It identifies individual resident cats from an RTSP stream in real-time. The pipeline is split into four sequential phases, each with its own script.

## Architecture & Data Flow

```
Phase 1  ingest.py           RTSP stream → YOLO detection → JPEG crops
            ↓  data/raw_crops/cat_TIMESTAMP_CONF.jpg
Phase 2  extract_features.py crops → MobileNetV3 embeddings
            ↓  embeddings.pkl  {"embeddings": (N, 1280) float32, "paths": [str]}
Phase 3  cluster.py          DBSCAN clustering → organized directories + knowledge base
            ↓  data/organized/cluster_N/*.jpg  +  cat_map.json  +  centroids.pkl
Phase 4  identify.py         RTSP stream → real-time detection + name overlay
```

**Phase 3 one-way destructive step**: `cluster.py` moves (not copies) images from `data/raw_crops/` into `data/organized/`. Re-running it on an empty `raw_crops/` directory is safe (missing files are silently skipped), but re-running it against fresh crops after Phase 2 will correctly reorganize them.

## Running the Pipeline

```bash
# Activate venv first
source .venv/bin/activate

# Phase 1 — live ingestion (requires RTSP_URL in .env)
python ingest.py

# Phase 2 — feature extraction (run after crops are collected)
python extract_features.py

# Phase 3 — cluster and organize
python cluster.py

# Phase 4 — real-time identification (requires RTSP_URL in .env)
python identify.py
```

There are no tests, linters, or CI configured in this project.

## Configuration

All config is loaded from `.env` via `python-dotenv`. Copy `.env.example` to get started.

| Variable | Used by | Default | Notes |
|---|---|---|---|
| `RTSP_URL` | Phase 1, 4 | *(required)* | Full RTSP URL |
| `MODEL_PATH` | Phase 1 | `yolo26n.mlpackage` | CoreML YOLO model (ANE accelerated) |
| `COREML_MODEL_PATH` | Phase 4 | `yolo26n.mlpackage` | CoreML model for ANE |
| `OUTPUT_DIR` | Phase 1 | `./data/raw_crops/` | Crop save directory |
| `CONFIDENCE_THRESHOLD` | Phase 1, 4 | `0.7` | YOLO detection threshold |
| `FRAME_SKIP` | Phase 1, 4 | `1` | Process every Nth frame |
| `MAX_RECONNECT_ATTEMPTS` | Phase 1, 4 | `10` | RTSP reconnect limit |
| `RECONNECT_DELAY` | Phase 1, 4 | `5` | Seconds between reconnects |
| `DBSCAN_EPS` | Phase 3 | `0.20` | Cosine distance threshold |
| `DBSCAN_MIN_SAMPLES` | Phase 3 | `5` | Min points to form cluster |
| `RESIDENT_MIN_IMAGES` | Phase 3 | `5` | Cluster size for "resident cat" |
| `MATCH_THRESHOLD` | Phase 4 | `0.25` | Cosine distance for ID match |
| `DEBUG` | All | `False` | Enables DEBUG log level |

## Key Conventions

### Config pattern
All scripts follow the same pattern — no config classes:
```python
from dotenv import load_dotenv
load_dotenv()
SOME_VALUE = float(os.getenv("SOME_VALUE", "0.7"))
DEBUG = os.getenv("DEBUG", "False").lower() in ("1", "true", "yes")
```

### Logging pattern
All scripts use the same setup at module level:
```python
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
```

### Device selection (MPS → CPU)
```python
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
```
Always MPS-first; never CUDA. This system targets Apple Silicon exclusively.

### Embeddings are always L2-normalized float32
`F.normalize(feat, p=2, dim=1)` is applied after every model forward pass. Centroids stored in `centroids.pkl` are also L2-normalized. This means cosine distance equals half the squared Euclidean distance — both metrics give equivalent ranking.

### sklearn cosine metric requires float64
The DBSCAN call in `cluster.py` casts embeddings to `float64` before passing to sklearn to avoid overflow warnings in sklearn's cosine matmul. All other torch/timm code uses `float32`.

### RTSP stream loop structure
Both `ingest.py` and `identify.py` share the same two-level loop pattern:
- **Outer loop**: Handles reconnection with counter + delay
- **Inner loop**: Reads frames; breaks on read failure to trigger reconnect
- `KeyboardInterrupt` is caught inside the inner try/except; `cap.release()` always runs in `finally`

### Cat class ID
`CAT_CLASS_ID = 15` — the COCO dataset index for "cat". Used in both Phase 1 and Phase 4.

### `cat_map.json` key type
Keys in the JSON file are strings (JSON limitation). When loading, always cast: `{int(k): v for k, v in raw.items()}`. `centroids.pkl` keys are native Python `int`.

## Artifacts & Model Files

| File | Description |
|---|---|
| `yolo26n.pt` | YOLO detection model (PyTorch) — used by Phase 1 |
| `yolo26n.mlpackage/` | YOLO CoreML export — used by Phase 4 for ANE acceleration |
| `embeddings.pkl` | Phase 2 output: `{"embeddings": ndarray, "paths": [str]}` |
| `cat_map.json` | Phase 3 output: cluster metadata + centroid (as list) per cat |
| `centroids.pkl` | Phase 3 output: `{cluster_id: centroid_ndarray}` for fast Phase 4 matching |
| `data/raw_crops/` | Phase 1 output; consumed (moved) by Phase 3 |
| `data/organized/` | Phase 3 output; organized by cluster |

To regenerate the CoreML model:
```python
from ultralytics import YOLO
YOLO("yolo26n.pt").export(format="coreml")
```
