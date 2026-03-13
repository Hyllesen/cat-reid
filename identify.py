"""
identify.py — Phase 4: Real-Time Cat Re-Identification

Runs the RTSP stream through YOLO (CoreML, ANE-accelerated) for cat detection,
extracts a MobileNetV3 embedding per crop on MPS, compares against Phase 3
cluster centroids, and overlays the matched cat's name (or "New Visitor") on
the live preview window.

Config via .env (see .env.example):
    RTSP_URL, MODEL_PATH, OUTPUT_DIR, CONFIDENCE_THRESHOLD,
    MAX_RECONNECT_ATTEMPTS, RECONNECT_DELAY, FRAME_SKIP, DEBUG

Phase-4-specific env vars:
    COREML_MODEL_PATH   path to CoreML model   (default: yolo26n.mlpackage)
    CAT_MAP_PATH        path to cat_map.json   (default: cat_map.json)
    CENTROIDS_PATH      path to centroids.pkl  (default: centroids.pkl)
    MATCH_THRESHOLD     cosine distance cutoff  (default: 0.25)

Press 'q' in the preview window or Ctrl-C to stop.
"""

import json
import logging
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from scipy.spatial.distance import cosine as cosine_dist
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from ultralytics import YOLO

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RTSP_URL               = os.getenv("RTSP_URL", "")
CONFIDENCE_THRESHOLD   = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
MAX_RECONNECT_ATTEMPTS = int(os.getenv("MAX_RECONNECT_ATTEMPTS", "10"))
RECONNECT_DELAY        = float(os.getenv("RECONNECT_DELAY", "5"))
FRAME_SKIP             = int(os.getenv("FRAME_SKIP", "1"))
DEBUG                  = os.getenv("DEBUG", "False").lower() in ("1", "true", "yes")

COREML_MODEL_PATH = os.getenv("COREML_MODEL_PATH", "yolo26n.mlpackage")
CAT_MAP_PATH      = Path(os.getenv("CAT_MAP_PATH",     "cat_map.json"))
CENTROIDS_PATH    = Path(os.getenv("CENTROIDS_PATH",   "centroids.pkl"))
MATCH_THRESHOLD   = float(os.getenv("MATCH_THRESHOLD", "0.25"))

EMBED_MODEL_NAME = "mobilenetv3_large_100"
CAT_CLASS_ID     = 15   # COCO cat index
CROP_MARGIN      = 0.10

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------
BOX_COLOR_KNOWN   = (0, 255, 0)    # green  — known resident
BOX_COLOR_VISITOR = (0, 165, 255)  # orange — new visitor
BOX_THICKNESS     = 2
FONT              = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE        = 0.65
FONT_THICKNESS    = 2


# ---------------------------------------------------------------------------
# Embedding model  (loaded once, kept on MPS)
# ---------------------------------------------------------------------------

class EmbeddingModel:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            log.info("Embedding model using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            log.info("MPS unavailable — embedding model using CPU")

        self.model = timm.create_model(EMBED_MODEL_NAME, pretrained=True, num_classes=0)
        self.model.eval().to(self.device)

        cfg = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**cfg)

    @torch.no_grad()
    def embed(self, bgr_crop: "np.ndarray") -> np.ndarray:
        """Return a 1D L2-normalised float32 embedding for a BGR crop."""
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)  # (1, C, H, W)
        feat = self.model(tensor)                                   # (1, D)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_centroids(path: Path) -> dict[int, np.ndarray]:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    # Normalise keys to plain Python int
    return {int(k): v.astype(np.float32) for k, v in raw.items()}


def load_cat_map(path: Path) -> dict[int, dict]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def identify(embedding: np.ndarray,
             centroids: dict[int, np.ndarray],
             cat_map: dict[int, dict]) -> tuple[str, float]:
    """Return (label_str, best_distance).  Label is cat name or 'New Visitor'."""
    best_dist  = float("inf")
    best_label = "New Visitor"

    for cluster_id, centroid in centroids.items():
        d = float(cosine_dist(embedding.astype(np.float64),
                              centroid.astype(np.float64)))
        if d < best_dist:
            best_dist  = d
            if d < MATCH_THRESHOLD:
                best_label = cat_map.get(cluster_id, {}).get("name", f"Cat {cluster_id}")

    return best_label, best_dist


def expand_crop(frame: np.ndarray, box: list[float]) -> tuple[np.ndarray, bool]:
    """Return crop with 10 % margin, clipped to frame bounds."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    mx = int((x2 - x1) * CROP_MARGIN)
    my = int((y2 - y1) * CROP_MARGIN)
    x1 = max(0, x1 - mx);  y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx);  y2 = min(h, y2 + my)
    crop = frame[y1:y2, x1:x2]
    return crop, crop.size > 0


def draw_detection(frame: np.ndarray, box: list[float],
                   label: str, conf: float) -> None:
    """Draw bounding box and name label in-place."""
    is_known  = label != "New Visitor"
    color     = BOX_COLOR_KNOWN if is_known else BOX_COLOR_VISITOR
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    text = f"{label} ({conf:.2f})"
    (tw, th), base = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(frame, (x1, y1 - th - base - 4), (x1 + tw, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - base - 2),
                FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)


def open_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    if not RTSP_URL:
        log.error("RTSP_URL is not set. Check your .env file.")
        sys.exit(1)

    # Load knowledge base
    log.info("Loading centroids from %s", CENTROIDS_PATH)
    centroids = load_centroids(CENTROIDS_PATH)
    log.info("Loading cat map from %s", CAT_MAP_PATH)
    cat_map   = load_cat_map(CAT_MAP_PATH)
    log.info("Loaded %d resident cats", len(centroids))

    # Load embedding model
    log.info("Loading embedding model: %s", EMBED_MODEL_NAME)
    embedder = EmbeddingModel()

    # Load YOLO with CoreML backend for ANE acceleration
    log.info("Loading YOLO CoreML model: %s", COREML_MODEL_PATH)
    detector = YOLO(COREML_MODEL_PATH, task="detect")

    reconnect_count = 0
    frame_index     = 0
    fps_times: deque = deque(maxlen=30)   # rolling window for FPS display

    log.info("Connecting to stream: %s", RTSP_URL)

    while True:
        cap = open_stream(RTSP_URL)

        if not cap.isOpened():
            reconnect_count += 1
            log.warning("Stream unavailable (attempt %d/%d). Retrying in %.1fs …",
                        reconnect_count, MAX_RECONNECT_ATTEMPTS, RECONNECT_DELAY)
            if reconnect_count >= MAX_RECONNECT_ATTEMPTS:
                log.error("Max reconnection attempts reached. Exiting.")
                break
            time.sleep(RECONNECT_DELAY)
            continue

        log.info("Stream opened. Press 'q' to quit.")
        reconnect_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Frame read failed — reconnecting …")
                    break

                frame_index += 1
                if frame_index % FRAME_SKIP != 0:
                    continue

                t0 = time.perf_counter()

                results = detector(frame, verbose=False)

                for result in results:
                    for det in result.boxes:
                        if int(det.cls[0]) != CAT_CLASS_ID:
                            continue
                        conf = float(det.conf[0])
                        if conf < CONFIDENCE_THRESHOLD:
                            continue

                        box  = det.xyxy[0].tolist()
                        crop, valid = expand_crop(frame, box)
                        if not valid:
                            continue

                        embedding        = embedder.embed(crop)
                        label, best_dist = identify(embedding, centroids, cat_map)
                        draw_detection(frame, box, label, conf)

                        if DEBUG:
                            log.debug("Det: %s  dist=%.4f", label, best_dist)

                # FPS overlay
                fps_times.append(time.perf_counter() - t0)
                if len(fps_times) == fps_times.maxlen:
                    fps = 1.0 / (sum(fps_times) / len(fps_times))
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                FONT, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow("Cat Re-ID — Phase 4", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Quit requested via 'q'.")
                    return

        except KeyboardInterrupt:
            log.info("Interrupted by user.")
            return
        finally:
            cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run()
    finally:
        cv2.destroyAllWindows()
