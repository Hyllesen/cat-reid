"""
ingest.py — Phase 1: Cat Re-ID RTSP ingest & crop pipeline

Reads a live RTSP stream, runs YOLO cat detection, displays a live preview
window with bounding boxes, and saves high-confidence crops to disk.

Config is loaded from .env (see .env.example for all options).
Press 'q' in the preview window or Ctrl-C to stop.
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RTSP_URL = os.getenv("RTSP_URL", "")
MODEL_PATH = os.getenv("MODEL_PATH", "yolo26n.mlpackage")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./data/raw_crops/"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
MAX_RECONNECT_ATTEMPTS = int(os.getenv("MAX_RECONNECT_ATTEMPTS", "10"))
RECONNECT_DELAY = float(os.getenv("RECONNECT_DELAY", "5"))
DEBUG = os.getenv("DEBUG", "False").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# YOLO class index for 'cat' in the standard COCO dataset
CAT_CLASS_ID = 15

# Bounding box drawing style
BOX_COLOR = (0, 255, 0)   # green
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_crop(frame: "cv2.Mat", box: list[float], conf: float) -> None:
    """Crop the bounding box region (with 10 % margin) and save to OUTPUT_DIR."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)

    margin_x = int((x2 - x1) * 0.10)
    margin_y = int((y2 - y1) * 0.10)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        log.warning("Empty crop skipped (box: %s)", box)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = OUTPUT_DIR / f"cat_{timestamp}_{conf:.2f}.jpg"
    cv2.imwrite(str(filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    log.debug("Saved crop: %s", filename)


def draw_detections(frame: "cv2.Mat", detections: list[tuple]) -> "cv2.Mat":
    """Draw bounding boxes and labels onto frame (in-place copy)."""
    annotated = frame.copy()
    for box, conf in detections:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"cat: {conf:.2f}"
        (lw, lh), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(annotated, (x1, y1 - lh - baseline - 4), (x1 + lw, y1), BOX_COLOR, -1)
        cv2.putText(annotated, label, (x1, y1 - baseline - 2),
                    FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)
    return annotated


def open_stream(url: str) -> cv2.VideoCapture:
    """Open an RTSP stream with low-latency settings."""
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Loading model: %s", MODEL_PATH)
    model = YOLO(MODEL_PATH, task="detect")

    reconnect_count = 0

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
                    log.warning("Frame read failed — attempting reconnect …")
                    break

                # Run inference (verbose=False suppresses per-frame YOLO logs)
                results = model(frame, verbose=False)

                detections = []
                for result in results:
                    for det in result.boxes:
                        cls_id = int(det.cls[0])
                        conf = float(det.conf[0])
                        if cls_id != CAT_CLASS_ID:
                            continue
                        box = det.xyxy[0].tolist()
                        detections.append((box, conf))
                        if conf >= CONFIDENCE_THRESHOLD:
                            save_crop(frame, box, conf)

                annotated = draw_detections(frame, detections)
                cv2.imshow("Cat Re-ID — Phase 1", annotated)

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
