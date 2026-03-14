"""
run_gatekeeper.py

Production gatekeeper for a Mac Mini M4.
Watches an RTSP stream, detects cats via yolo26n.pt (MPS), classifies their
colour with cat_color_classifier.mlpackage (CoreML / ANE), and fires an HTTP
trigger when a grey stray cat is confidently identified for a sustained period.

All sensitive config is loaded from .env (never hard-coded).

Usage:
    python run_gatekeeper.py

Quit: press 'q' in the Gatekeeper Live window.
"""

import logging
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import coremltools as ct
import requests
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gatekeeper")

# ── Config (loaded from .env) ─────────────────────────────────────────────────

load_dotenv()

RTSP_URL         = os.environ["RTSP_URL"]
TRIGGER_URL      = os.environ["TRIGGER_URL"]
DETECTOR_MODEL   = os.getenv("DETECTOR_MODEL",   "yolo26n.pt")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "cat_color_classifier.mlpackage")
DEVICE           = "mps"
DETECT_CONF      = float(os.getenv("DETECT_CONF",      "0.5"))
CLASSIFY_CONF    = float(os.getenv("CLASSIFY_CONF",    "0.8"))
GREY_MAJORITY    = int(os.getenv("GREY_MAJORITY",       "10"))
DEQUE_LEN        = int(os.getenv("DEQUE_LEN",           "15"))
COOLDOWN_SECONDS = float(os.getenv("COOLDOWN_SECONDS",  "5.0"))
DETECTIONS_DIR   = Path(os.getenv("DETECTIONS_DIR",     "detections"))

CAT_CLASS_ID   = 15        # COCO class index for "cat"
WINDOW_TITLE   = "Gatekeeper Live"
RECONNECT_DELAY = 3        # seconds between stream reconnect attempts

# Overlay colours (BGR)
_C_BOX_DEFAULT = (0,   220,  0)
_C_BOX_GREY    = (180, 180, 180)
_C_TEXT_BG     = (0,   0,    0)
_C_TEXT        = (255, 255, 255)
_C_ALERT       = (0,   0,    255)


# ── CoreML classifier wrapper ─────────────────────────────────────────────────

class CoreMLClassifier:
    """
    Wraps a CoreML classification .mlpackage for use inside the video loop.

    The model exported by Ultralytics expects:
      - input  : 'image'  — PIL.Image RGB 224×224
      - outputs: 'classLabel' (str), 'classLabel_probs' (dict[str, float])

    CoreML runs on the Apple Neural Engine / GPU automatically; no device
    selection is needed at inference time.
    """

    def __init__(self, model_path: str):
        self._model = ct.models.MLModel(model_path)

    def predict(self, bgr_crop) -> tuple[str, float]:
        """
        Args:
            bgr_crop: uint8 BGR numpy array (any size).
        Returns:
            (label, confidence) — e.g. ('grey', 0.97)
        """
        pil_img = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.resize((224, 224), Image.BILINEAR)
        out     = self._model.predict({"image": pil_img})
        label   = out["classLabel"]
        conf    = float(out["classLabel_probs"].get(label, 0.0))
        return label, conf


# ── Horn trigger (thread-safe) ────────────────────────────────────────────────

class HornTrigger:
    """Fires an HTTP GET to TRIGGER_URL in a daemon thread with cooldown."""

    def __init__(self, url: str, cooldown: float):
        self._url       = url
        self._cooldown  = cooldown
        self._lock      = threading.Lock()
        self._last_fired: float = 0.0

    @property
    def in_cooldown(self) -> bool:
        return (time.monotonic() - self._last_fired) < self._cooldown

    def maybe_fire(self) -> bool:
        """
        Attempts to fire the trigger. Returns True if sent, False if
        the cooldown has not yet elapsed.
        """
        with self._lock:
            if self.in_cooldown:
                return False
            self._last_fired = time.monotonic()
        threading.Thread(target=self._send, daemon=True).start()
        return True

    def _send(self):
        try:
            # resp = requests.get(self._url, timeout=3)
            # log.info("STRAY DETECTED: HORN TRIGGERED  (HTTP %s)", resp.status_code)
            log.info("STRAY DETECTED: HORN TRIGGERED  (simulated)")
        except requests.RequestException as exc:
            log.error("Horn trigger request failed: %s", exc)


# ── Per-event raw video recorder ──────────────────────────────────────────────

class DetectionRecorder:
    """
    Writes raw (overlay-free) frames to a timestamped MP4 per detection event.
    Starts recording on the first cat frame; closes when cat leaves the frame.
    """

    def __init__(self, output_dir: Path, fps: float = 15.0):
        self._dir    = output_dir
        self._fps    = fps
        self._writer: cv2.VideoWriter | None = None
        output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active(self) -> bool:
        return self._writer is not None

    def start(self, frame, label: str = "unknown", conf: float = 0.0):
        if self.active:
            return
        h, w  = frame.shape[:2]
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        path  = self._dir / f"{label}_{ts}_{conf:.2f}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, self._fps, (w, h))
        log.info("Recording started  ->  %s", path.name)

    def write(self, frame):
        if self._writer is not None:
            self._writer.write(frame)

    def stop(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            log.info("Recording stopped.")


# ── Overlay helpers ───────────────────────────────────────────────────────────

def _put_label(img, text: str, x1: int, y1: int, alert: bool = False):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 1
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    bg_y1 = max(y1 - th - bl - 6, 0)
    cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 6, y1),
                  _C_ALERT if alert else _C_TEXT_BG, -1)
    cv2.putText(img, text, (x1 + 3, y1 - bl - 2),
                font, scale, _C_TEXT, thick, cv2.LINE_AA)


def draw_cat_box(img, x1: int, y1: int, x2: int, y2: int,
                 label: str, conf: float, is_grey: bool):
    colour = _C_BOX_GREY if is_grey else _C_BOX_DEFAULT
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    _put_label(img, f"{label}  {conf:.0%}", x1, y1, alert=is_grey)


def draw_hud(img, counts: Counter, in_cooldown: bool):
    """Small status overlay in the top-left corner."""
    lines = [f"deque  {dict(counts)}"]
    if in_cooldown:
        lines.append("HORN COOLDOWN")
    for i, line in enumerate(lines):
        colour = _C_ALERT if in_cooldown else _C_TEXT
        cv2.putText(img, line, (10, 26 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


# ── Stream helpers ────────────────────────────────────────────────────────────

def open_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency low
    return cap


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    log.info("Loading detector   : %s", DETECTOR_MODEL)
    detector = YOLO(DETECTOR_MODEL)

    log.info("Loading classifier : %s  (CoreML)", CLASSIFIER_MODEL)
    classifier = CoreMLClassifier(CLASSIFIER_MODEL)

    horn     = HornTrigger(TRIGGER_URL, COOLDOWN_SECONDS)
    recorder = DetectionRecorder(DETECTIONS_DIR)
    history: deque[str] = deque(maxlen=DEQUE_LEN)

    log.info("Connecting to stream ...")
    cap = open_stream(RTSP_URL)

    while True:
        # ── Reconnect on dropped stream ───────────────────────────────────────
        if not cap.isOpened():
            log.warning("Stream lost — reconnecting in %ds ...", RECONNECT_DELAY)
            cap.release()
            time.sleep(RECONNECT_DELAY)
            cap = open_stream(RTSP_URL)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            log.warning("Empty frame — reconnecting ...")
            cap.release()
            time.sleep(RECONNECT_DELAY)
            cap = open_stream(RTSP_URL)
            continue

        raw_frame    = frame.copy()   # clean copy for recording (no overlays)
        display      = frame          # draw overlays directly on this
        cat_detected = False
        last_label   = "unknown"
        last_conf    = 0.0

        # ── Object detection ──────────────────────────────────────────────────
        det_results = detector.predict(
            source=frame,
            device=DEVICE,
            classes=[CAT_CLASS_ID],
            conf=DETECT_CONF,
            verbose=False,
        )

        for result in det_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cat_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                fh, fw = frame.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(fw, x2), min(fh, y2)

                crop = frame[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue

                # ── Classification (CoreML) ───────────────────────────────────
                label, conf = classifier.predict(crop)
                last_label  = label
                last_conf   = conf
                history.append(label)
                is_grey = label == "grey"

                # ── Grey trigger check ────────────────────────────────────────
                grey_count = Counter(history).get("grey", 0)
                if is_grey and conf > CLASSIFY_CONF and grey_count >= GREY_MAJORITY:
                    horn.maybe_fire()

                # ── Bounding box overlay ──────────────────────────────────────
                draw_cat_box(display, x1, y1, x2, y2, label, conf, is_grey)

        # ── HUD overlay ───────────────────────────────────────────────────────
        draw_hud(display, Counter(history), horn.in_cooldown)

        # ── Recording ─────────────────────────────────────────────────────────
        if cat_detected:
            if not recorder.active:
                recorder.start(raw_frame, label=last_label, conf=last_conf)
            recorder.write(raw_frame)
        elif recorder.active:
            recorder.stop()
            history.clear()   # reset deque between distinct events

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow(WINDOW_TITLE, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("Quit requested by user.")
            break

    # ── Shutdown ──────────────────────────────────────────────────────────────
    recorder.stop()
    cap.release()
    cv2.destroyAllWindows()
    log.info("Gatekeeper stopped.")


if __name__ == "__main__":
    run()