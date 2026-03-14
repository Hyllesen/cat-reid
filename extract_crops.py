"""
extract_crops.py
Extracts cat crops from raw videos for YOLO-cls training data.
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

RAW_VIDEOS_DIR = Path("raw_videos")
OUTPUT_DIR = Path("dataset/raw_crops")
MODEL_PATH = "yolo26n.pt"
DEVICE = "mps"
FRAME_INTERVAL = 15
CONFIDENCE_THRESHOLD = 0.7
CAT_CLASS_ID = 15  # COCO class 15 = cat


def extract_crops():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    video_paths = sorted(RAW_VIDEOS_DIR.glob("*.mp4"))

    if not video_paths:
        print(f"No .mp4 files found in '{RAW_VIDEOS_DIR}'")
        return

    total_crops = 0

    for video_path in video_paths:
        video_stem = video_path.stem
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"  [skip] Could not open {video_path.name}")
            continue

        print(f"Processing: {video_path.name}")
        frame_index = 0
        crops_this_video = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % FRAME_INTERVAL == 0:
                results = model.predict(
                    source=frame,
                    device=DEVICE,
                    classes=[CAT_CLASS_ID],
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                )

                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        conf = float(box.conf[0])
                        if conf < CONFIDENCE_THRESHOLD:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Clamp coordinates to frame bounds
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        out_name = f"{video_stem}_frame{frame_index:06d}.jpg"
                        cv2.imwrite(str(OUTPUT_DIR / out_name), crop)
                        crops_this_video += 1

            frame_index += 1

        cap.release()
        print(f"  Crops saved: {crops_this_video}")
        total_crops += crops_this_video

    print()
    print(f"Total videos processed : {len(video_paths)}")
    print(f"Total crops saved      : {total_crops}")


if __name__ == "__main__":
    extract_crops()
