Act as a Python Computer Vision engineer. Write a script for Phase 1 of a Cat Re-ID system.

1. Use OpenCV to ingest a live RTSP stream using the zsh environment.
2. Load the 'yolo26n.pt' model. Use the latest ultralytics or compatible framework.
3. Filter detections exclusively for the 'cat' class.
4. Implement a 'save_crop' function: if a cat is detected with >0.7 confidence, crop the bounding box with a 10% margin and save it to /data/raw_crops/ using a timestamped filename.
5. Ensure the script handles stream reconnections and frame-skipping to prevent lag.
