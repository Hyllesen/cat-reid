Act as a Software Architect. Integrate all parts for Phase 4.

1. Create a script that runs the RTSP feed and 'yolo26n.pt' for detection.
2. For every detected cat, extract the embedding in real-time.
3. Use 'scipy.spatial.distance.cosine' to compare the real-time embedding against the Cluster Centroids from Phase 3.
4. If a match is found (distance < 0.25), overlay the name from 'cat_map.json' on the video feed.
5. If no match is found, label as "New Visitor."
6. Optimize the inference loop specifically for a Mac Mini M4 using CoreML for the YOLO26n model to ensure 30+ FPS.
