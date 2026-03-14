Act as a Python Developer. I need to extract training data from my cat videos for YOLO26-cls.

Write a script that:

1. Loops through all .mp4 files in a folder named 'raw_videos'.
2. Uses 'yolo26n.pt' on 'mps' to find cats.
3. Logic: Extract every 15th frame. If a cat is found with > 0.7 confidence, save the crop as a .jpg to 'dataset/raw_crops/'.
4. Naming: The filename should include the original video name and the frame index to keep them unique.
5. Provide a summary at the end: "Total videos processed" and "Total crops saved".

Note: Use zsh and do not use exclamation marks in any commands.
