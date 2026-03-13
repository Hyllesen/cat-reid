Act as a Machine Learning Engineer. We are moving to Phase 2: Feature Extraction.

1. Write a Python script using PyTorch and 'timm'.
2. Use a 'mobilenetv3_large_100' backbone to extract embeddings from the cat crops in /data/raw_crops/.
3. Ensure the output is a 1D normalized feature vector (embedding).
4. Save all generated embeddings into a 'embeddings.pkl' file along with their corresponding file paths.
5. Optimize the script to run efficiently on Apple Silicon (MPS device).
