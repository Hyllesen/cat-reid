Act as a Data Scientist. We have embeddings in 'embeddings.pkl'.

1. Implement the DBSCAN algorithm to group these embeddings.
2. Logic: If a cluster has more than 5 images, consider it a "resident cat."
3. Automatically move the source images into /data/organized/cluster_0, cluster_1, etc.
4. Generate a 'cat_map.json' that initializes each Cluster ID with a default name like "Unknown Cat [ID]".
5. Provide a way to calculate the "Centroid" (average embedding) for each cluster to use for future real-time matching.
