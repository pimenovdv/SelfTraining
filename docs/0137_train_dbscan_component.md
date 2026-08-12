# Experiment 0137: Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

**Script:** `train_dbscan_component.py`

## Hypothesis
Density-Based Spatial Clustering of Applications with Noise (DBSCAN) can effectively identify clusters of arbitrary, non-convex shapes (like concentric circles or interleaving moons) by grouping together closely packed points, while explicitly handling outliers as noise, overcoming the limitations of centroid-based methods like K-Means or GMM.

## Method
- Implemented the DBSCAN algorithm from scratch mathematically in pure NumPy.
- The algorithm iterates through unvisited points:
  - Finds neighbors within a distance `eps` using $L_2$ norm.
  - If a point has fewer than `min_samples` neighbors, it is marked as noise.
  - If a point has at least `min_samples` neighbors, it becomes a "core point," and a new cluster is formed.
  - The cluster is recursively expanded by adding density-reachable points (neighbors of core points).
- Tested the implementation on a synthetic dataset of two interleaving half-circles ("two moons") with added Gaussian noise, a classic non-convex clustering problem where K-Means typically fails.

## Results
- **Success:** Yes.
- The DBSCAN implementation successfully clustered the non-convex data.
- It correctly identified 2 distinct clusters corresponding to the two moons and accurately labeled sparse outlier points as noise.
- The reliance on local density criteria rather than global centroids allowed it to follow the non-linear structure of the clusters.

## Next Steps
- Explore applying DBSCAN to high-dimensional representations learned by other components (e.g., autoencoders or contrastive models) for unsupervised pattern discovery.
- Investigate hierarchical density-based clustering (HDBSCAN) to alleviate the sensitivity to the `eps` hyperparameter.
