# Experiment: K-Means Clustering

**Script:** `train_kmeans_component.py`

## Hypothesis
By iteratively updating centroids to the mean of assigned data points based on Euclidean distance, K-Means can partition an unlabeled dataset into $k$ distinct clusters, minimizing the within-cluster sum of squares (inertia).

## Implementation Details
1. **Initialization:** Centroids are initialized by randomly selecting $k$ points from the input dataset.
2. **Assignment Step:** Each data point is assigned to the nearest centroid based on Euclidean distance.
3. **Update Step:** The new centroid for each cluster is computed as the mean of all points assigned to that cluster.
4. **Convergence:** The process repeats until the centroids no longer change significantly (below a defined tolerance) or the maximum number of iterations is reached.

## Results
- **Outcome:** The implementation successfully clustered synthetic 2D data into 3 clusters, achieving an inertia of 286.1842.
- **Success:** Yes.
- **Fixes Required:** No fixes were required. The pure NumPy implementation successfully converged on the test data.

## Next Steps
Extend clustering techniques to soft assignments (e.g., Gaussian Mixture Models via Expectation-Maximization) or hierarchical methods.
