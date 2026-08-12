# Experiment: Gaussian Mixture Models (GMM)

**Script:** `train_gmm_component.py`

## Hypothesis
By employing Expectation-Maximization (EM), a model can iteratively learn the parameters (means, covariances, and weights) of multiple Gaussian distributions to model complex data distributions and perform soft clustering.

## Implementation Details
1. **Initialization:** Means are initialized randomly from data points, covariances as identity matrices, and weights uniformly.
2. **E-step:** Calculate the probability (responsibility) that each data point belongs to each Gaussian component.
3. **M-step:** Update the mean, covariance, and weight for each component based on the responsibilities.
4. **Convergence:** The process repeats until the log-likelihood of the data no longer increases significantly.

## Results
- **Outcome:** The implementation successfully clustered synthetic 2D data into 2 clusters, accurately recovering the underlying parameters and achieving a log-likelihood of -682.86.
- **Success:** Yes.
- **Fixes Required:** No fixes were required. The pure NumPy implementation successfully converged on the test data.

## Next Steps
Explore more advanced clustering techniques such as Density-Based Spatial Clustering (DBSCAN) or spectral clustering for non-convex shapes.
