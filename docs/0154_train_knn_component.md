# Experiment 0154: K-Nearest Neighbors (KNN)

## Hypothesis
By classifying a given sample based on the majority vote of its $k$-nearest neighbors in the feature space using a distance metric, a non-parametric model can effectively learn non-linear decision boundaries for classification tasks without making explicit assumptions about the underlying data distribution.

## Action
Implemented K-Nearest Neighbors in `train_knn_component.py` mathematically in pure NumPy, using Euclidean distance to compute similarities between training points and test samples, and applying a majority voting scheme for classification.

## Outcome
The implementation successfully classified a synthetic linearly separable dataset, achieving a high accuracy of 100.00% and correctly distinguishing between the two Gaussian clusters. The KNN algorithm effectively generalized the distance-based classification logic.

## Next Steps
Evaluate KNN on multi-class datasets and explore alternative distance metrics such as Manhattan or Minkowski distances, as well as distance-weighted voting strategies to account for the varying influences of closer vs. further neighbors.

**Script:** `train_knn_component.py`
