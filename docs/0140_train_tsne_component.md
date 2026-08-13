# Component: t-Distributed Stochastic Neighbor Embedding (t-SNE)

**Script:** `train_tsne_component.py`

## Mathematical Basis

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions.

Let $X$ be the high-dimensional data points. We compute pairwise affinities:
$p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$

The joint probabilities are symmetric:
$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$

Let $Y$ be the low-dimensional map points. We use a Student-t distribution with one degree of freedom to measure similarities in the embedded space, alleviating the crowding problem:
$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}$

The objective is to minimize the Kullback-Leibler divergence:
$C = KL(P || Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$

The gradient is computed as:
$\frac{\partial C}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + ||y_i - y_j||^2)^{-1}$

During optimization, gradient descent with momentum and early exaggeration (multiplying $p_{ij}$ by a constant early in training) is used to find the optimal $Y$.

## Verification

The component is evaluated on a synthetic dataset consisting of two distinct clusters of 50 points each in a 10-dimensional space.
- **Data Generation:** $X_1 \sim \mathcal{N}(5, I)$ and $X_2 \sim \mathcal{N}(-5, I)$.
- **Dimensionality Reduction:** The model successfully projects the 100 points down to 2 dimensions.
- **Result:** The script successfully minimizes the KL divergence, and the two resulting clusters in 2D space remain highly separated with compact intra-cluster variances.
