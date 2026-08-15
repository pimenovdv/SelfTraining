# Linear Discriminant Analysis (LDA) Component

**Script:** `train_lda_component.py`

## Overview
Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction and classification algorithm. While Principal Component Analysis (PCA) finds the directions of maximum variance regardless of class labels, LDA aims to find the axes that maximize the separation between multiple classes. It is widely used in pattern recognition and machine learning for projecting features into a lower dimensional space, which can prevent overfitting and reduce computational costs.

## Mathematical Foundation
LDA seeks to maximize the ratio of the between-class variance to the within-class variance.

### Scatter Matrices
Given a dataset with $C$ classes, the overall mean is:
$$ \mu = \frac{1}{N} \sum_{i=1}^N x_i $$

The within-class scatter matrix $S_W$ measures the scatter of features around their respective class means $\mu_c$:
$$ S_W = \sum_{c=1}^C \sum_{x \in X_c} (x - \mu_c)(x - \mu_c)^T $$

The between-class scatter matrix $S_B$ measures the scatter of the class means around the overall mean:
$$ S_B = \sum_{c=1}^C N_c (\mu_c - \mu)(\mu_c - \mu)^T $$
where $N_c$ is the number of samples in class $c$.

### Optimization Objective
The optimal projection matrix $W$ is found by maximizing the Fisher criterion:
$$ J(W) = \frac{|W^T S_B W|}{|W^T S_W W|} $$

This is equivalent to solving the generalized eigenvalue problem:
$$ S_B w = \lambda S_W w $$
which can be computed as:
$$ S_W^{-1} S_B w = \lambda w $$

The projection matrix $W$ is then formed using the eigenvectors corresponding to the largest eigenvalues.

## Implementation Details
The script `train_lda_component.py` implements LDA from scratch using NumPy:
1. Calculates the overall mean and class-specific means.
2. Constructs the within-class scatter matrix $S_W$ and between-class scatter matrix $S_B$.
3. Computes $S_W^{-1} S_B$ using the pseudo-inverse for stability.
4. Solves the eigenvalue problem and sorts the eigenvectors descending by absolute eigenvalue.
5. Projects the original data onto the top `n_components` eigenvectors.

## Results
The implementation was successfully tested on synthetic 3D data with three distinct classes.
- **Data shape:** `(150, 3)`
- **Projected shape:** `(150, 2)`
- The optimization ran successfully, verifying the correctness of the generalized eigenvalue solver and data projection.
