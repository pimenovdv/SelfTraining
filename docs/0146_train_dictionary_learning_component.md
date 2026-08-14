# Experiment Report: Dictionary Learning Component

## Overview
This experiment implements and tests Dictionary Learning mathematically using pure NumPy. Dictionary learning aims to find a sparse representation of the input data in a learned basis (dictionary). It alternates between sparse coding (finding the sparse coefficients given the dictionary) using the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) and updating the dictionary via gradient descent.

**Script:** `train_dictionary_learning_component.py`
**Description:** Evaluates a Dictionary Learning component mathematically in pure NumPy, testing its ability to learn a dictionary that provides sparse representations of synthetic data using an alternating optimization scheme with FISTA.

## Results
- **Success:** Yes.
- The model successfully learned a dictionary that reconstructed the input data with low Mean Squared Error (MSE).
- The sparse coding step using FISTA achieved the desired sparsity (L1 regularization), demonstrating the successful integration of proximal gradient methods.

## Mathematical Formulation
The Dictionary Learning objective is to minimize:
$\min_{D, A} \frac{1}{2N} \|X - A D\|_F^2 + \lambda \|A\|_1$
subject to $\|d_j\|_2 \le 1$ for all columns $d_j$ in $D$.

This is solved by alternating optimization:
1. **Sparse Coding (Update A, fixed D):**
   $a_i = \arg\min_a \frac{1}{2} \|x_i - a D\|_2^2 + \lambda \|a\|_1$
   Solved using FISTA (a proximal gradient method) with the soft-thresholding operator:
   $S_\lambda(v) = \text{sign}(v) \max(|v| - \lambda, 0)$
2. **Dictionary Update (Update D, fixed A):**
   $D = D - \eta \nabla_D (\frac{1}{2N} \|X - A D\|_F^2)$
   $D = D - \eta \frac{1}{N} A^T (A D - X)$
   Followed by projecting each atom $d_j$ to the unit sphere: $d_j = \frac{d_j}{\|d_j\|_2}$.

## Next Steps
- Integrate the learned sparse representations as features for downstream tasks (e.g., classification).
- Compare the learned dictionary with representations obtained from Sparse Autoencoders (SAEs).
