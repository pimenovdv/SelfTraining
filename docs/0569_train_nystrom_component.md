# Nyström Method Component

**Script:** train_nystrom_component.py

## Overview
This component implements the Nyström method mathematically in pure NumPy. The Nyström method provides an efficient approximation of the kernel matrix, reducing the computational complexity of kernel methods from $O(N^3)$ to $O(N m^2)$, where $N$ is the total number of samples and $m$ is a smaller number of selected landmark points.

## Mathematical Details
1. **Landmark Selection:** A subset of $m$ landmark points is randomly selected from the dataset $X$.
2. **Kernel Matrix Calculation:** The kernel matrix $C$ between the entire dataset and the landmarks, and the kernel matrix $W$ between the landmarks themselves, are computed using a Radial Basis Function (RBF) kernel:
   $$K(x, y) = \exp(-\gamma \|x - y\|^2)$$
3. **Approximation:** The kernel matrix is approximated as $K \approx C W^{-1} C^T$. To allow the approximation to act as a feature map, we decompose $W^{-1}$ using SVD and create the transformation:
   $$X_{transformed} = C \cdot W^{-1/2}$$
   where $W^{-1/2} = U S^{-1/2} V^T$.

## Results
The component was executed successfully and correctly reduced the dimensionality of the kernel space while maintaining a low relative Frobenius norm error ($0.1310$ on a subset) compared to the exact kernel matrix computation.
