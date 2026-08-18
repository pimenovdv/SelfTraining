# Support Vector Regression Component

## Overview
This experiment explores Support Vector Regression (SVR) mathematically and implements it from scratch using pure NumPy. While traditional SVR is often solved using sequential minimal optimization (SMO) or quadratic programming in its dual form, this implementation uses a linear kernel and subgradient descent on the primal objective, making it computationally simpler while retaining the core $\epsilon$-insensitive loss function characteristic of SVR.

## Mathematical Foundation
SVR aims to find a function $f(x) = w^T x + b$ that has at most $\epsilon$ deviation from the actually obtained targets $y_i$ for all the training data, and at the same time is as flat as possible (minimizing $\|w\|^2$).

The objective function we minimize is:
$$ \min_{w,b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \max(0, |y_i - (w^T x_i + b)| - \epsilon) $$

In this subgradient descent implementation, we use a slightly reformulated version equivalent to L2-regularization:
$$ L(w, b) = \lambda \|w\|^2 + \sum_{i=1}^n \max(0, |y_i - f(x_i)| - \epsilon) $$

The gradients are computed as follows:
- If $y_i - f(x_i) > \epsilon$: $dw = 2\lambda w - x_i$, $db = -1$
- If $y_i - f(x_i) < -\epsilon$: $dw = 2\lambda w + x_i$, $db = 1$
- Otherwise (within the $\epsilon$-tube): $dw = 2\lambda w$, $db = 0$

## Implementation Details
- **Architecture**: A linear Support Vector Regression model.
- **Optimization**: Subgradient descent over multiple epochs.
- **Key Parameters**: `learning_rate` for step size, `lambda_param` for regularization strength (inversely related to C), and `epsilon` defining the tolerance tube.

## Results
The implementation was successfully tested on a synthetic 1D dataset ($y = 3x + 2 + \text{noise}$). The model converged and successfully recovered the underlying linear parameters.

- **MSE**: ~0.2328
- **Learned Weight**: ~3.0859 (True: 3.0)
- **Learned Bias**: ~1.9400 (True: 2.0)

## Conclusion
The linear Support Vector Regression implementation with subgradient descent successfully fits linear data with noise while ignoring errors within the $\epsilon$-tube. This primal approach provides a simple, gradient-based way to understand and utilize the principles of support vector machines for regression.
**Script:** `train_svr_component.py`
