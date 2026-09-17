import numpy as np
import math

class MADGRAD:
    """
    MADGRAD Optimizer component.
    A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization.
    Combines the benefits of Adam (adaptive) and SGD+Momentum (generalization).
    """
    def __init__(self, params, learning_rate=0.01, momentum=0.9, weight_decay=0.0, eps=1e-6):
        self.params = np.array(params, dtype=np.float32)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eps = eps

        self.v = np.zeros_like(self.params)
        self.s = np.zeros_like(self.params)
        self.x0 = np.copy(self.params)
        self.k = 0

    def step(self, grads):
        self.k += 1
        grads = np.array(grads, dtype=np.float32)

        # Weight decay (L2 penalty)
        if self.weight_decay != 0:
            grads = grads + self.weight_decay * self.params

        # According to original paper: lambda_k = lr * sqrt(k) (or similar schedule)
        lam = self.learning_rate * math.sqrt(self.k)

        # s_k = s_{k-1} + \lambda_k \nabla f(x_k)
        self.s = self.s + lam * grads

        # v_k = v_{k-1} + (\lambda_k \nabla f(x_k))^2
        self.v = self.v + (lam * grads)**2

        # z_k = x_0 - \frac{1}{\sqrt{v_k} + \epsilon} s_k
        z = self.x0 - (1.0 / (np.sqrt(self.v) + self.eps)) * self.s

        # x_{k+1} = (1 - c_{k+1}) x_k + c_{k+1} z_k
        # c_{k+1} is related to momentum (e.g., 1 - momentum)
        self.params = self.momentum * self.params + (1 - self.momentum) * z

        return self.params

def test_madgrad_component():
    print("Testing MADGRAD Optimizer component...")

    # 1. Simple quadratic objective: f(x) = x^2 + 2y^2
    # Minima is at x=0, y=0
    initial_params = [5.0, -3.0]
    optimizer = MADGRAD(initial_params, learning_rate=0.1, momentum=0.9)

    epochs = 300
    params = optimizer.params

    for _ in range(epochs):
        # f'(x) = 2x, f'(y) = 4y
        grads = [2 * params[0], 4 * params[1]]
        params = optimizer.step(grads)

    print(f"Final params: x={params[0]:.4f}, y={params[1]:.4f} (Expected close to 0.0)")
    assert np.abs(params[0]) < 0.1, "MADGRAD failed to converge on x!"
    assert np.abs(params[1]) < 0.1, "MADGRAD failed to converge on y!"
    print("MADGRAD Optimizer component test passed!")

if __name__ == "__main__":
    test_madgrad_component()
