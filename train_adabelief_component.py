import numpy as np

class AdaBelief:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-16):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.s = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # AdaBelief modifies the s term to use the variance of the prediction error
            self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * (g - self.m[i])**2 + self.eps

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            s_hat = self.s[i] / (1 - self.beta2**self.t)

            p -= self.lr * m_hat / (np.sqrt(s_hat) + self.eps)

def test_adabelief():
    # Simple quadratic objective: f(x) = x_0^2 + 10 * x_1^2
    # Global minimum is at (0, 0)

    np.random.seed(42)
    params = [np.array([5.0]), np.array([5.0])]

    optimizer = AdaBelief(params, lr=0.1)

    for i in range(200):
        # Gradients
        g0 = 2 * params[0]
        g1 = 20 * params[1]
        grads = [g0, g1]

        optimizer.step(params, grads)

    print(f"Final parameters: x_0 = {params[0][0]:.4f}, x_1 = {params[1][0]:.4f}")
    assert np.abs(params[0][0]) < 1e-1, "Failed to converge on x_0"
    assert np.abs(params[1][0]) < 1e-1, "Failed to converge on x_1"
    print("AdaBelief component testing successful.")

if __name__ == '__main__':
    test_adabelief()
