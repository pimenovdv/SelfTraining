import numpy as np

class Reptile:
    def __init__(self, inner_lr=0.01, meta_lr=0.1, inner_steps=5):
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.theta = np.random.randn(2)

    def get_task(self):
        # Generate a random task: amplitude and phase
        amp = np.random.uniform(0.1, 5.0)
        phase = np.random.uniform(0, np.pi)
        return amp, phase

    def inner_loop(self, amp, phase):
        phi = self.theta.copy()
        for _ in range(self.inner_steps):
            # Sample x
            x = np.random.uniform(-5, 5)
            y_true = amp * np.sin(x + phase)
            y_pred = phi[0] * np.sin(x + phi[1])

            # Gradients
            grad_a = 2 * (y_pred - y_true) * np.sin(x + phi[1])
            grad_p = 2 * (y_pred - y_true) * phi[0] * np.cos(x + phi[1])

            phi[0] -= self.inner_lr * grad_a
            phi[1] -= self.inner_lr * grad_p
        return phi

    def train(self, meta_iters=1000):
        for _ in range(meta_iters):
            amp, phase = self.get_task()
            phi = self.inner_loop(amp, phase)
            self.theta += self.meta_lr * (phi - self.theta)

def test_reptile():
    print("Testing Reptile...")
    reptile = Reptile()
    reptile.train(100)
    print("Reptile completed successfully.")

if __name__ == "__main__":
    test_reptile()
