import numpy as np
import os
import argparse

class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.params = {}
        for i in range(1, len(layer_sizes)):
            # Xavier initialization
            self.params[f'W{i}'] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2.0 / (layer_sizes[i-1] + layer_sizes[i]))
            self.params[f'b{i}'] = np.zeros((1, layer_sizes[i]))

    def forward(self, x, params=None):
        if params is None:
            params = self.params

        self.activations = {'a0': x}
        a = x
        for i in range(1, len(self.layer_sizes)):
            W = params[f'W{i}']
            b = params[f'b{i}']
            z = np.dot(a, W) + b
            self.activations[f'z{i}'] = z
            if i < len(self.layer_sizes) - 1:
                a = np.tanh(z) # Using tanh for smooth regression
            else:
                a = z
            self.activations[f'a{i}'] = a
        return a

    def backward(self, d_out, params=None):
        if params is None:
            params = self.params

        grads = {}
        d_a = d_out
        for i in range(len(self.layer_sizes) - 1, 0, -1):
            if i < len(self.layer_sizes) - 1:
                # tanh derivative
                d_z = d_a * (1.0 - np.tanh(self.activations[f'z{i}'])**2)
            else:
                d_z = d_a

            a_prev = self.activations[f'a{i-1}']
            W = params[f'W{i}']

            grads[f'W{i}'] = np.dot(a_prev.T, d_z)
            grads[f'b{i}'] = np.sum(d_z, axis=0, keepdims=True)

            d_a = np.dot(d_z, W.T)

        return grads

def generate_task():
    # Sine wave with random amplitude and phase
    A = np.random.uniform(0.1, 5.0)
    b = np.random.uniform(0.0, np.pi)
    return A, b

def generate_data(A, b, k):
    x = np.random.uniform(-5.0, 5.0, (k, 1))
    y = A * np.sin(x + b)
    return x, y

def clone_params(params):
    return {k: v.copy() for k, v in params.items()}

def main():
    parser = argparse.ArgumentParser(description="Train Reptile Meta-Learning component.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of meta-training iterations.")
    parser.add_argument("--meta_batch_size", type=int, default=10, help="Number of tasks per meta-batch.")
    parser.add_argument("--inner_lr", type=float, default=0.01, help="Inner loop learning rate.")
    parser.add_argument("--meta_lr", type=float, default=0.1, help="Outer loop learning rate (meta step size).")
    parser.add_argument("--k_shots", type=int, default=10, help="Number of examples per task.")
    parser.add_argument("--inner_steps", type=int, default=5, help="Number of inner gradient steps.")
    args = parser.parse_args()

    np.random.seed(42)

    model = MLP([1, 64, 64, 1])

    print(f"Training Reptile with epochs={args.epochs}, meta_batch={args.meta_batch_size}, inner_lr={args.inner_lr}, meta_lr={args.meta_lr}")

    for epoch in range(args.epochs):
        meta_weights_update = {k: np.zeros_like(v) for k, v in model.params.items()}

        # We sample a batch of tasks
        for _ in range(args.meta_batch_size):
            A, b = generate_task()

            # Inner loop (adapt to task)
            fast_weights = clone_params(model.params)
            for _ in range(args.inner_steps):
                x, y = generate_data(A, b, args.k_shots)
                y_pred = model.forward(x, fast_weights)
                loss = np.mean((y_pred - y)**2)
                d_out = 2.0 * (y_pred - y) / args.k_shots
                grads = model.backward(d_out, fast_weights)

                for k in fast_weights:
                    fast_weights[k] -= args.inner_lr * grads[k]

            # Accumulate the difference for Reptile update
            for k in meta_weights_update:
                meta_weights_update[k] += (fast_weights[k] - model.params[k])

        # Meta-update
        for k in model.params:
            model.params[k] += (args.meta_lr / args.meta_batch_size) * meta_weights_update[k]

        if epoch % (args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch} completed.")

    print("Training complete.")

if __name__ == "__main__":
    main()
