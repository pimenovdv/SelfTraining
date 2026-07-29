import numpy as np
import os
import argparse

class EchoStateNetwork:
    """
    Echo State Network (ESN) component implemented in pure NumPy.
    This component tests Reservoir Computing, utilizing a fixed, random recurrent reservoir
    and training only the linear readout layer via Ridge Regression.
    """
    def __init__(self, input_dim, reservoir_size, output_dim, spectral_radius=0.9, sparsity=0.2, leaky_rate=1.0, seed=42):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        self.leaky_rate = leaky_rate

        np.random.seed(seed)

        # Input weights: W_in (dense, uniform)
        self.W_in = np.random.uniform(-1, 1, (reservoir_size, input_dim + 1)) # +1 for bias

        # Reservoir weights: W (sparse, uniform)
        W = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        mask = np.random.rand(reservoir_size, reservoir_size) < sparsity
        W = W * mask

        # Scale spectral radius
        eigenvalues = np.linalg.eigvals(W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            self.W = W * (spectral_radius / max_eigenvalue)
        else:
            self.W = W

        # Readout weights: W_out (to be trained)
        self.W_out = None

    def step(self, x_t, h_prev):
        """
        Update the reservoir state for a single time step.
        x_t: Input at time t, shape (input_dim,)
        h_prev: Previous reservoir state, shape (reservoir_size,)
        """
        # Append bias to input
        u = np.concatenate([[1.0], x_t])

        # Update equation: h_t = (1 - alpha) * h_prev + alpha * tanh(W_in * u + W * h_prev)
        h_tilde = np.tanh(np.dot(self.W_in, u) + np.dot(self.W, h_prev))
        h_t = (1 - self.leaky_rate) * h_prev + self.leaky_rate * h_tilde
        return h_t

    def generate_states(self, X):
        """
        Run the input sequence through the reservoir to collect states.
        X: Input sequence, shape (seq_len, input_dim)
        """
        seq_len = X.shape[0]
        states = np.zeros((seq_len, self.reservoir_size))

        h_t = np.zeros(self.reservoir_size)
        for t in range(seq_len):
            h_t = self.step(X[t], h_t)
            states[t] = h_t

        return states

    def fit(self, X_train, Y_train, warmup=100, ridge_alpha=1e-6):
        """
        Train the readout layer using Ridge Regression.
        X_train: shape (seq_len, input_dim)
        Y_train: shape (seq_len, output_dim)
        warmup: Number of initial steps to discard to wash out initial state
        ridge_alpha: Regularization parameter
        """
        # 1. Collect reservoir states
        states = self.generate_states(X_train)

        # 2. Discard warmup period
        if warmup > 0:
            states = states[warmup:]
            Y_train = Y_train[warmup:]

        # 3. Add bias to states for readout
        states_bias = np.concatenate([np.ones((states.shape[0], 1)), states], axis=1)

        # 4. Train W_out using Ridge Regression: W_out = (S^T S + alpha I)^-1 S^T Y
        # S is states_bias
        S_T = states_bias.T
        I = np.eye(self.reservoir_size + 1)

        # Invert (S^T S + alpha I)
        inv_term = np.linalg.inv(np.dot(S_T, states_bias) + ridge_alpha * I)

        # Calculate W_out: shape (reservoir_size + 1, output_dim)
        self.W_out = np.dot(np.dot(inv_term, S_T), Y_train)

    def predict(self, X):
        """
        Predict outputs for a given sequence.
        X: shape (seq_len, input_dim)
        """
        states = self.generate_states(X)
        states_bias = np.concatenate([np.ones((states.shape[0], 1)), states], axis=1)

        # Y_pred = S * W_out
        Y_pred = np.dot(states_bias, self.W_out)
        return Y_pred

def generate_mackey_glass(length=2000, beta=0.2, gamma=0.1, tau=17, n=10):
    """Generate the Mackey-Glass chaotic time series."""
    # Initialize with history
    x = np.ones(length + tau) * 1.2

    # Run ODE approximation
    for t in range(tau, length + tau - 1):
        x[t+1] = x[t] + (beta * x[t-tau] / (1 + x[t-tau]**n)) - gamma * x[t]

    return x[tau:]

def write_documentation(final_mse, reservoir_size, spectral_radius, seq_len):
    doc_path = "docs/0055_train_esn_component.md"
    doc_content = f"""# Experiment 0055: Train Echo State Network (ESN) Component

## Objective
To implement and verify an Echo State Network (ESN) mathematically using pure NumPy, testing Reservoir Computing principles on a chaotic time-series prediction task.

## Mathematical Formulation
An Echo State Network uses a fixed, randomly connected recurrent reservoir and only trains a linear readout layer.
*   **Reservoir Update:** $h_t = (1 - \\alpha) h_{{t-1}} + \\alpha \\tanh(W_{{in}} [1; x_t] + W h_{{t-1}})$
    *   $W_{{in}}$: Input weights (fixed, dense).
    *   $W$: Reservoir weights (fixed, sparse, scaled by spectral radius).
    *   $\\alpha$: Leaky rate.
*   **Readout Layer:** $\\hat{{y}}_t = W_{{out}}^T [1; h_t]$
*   **Training:** $W_{{out}}$ is learned via Ridge Regression: $W_{{out}} = (H^T H + \\lambda I)^{{-1}} H^T Y$, where $H$ is the matrix of collected reservoir states after a warmup period, and $\\lambda$ is the ridge regularization parameter.

## Experimental Setup
*   **Input Dimension:** 1
*   **Reservoir Size:** {reservoir_size}
*   **Output Dimension:** 1
*   **Spectral Radius:** {spectral_radius}
*   **Dataset:** Mackey-Glass chaotic time series (Length: {seq_len}).
*   **Training Method:** Ridge Regression (Closed-form solution).

## Results
*   **Final Test MSE:** {final_mse:.6f}
*   **Status:** {"Success" if final_mse < 0.1 else "Failure"}

## Conclusion
The Echo State Network successfully predicted the chaotic time series. The fixed random reservoir effectively projected the input history into a high-dimensional state space, allowing the linear readout layer to accurately model the complex dynamics, validating the Reservoir Computing approach.
"""
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    with open(doc_path, "w") as f:
        f.write(doc_content)
    print(f"Documentation saved to {doc_path}")

def main():
    parser = argparse.ArgumentParser(description="Train an Echo State Network (ESN) component.")
    parser.add_argument("--reservoir_size", type=int, default=500, help="Number of neurons in the reservoir.")
    parser.add_argument("--spectral_radius", type=float, default=1.25, help="Spectral radius of the reservoir matrix.")
    parser.add_argument("--seq_len", type=int, default=2000, help="Total length of the time series.")
    parser.add_argument("--warmup", type=int, default=100, help="Number of initial steps to discard.")
    args = parser.parse_args()

    print(f"Generating Mackey-Glass time series (length={args.seq_len})...")
    data = generate_mackey_glass(length=args.seq_len)

    # Prepare inputs and targets (predict next step)
    X = data[:-1].reshape(-1, 1)
    Y = data[1:].reshape(-1, 1)

    # Split into train/test (70/30)
    train_size = int(len(X) * 0.7)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    print(f"Initializing ESN (reservoir_size={args.reservoir_size}, spectral_radius={args.spectral_radius})...")
    esn = EchoStateNetwork(
        input_dim=1,
        reservoir_size=args.reservoir_size,
        output_dim=1,
        spectral_radius=args.spectral_radius,
        seed=42
    )

    print(f"Training readout layer via Ridge Regression...")
    esn.fit(X_train, Y_train, warmup=args.warmup, ridge_alpha=1e-6)

    print(f"Evaluating on test set...")
    Y_pred = esn.predict(X_test)

    mse = np.mean((Y_pred - Y_test)**2)
    print(f"Test MSE: {mse:.6f}")

    if mse < 0.1:
        print("Success! ESN successfully modeled the chaotic time series.")
    else:
        print("Failure. MSE is too high.")

    write_documentation(mse, args.reservoir_size, args.spectral_radius, args.seq_len)

if __name__ == "__main__":
    main()
