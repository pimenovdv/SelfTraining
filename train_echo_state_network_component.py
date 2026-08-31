import numpy as np
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class EchoStateNetwork:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.2):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size

        # Initialize input weights
        self.W_in = np.random.uniform(-1, 1, (reservoir_size, input_size))

        # Initialize reservoir weights with sparsity
        W_res = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        mask = np.random.rand(reservoir_size, reservoir_size) > sparsity
        W_res[mask] = 0.0

        # Scale reservoir weights to desired spectral radius
        eigenvalues = np.linalg.eigvals(W_res)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        self.W_res = W_res * (spectral_radius / max_eigenvalue)

        # Output weights (to be trained)
        self.W_out = np.zeros((output_size, reservoir_size + input_size))

    def step(self, state, x):
        """Perform one step of reservoir state update."""
        # Using tanh activation for the reservoir
        pre_activation = np.dot(self.W_in, x) + np.dot(self.W_res, state)
        return np.tanh(pre_activation)

    def train(self, X_train, Y_train, discard_steps=10):
        """Train the readout layer using Ridge Regression."""
        seq_length, input_size = X_train.shape
        _, output_size = Y_train.shape

        # Matrix to collect states
        states = np.zeros((seq_length, self.reservoir_size))
        state = np.zeros(self.reservoir_size)

        # Run the reservoir with the input sequence
        for t in range(seq_length):
            state = self.step(state, X_train[t])
            states[t] = state

        # Discard the initial transient states
        X_extended = np.hstack((X_train[discard_steps:], states[discard_steps:]))
        Y_target = Y_train[discard_steps:]

        # Ridge regression to find W_out
        # W_out = (Y * X^T) * (X * X^T + lambda * I)^-1
        ridge_lambda = 1e-4
        identity = np.eye(X_extended.shape[1])

        self.W_out = np.dot(
            np.dot(Y_target.T, X_extended),
            np.linalg.inv(np.dot(X_extended.T, X_extended) + ridge_lambda * identity)
        )

    def predict(self, X_test, initial_state=None):
        """Predict outputs for a sequence of inputs."""
        seq_length, _ = X_test.shape
        states = np.zeros((seq_length, self.reservoir_size))

        if initial_state is None:
            state = np.zeros(self.reservoir_size)
        else:
            state = initial_state

        predictions = np.zeros((seq_length, self.output_size))

        for t in range(seq_length):
            state = self.step(state, X_test[t])
            states[t] = state

            x_ext = np.hstack((X_test[t], state))
            predictions[t] = np.dot(self.W_out, x_ext)

        return predictions, states

def main():
    print("--- Training Echo State Network (ESN) Component ---")
    np.random.seed(42)

    # Generate synthetic sequence data: predicting a sine wave with varying frequency
    t = np.linspace(0, 50, 1000)
    signal = np.sin(t) + 0.5 * np.sin(2.5 * t)

    # Input is previous value, output is next value
    X = signal[:-1].reshape(-1, 1)
    Y = signal[1:].reshape(-1, 1)

    # Split into train and test
    train_size = 800
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    # Initialize and train ESN
    esn = EchoStateNetwork(input_size=1, reservoir_size=50, output_size=1, spectral_radius=0.9, sparsity=0.2)
    esn.train(X_train, Y_train, discard_steps=50)

    # Get last state of training to initialize testing
    _, states_train = esn.predict(X_train)
    last_train_state = states_train[-1]

    # Predict on test set
    predictions, _ = esn.predict(X_test, initial_state=last_train_state)

    # Calculate MSE
    mse = np.mean((predictions - Y_test) ** 2)
    print(f"Test MSE: {mse:.6f}")

    success = mse < 0.05
    if success:
        print("Echo State Network successfully learned to predict the time series!")
    else:
        print("Echo State Network failed to achieve target accuracy.")

if __name__ == "__main__":
    main()
