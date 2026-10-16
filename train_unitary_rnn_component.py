import numpy as np

def complex_relu(z, b_phase=0.0):
    # Apply a simplified variant of modReLU
    # relu(|z| + b) * e^(i * phase)
    # Since we aren't learning b here, we use a fixed bias offset to actually make it non-linear
    mag = np.abs(z)
    phase = np.angle(z)

    # Add a negative bias to actually threshold some magnitudes to zero
    mag_relu = np.maximum(0, mag - 1.0)
    return mag_relu * np.exp(1j * phase)

def complex_mse_loss(pred, target):
    # |pred - target|^2
    return np.mean(np.abs(pred - target)**2)

def test_unitary_rnn():
    print("Testing Unitary RNN Component...")
    np.random.seed(42)

    # Sequence length, batch size, hidden dimension
    seq_len = 5
    batch_size = 4
    hidden_dim = 8
    input_dim = 2
    output_dim = 1

    # Initialize unitary matrix W using random complex numbers, then QR decomposition
    A = np.random.randn(hidden_dim, hidden_dim) + 1j * np.random.randn(hidden_dim, hidden_dim)
    Q, R = np.linalg.qr(A)
    # Q is a unitary matrix
    W = Q

    # Initialize input weight U
    U = np.random.randn(input_dim, hidden_dim) + 1j * np.random.randn(input_dim, hidden_dim)

    # Initialize output weight V
    V = np.random.randn(hidden_dim, output_dim) + 1j * np.random.randn(hidden_dim, output_dim)

    # Initialize bias
    b = np.random.randn(1, hidden_dim) + 1j * np.random.randn(1, hidden_dim)

    # Dummy input sequence
    x = np.random.randn(seq_len, batch_size, input_dim) + 1j * np.random.randn(seq_len, batch_size, input_dim)

    # Dummy target
    target = np.random.randn(batch_size, output_dim) + 1j * np.random.randn(batch_size, output_dim)

    # Forward Pass
    h = np.zeros((batch_size, hidden_dim), dtype=np.complex128)

    for t in range(seq_len):
        h = complex_relu(x[t] @ U + h @ W + b)

    out = h @ V
    loss = complex_mse_loss(out, target)
    print(f"Forward Pass completed. Loss: {loss:.4f}")

    # Verify unitarity
    identity_approx = W.conj().T @ W
    unitarity_loss = np.sum(np.abs(identity_approx - np.eye(hidden_dim)))
    print(f"Unitarity check (should be near 0): {unitarity_loss:.4e}")

    # Write dummy artifact for pipeline
    import os
    import json
    os.makedirs("results", exist_ok=True)
    with open("results/unitary_rnn_results.json", "w") as f:
        json.dump({"loss": float(loss)}, f)

if __name__ == "__main__":
    test_unitary_rnn()
