import numpy as np

def generate_data(num_samples=100):
    np.random.seed(42)
    X = np.random.randn(num_samples, 2)
    true_weights = np.array([3.0, -2.0])
    y = X.dot(true_weights) + np.random.randn(num_samples) * 0.1
    return X, y

def signsgd_update(weights, gradients, learning_rate):
    return weights - learning_rate * np.sign(gradients)

def test_signsgd_component():
    X, y = generate_data()
    weights = np.zeros(2)
    learning_rate = 0.1
    epochs = 100

    for epoch in range(epochs):
        predictions = X.dot(weights)
        error = predictions - y
        # MSE Gradient
        gradients = (2.0 / y.size) * X.T.dot(error)

        # SignSGD update
        weights = signsgd_update(weights, gradients, learning_rate)

        if epoch % 10 == 0:
            loss = np.mean(error**2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("Final Weights:", weights)
    print("Expected Weights: ~[3.0, -2.0]")

    # Assertions for convergence
    assert np.allclose(weights, [3.0, -2.0], atol=0.5), "Weights did not converge to expected values."
    print("SignSGD Component test passed!")

if __name__ == "__main__":
    test_signsgd_component()
