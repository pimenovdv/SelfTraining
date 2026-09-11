import numpy as np

def bcm_rule(X, num_neurons=1, epochs=100, eta=0.01, tau=10):
    num_samples, num_features = X.shape
    W = np.random.randn(num_features, num_neurons) * 0.01
    theta = np.zeros((1, num_neurons))

    for epoch in range(epochs):
        for i in range(num_samples):
            x = X[i:i+1]
            y = np.dot(x, W)

            dW = eta * np.dot(x.T, y * (y - theta))
            W += dW

            theta += (y**2 - theta) / tau

    return W, theta

def test_component():
    print("Training BCM Component...")
    np.random.seed(42)
    X = np.random.randn(100, 5)

    W, theta = bcm_rule(X, num_neurons=2, epochs=50, eta=0.001, tau=20)
    print("Final Weights:\n", W)
    print("Final Thresholds:\n", theta)
    print("BCM training completed successfully.")

if __name__ == "__main__":
    test_component()
