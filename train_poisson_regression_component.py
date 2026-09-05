import numpy as np

def poisson_regression_train(X, y, lr=0.01, epochs=1000):
    """
    Trains a Poisson regression model using Gradient Descent on negative log-likelihood.
    """
    np.random.seed(42)
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * 0.01
    bias = 0.0

    for _ in range(epochs):
        linear_pred = np.dot(X, weights) + bias
        # Clip to prevent overflow
        linear_pred = np.clip(linear_pred, -250, 250)
        y_pred = np.exp(linear_pred)

        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        weights -= lr * dw
        bias -= lr * db

    return weights, bias

def main():
    print("Initializing Poisson Regression training...")

    np.random.seed(42)
    X = np.random.randn(100, 2)
    true_weights = np.array([0.5, -0.2])
    true_bias = 0.1

    linear_comp = np.dot(X, true_weights) + true_bias
    lambda_param = np.exp(linear_comp)
    y = np.random.poisson(lambda_param)

    weights, bias = poisson_regression_train(X, y, lr=0.1, epochs=1000)

    print(f"True weights: {true_weights}, True bias: {true_bias}")
    print(f"Learned weights: {weights}, Learned bias: {bias}")

    lambda_pred = np.exp(np.dot(X, weights) + bias)
    mse = np.mean((y - lambda_pred)**2)
    print(f"Mean Squared Error: {mse:.4f}")

    if np.allclose(weights, true_weights, atol=0.2) and np.isclose(bias, true_bias, atol=0.2):
        print("Poisson Regression successfully modeled the count data.")
    else:
        print("Poisson Regression weights did not converge to expected values.")
        exit(1)

if __name__ == "__main__":
    main()
