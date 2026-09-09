import numpy as np

def generate_data(num_samples=1000):
    np.random.seed(42)
    X = np.random.uniform(-5, 5, (num_samples, 1))
    # Heteroscedastic noise
    y = 2 * X.squeeze() + 1 + np.random.normal(0, 0.5 + 0.1 * np.abs(X.squeeze()))
    return X, y

def pinball_loss(y_true, y_pred, quantile):
    err = y_true - y_pred
    return np.mean(np.where(err >= 0, quantile * err, (quantile - 1) * err))

def pinball_gradient(X, y_true, y_pred, quantile):
    err = y_true - y_pred
    grad_y_pred = np.where(err >= 0, -quantile, 1 - quantile)
    N = y_true.shape[0]
    grad_w = X.T.dot(grad_y_pred) / N
    grad_b = np.sum(grad_y_pred) / N
    return grad_w, grad_b

def train_quantile_regression():
    X, y = generate_data()
    # Normalize X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / X_std

    quantile = 0.9 # 90th percentile

    # Initialize weights
    w = np.zeros(X.shape[1])
    b = 0.0

    learning_rate = 0.1
    epochs = 1000

    for epoch in range(epochs):
        y_pred = X_norm.dot(w) + b
        loss = pinball_loss(y, y_pred, quantile)

        grad_w, grad_b = pinball_gradient(X_norm, y, y_pred, quantile)

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    y_pred_final = X_norm.dot(w) + b
    final_loss = pinball_loss(y, y_pred_final, quantile)
    print(f"Final Quantile ({quantile}) Loss: {final_loss:.4f}")

    if final_loss < 2.0:
        print("Success: Quantile Regression model trained successfully.")
    else:
        print("Failure: Model did not converge.")

if __name__ == "__main__":
    train_quantile_regression()
