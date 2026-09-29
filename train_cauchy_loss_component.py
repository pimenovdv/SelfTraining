import numpy as np

def cauchy_loss(y_true, y_pred, c=1.0):
    residual = y_pred - y_true
    return np.mean((c**2 / 2.0) * np.log(1.0 + (residual / c)**2))

def cauchy_loss_derivative(y_true, y_pred, c=1.0):
    residual = y_pred - y_true
    # d/d_ypred [ (c^2 / 2) * ln(1 + (r/c)^2) ]
    # = (c^2 / 2) * [ 1 / (1 + (r/c)^2) ] * [ 2 * r / c^2 ]
    # = r / (1 + (r/c)^2)
    # Mean derivative across batch requires division by N
    N = y_true.size
    return (residual / (1.0 + (residual / c)**2)) / N

class LinearModel:
    def __init__(self, n_features):
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * 0.1
        self.bias = 0.0

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def train():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_weights = np.array([1.5, -2.0, 3.0, 0.5, -1.0])
    y_true = np.dot(X, true_weights) + 2.5

    # Add some outliers to see if Cauchy loss is robust
    y_true[::10] += 20.0 * np.random.randn(10)

    model = LinearModel(n_features=5)
    learning_rate = 0.5
    c = 1.0

    print("Initial Loss:", cauchy_loss(y_true, model.predict(X), c))

    for epoch in range(500):
        y_pred = model.predict(X)
        loss = cauchy_loss(y_true, y_pred, c)

        grad_y_pred = cauchy_loss_derivative(y_true, y_pred, c)

        # d_loss / d_weights = X.T * grad_y_pred
        # Wait, shape of grad_y_pred is (N, ). X is (N, 5).
        # We can dot product.
        dw = np.dot(X.T, grad_y_pred)
        db = np.sum(grad_y_pred)

        model.weights -= learning_rate * dw
        model.bias -= learning_rate * db

    print("Final Loss:", cauchy_loss(y_true, model.predict(X), c))
    print("True weights:", true_weights)
    print("Learned weights:", model.weights)

if __name__ == "__main__":
    train()
