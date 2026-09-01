import numpy as np

class BayesianRidgeRegression:
    def __init__(self, alpha_init=1.0, lambda_init=1.0, max_iter=300, tol=1e-3):
        self.alpha = alpha_init # precision of the noise
        self.lambda_ = lambda_init # precision of the weights
        self.max_iter = max_iter
        self.tol = tol
        self.weights_mean = None
        self.weights_cov = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        X = np.c_[np.ones(n_samples), X]
        n_features += 1

        XT_X = X.T.dot(X)
        XT_y = X.T.dot(y)

        eigenvalues = np.linalg.eigvalsh(XT_X)

        for i in range(self.max_iter):
            alpha_old = self.alpha
            lambda_old = self.lambda_

            self.weights_cov = np.linalg.inv(self.lambda_ * np.eye(n_features) + self.alpha * XT_X)
            self.weights_mean = self.alpha * self.weights_cov.dot(XT_y)

            gamma = np.sum((self.alpha * eigenvalues) / (self.lambda_ + self.alpha * eigenvalues))

            self.lambda_ = gamma / np.sum(self.weights_mean ** 2)

            resid = y - X.dot(self.weights_mean)
            self.alpha = (n_samples - gamma) / np.sum(resid ** 2)

            if np.abs(alpha_old - self.alpha) < self.tol and np.abs(lambda_old - self.lambda_) < self.tol:
                break

        self.bias = self.weights_mean[0]
        self.weights = self.weights_mean[1:]

    def predict(self, X, return_std=False):
        n_samples = X.shape[0]
        X = np.c_[np.ones(n_samples), X]
        y_mean = X.dot(self.weights_mean)

        if return_std:
            y_var = 1.0 / self.alpha + np.sum(X.dot(self.weights_cov) * X, axis=1)
            y_std = np.sqrt(y_var)
            return y_mean, y_std

        return y_mean

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 2)
    true_weights = np.array([3.5, -2.0])
    true_bias = 1.0
    y = X.dot(true_weights) + true_bias + np.random.randn(100) * 0.5

    model = BayesianRidgeRegression()
    model.fit(X, y)

    predictions, std_dev = model.predict(X, return_std=True)

    mse = np.mean((predictions - y) ** 2)

    print("True Weights:", true_weights)
    print("Fitted Weights:", model.weights)
    print("True Bias:", true_bias)
    print("Fitted Bias:", model.bias)
    print("MSE:", mse)
    print("Mean Std Dev:", np.mean(std_dev))
