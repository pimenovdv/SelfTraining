import numpy as np

np.random.seed(42)

def multivariate_gaussian(X, mean, cov):
    n = X.shape[1]
    diff = X - mean
    inv_cov = np.linalg.inv(cov)
    exponent = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
    return np.exp(-0.5 * exponent) / np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))

class GMM:
    def __init__(self, k, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.means = X[indices].copy()
        self.covs = [np.eye(n_features) for _ in range(self.k)]
        self.weights = np.ones(self.k) / self.k
        log_likelihood_old = 0

        for iteration in range(self.max_iters):
            likelihoods = np.zeros((n_samples, self.k))
            for i in range(self.k):
                likelihoods[:, i] = self.weights[i] * multivariate_gaussian(X, self.means[i], self.covs[i])

            sum_likelihoods = np.sum(likelihoods, axis=1, keepdims=True)
            self.responsibilities = likelihoods / sum_likelihoods

            log_likelihood_new = np.sum(np.log(sum_likelihoods))
            if iteration > 0 and np.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood_new

            N_k = np.sum(self.responsibilities, axis=0)

            for i in range(self.k):
                resp = self.responsibilities[:, i:i+1]
                self.means[i] = np.sum(resp * X, axis=0) / N_k[i]
                diff = X - self.means[i]
                self.covs[i] = np.dot((resp * diff).T, diff) / N_k[i]
                self.covs[i] += np.eye(n_features) * 1e-6
                self.weights[i] = N_k[i] / n_samples

        self.log_likelihood = log_likelihood_new
        self.iterations = iteration + 1

if __name__ == "__main__":
    X1 = np.random.randn(100, 2) + np.array([2, 2])
    X2 = np.random.randn(100, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    gmm = GMM(k=2)
    gmm.fit(X)
    print("Fitted Means:\n", gmm.means)
    print("Fitted Weights:\n", gmm.weights)
    print("Iterations:", gmm.iterations)
    print("Log Likelihood:", gmm.log_likelihood)
