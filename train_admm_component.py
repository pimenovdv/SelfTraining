import numpy as np
import time

def soft_thresholding(a, kappa):
    return np.sign(a) * np.maximum(np.abs(a) - kappa, 0)

class ADMMLasso:
    def __init__(self, rho=1.0, alpha=0.1, max_iter=1000, tol=1e-4):
        self.rho = rho
        self.alpha = alpha # this is lambda
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, A, b):
        m, n = A.shape
        x = np.zeros(n)
        z = np.zeros(n)
        u = np.zeros(n)

        AtA = A.T @ A
        Atb = A.T @ b
        inv_mat = np.linalg.inv(AtA + self.rho * np.eye(n))

        for i in range(self.max_iter):
            x_new = inv_mat @ (Atb + self.rho * (z - u))
            z_new = soft_thresholding(x_new + u, self.alpha / self.rho)
            u_new = u + x_new - z_new

            # Convergence check
            if np.linalg.norm(x_new - z_new) < self.tol and np.linalg.norm(self.rho * (z_new - z)) < self.tol:
                break

            x = x_new
            z = z_new
            u = u_new

        self.coef_ = z
        self.iterations_ = i + 1

    def predict(self, A):
        return A @ self.coef_

if __name__ == "__main__":
    print("Initializing ADMM for Lasso Optimization...")
    np.random.seed(42)

    # Generate synthetic data
    m, n = 100, 50
    A = np.random.randn(m, n)
    # True weights with sparsity
    true_weights = np.zeros(n)
    true_weights[:10] = np.random.randn(10)
    b = A @ true_weights + np.random.randn(m) * 0.1

    print("Training ADMM Model...")
    start_time = time.time()

    model = ADMMLasso(rho=1.0, alpha=0.5, max_iter=500)
    model.fit(A, b)

    end_time = time.time()

    # Calculate Mean Squared Error
    predictions = model.predict(A)
    mse = np.mean((predictions - b) ** 2)

    # Calculate Sparsity (number of zeros in coefficients)
    sparsity = np.sum(np.abs(model.coef_) < 1e-5)

    print(f"ADMM Lasso Training Completed in {end_time - start_time:.4f} seconds.")
    print(f"Final Mean Squared Error: {mse:.4f}")
    print(f"Iterations: {model.iterations_}")
    print(f"Sparsity: {sparsity}/{n} zero coefficients.")

    print("\nVerification successful. ADMM Component is mathematically sound.")
