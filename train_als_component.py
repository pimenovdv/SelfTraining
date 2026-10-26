import numpy as np
import time

class AlternatingLeastSquares:
    """
    Alternating Least Squares (ALS) for Collaborative Filtering.
    Factorizes a matrix R into User and Item latent feature matrices.
    """
    def __init__(self, n_factors=10, regularization=0.1, n_epochs=10, random_state=None):
        self.n_factors = n_factors
        self.regularization = regularization
        self.n_epochs = n_epochs
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, R):
        m, n = R.shape
        # Initialize users and items matrices
        self.users = np.random.normal(scale=1./self.n_factors, size=(m, self.n_factors))
        self.items = np.random.normal(scale=1./self.n_factors, size=(n, self.n_factors))

        # Mask of observed values
        mask = R > 0

        for epoch in range(self.n_epochs):
            # Update Users
            for i in range(m):
                idx = np.where(mask[i, :])[0]
                if len(idx) > 0:
                    V_i = self.items[idx, :]
                    R_i = R[i, idx]
                    A = V_i.T.dot(V_i) + self.regularization * np.eye(self.n_factors)
                    b = V_i.T.dot(R_i)
                    self.users[i, :] = np.linalg.solve(A, b)

            # Update Items
            for j in range(n):
                idx = np.where(mask[:, j])[0]
                if len(idx) > 0:
                    U_j = self.users[idx, :]
                    R_j = R[idx, j]
                    A = U_j.T.dot(U_j) + self.regularization * np.eye(self.n_factors)
                    b = U_j.T.dot(R_j)
                    self.items[j, :] = np.linalg.solve(A, b)

            # Compute Train RMSE
            predictions = self.predict()
            error = (R[mask] - predictions[mask])**2
            rmse = np.sqrt(np.mean(error))
            print(f"Epoch {epoch+1:02d}/{self.n_epochs}, Train RMSE: {rmse:.4f}")

    def predict(self):
        return self.users.dot(self.items.T)

if __name__ == "__main__":
    print("Testing Alternating Least Squares (ALS) Component...")

    # Create a synthetic sparse ratings matrix
    np.random.seed(42)
    m_users, n_items, n_factors_true = 50, 60, 5

    # Generate true underlying low-rank matrices
    true_users = np.random.rand(m_users, n_factors_true)
    true_items = np.random.rand(n_items, n_factors_true)
    true_R = true_users.dot(true_items.T)

    # Add a little noise
    true_R += np.random.normal(0, 0.05, true_R.shape)
    # Ensure positive ratings as ALS implicitly handles zeros as missing in this implementation
    true_R = np.clip(true_R, 0.1, 5.0)

    # Simulate missing data (only 20% observed)
    R = true_R.copy()
    mask = np.random.rand(m_users, n_items) > 0.2
    R[mask] = 0 # 0 denotes missing

    print(f"Matrix shape: {R.shape}")
    print(f"Observed ratings: {np.sum(R > 0)} / {m_users * n_items}")

    als = AlternatingLeastSquares(n_factors=5, regularization=0.5, n_epochs=15, random_state=42)
    start_time = time.time()
    als.fit(R)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # Evaluate on true missing values
    predictions = als.predict()
    missing_mask = (R == 0)
    test_error = (true_R[missing_mask] - predictions[missing_mask])**2
    test_rmse = np.sqrt(np.mean(test_error))
    print(f"Test RMSE on unobserved ratings: {test_rmse:.4f}")
    assert test_rmse < 0.5, f"ALS did not generalize well enough on missing ratings. Test RMSE: {test_rmse:.4f}"
    print("Alternating Least Squares (ALS) Component mathematical verification passed.")
