import numpy as np

class SafetyAlignmentComponent:
    def __init__(self, feature_dim, constraint_threshold):
        self.feature_dim = feature_dim
        self.constraint_threshold = constraint_threshold
        # Primary objective weights
        self.w = np.random.randn(feature_dim)
        # Safety constraint weights
        self.w_c = np.random.randn(feature_dim)

    def optimize_with_constraints(self, X, y, X_c, y_c, num_epochs=100, lr=0.01, lambda_lr=0.01):
        """
        Optimizes a primary objective while maintaining a safety constraint using
        Primal-Dual gradient descent (Lagrangian multiplier).
        """
        num_samples = X.shape[0]
        lambda_multiplier = 0.0 # Lagrangian multiplier

        for epoch in range(num_epochs):
            # Primary loss and gradient
            preds = X.dot(self.w)
            loss = np.mean((preds - y) ** 2)
            grad_w = (2 / num_samples) * X.T.dot(preds - y)

            # Constraint loss and gradient
            preds_c = X_c.dot(self.w)
            constraint_val = np.mean(preds_c) - self.constraint_threshold

            if constraint_val > 0:
                grad_c = np.mean(X_c, axis=0)
            else:
                grad_c = np.zeros_like(self.w)

            # Update weights (Primal update)
            self.w -= lr * (grad_w + lambda_multiplier * grad_c)

            # Update multiplier (Dual update)
            lambda_multiplier += lambda_lr * constraint_val
            lambda_multiplier = max(0.0, lambda_multiplier) # Must be non-negative

        return loss, constraint_val

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    # Safety dataset: we want predictions on this dataset to be below a threshold
    X_c = np.random.randn(50, 5) + 1.0
    y_c = np.zeros(50) # not directly used, just minimizing output

    sac = SafetyAlignmentComponent(feature_dim=5, constraint_threshold=0.1)

    initial_preds_c = np.mean(X_c.dot(sac.w))
    print(f"Initial constraint value (should be high): {initial_preds_c - sac.constraint_threshold:.4f}")

    final_loss, final_constraint_val = sac.optimize_with_constraints(X, y, X_c, y_c, num_epochs=500, lr=0.1, lambda_lr=0.1)

    print(f"Final constraint value (should be <= 0): {final_constraint_val:.4f}")
    print(f"Final primary loss: {final_loss:.4f}")

    assert final_constraint_val <= 0.05, "Safety constraint not satisfied."
    print("Safety Alignment simulation successful.")
