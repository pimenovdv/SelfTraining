import numpy as np

def objective(X):
    # Rayleigh quotient maximization = minimize -tr(X^T A X)
    # where A is a symmetric matrix.
    # The minimum is the eigenvector corresponding to the largest eigenvalue of A
    A = np.diag([3.0, 2.0, 1.0])
    return -np.trace(X.T @ A @ X)

def euclidean_gradient(X):
    A = np.diag([3.0, 2.0, 1.0])
    return -2 * A @ X

def riemannian_gradient(X, egrad):
    # Projection of Euclidean gradient onto the tangent space of the Stiefel manifold
    # P_X(Z) = Z - X @ sym(X^T @ Z)
    # where sym(M) = (M + M^T)/2
    sym = 0.5 * (X.T @ egrad + egrad.T @ X)
    return egrad - X @ sym

def retraction(X, rgrad, lr):
    # Retraction via QR decomposition
    # R_X(V) = qf(X + V), where qf is the Q factor of the QR decomposition
    Y = X - lr * rgrad
    Q, R = np.linalg.qr(Y)
    # Fix the signs of the diagonal elements of R to make the QR decomposition unique
    sign_diag = np.sign(np.diag(R))
    Q = Q * sign_diag
    return Q

def train_riemannian_gradient_descent(epochs=50, lr=0.1):
    n, p = 3, 1
    # Initialize X on the Stiefel manifold St(p, n) (orthogonal columns)
    X = np.random.randn(n, p)
    Q, _ = np.linalg.qr(X)
    X = Q

    print("Starting Riemannian Gradient Descent on the Stiefel manifold...")

    for epoch in range(epochs):
        val = objective(X)
        egrad = euclidean_gradient(X)
        rgrad = riemannian_gradient(X, egrad)

        # Check convergence based on the norm of the Riemannian gradient
        rgrad_norm = np.linalg.norm(rgrad)

        X = retraction(X, rgrad, lr)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d} | Obj: {val:.4f} | RGrad Norm: {rgrad_norm:.6f}")

    return X, val

if __name__ == "__main__":
    np.random.seed(42)
    best_X, best_val = train_riemannian_gradient_descent()

    print("Training complete and successful.")
    print("Final Objective Value:", best_val)
    print("Final X (eigenvector for largest eigenvalue):\\n", best_X)

    # Largest eigenvalue is 3, so objective should be around -3
    assert best_val < -2.9, "Riemannian Gradient Descent did not converge to the correct optimum."
