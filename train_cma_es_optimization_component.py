import numpy as np

def rosenbrock(x):
    """Rosenbrock function to test optimization."""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def train_cma_es():
    """
    Evaluates a Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    component mathematically in pure NumPy on the Rosenbrock function.
    """
    N = 2  # Dimension of the problem
    xmean = np.random.rand(N)  # Initial mean
    sigma = 0.5  # Initial step size

    # Strategy parameter setting: Selection
    lambda_ = 4 + int(3 * np.log(N))  # Population size, offspring number
    mu = lambda_ // 2  # Number of parents/points for recombination
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)) # Recombination weights
    mu = int(mu)
    weights = weights / np.sum(weights)  # Normalize recombination weights
    mueff = np.sum(weights)**2 / np.sum(weights**2) # Variance-effectiveness of sum w_i x_i

    # Strategy parameter setting: Adaptation
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)  # Time constant for cumulation for C
    cs = (mueff + 2) / (N + mueff + 5)  # Time constant for cumulation for sigma control
    c1 = 2 / ((N + 1.3)**2 + mueff)  # Learning rate for rank-one update of C
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)**2 + mueff))  # Learning rate for rank-mu update

    # Initialize dynamic (internal) strategy parameters and constants
    pc = np.zeros(N)  # Evolution paths for C
    ps = np.zeros(N)  # Evolution paths for sigma
    B = np.eye(N)     # B defines the coordinate system
    D = np.ones(N)    # Diagonal D defines the scaling
    C = B @ np.diag(D**2) @ B.T  # Covariance matrix C
    invsqrtC = B @ np.diag(D**-1) @ B.T # C^-1/2

    eigeneval = 0  # Track update of B and D
    chiN = N**0.5 * (1 - 1 / (4 * N) + 1 / (21 * N**2))  # Expectation of ||N(0,I)||

    best_val = float('inf')
    best_x = None

    for gen in range(200):
        # Generate and evaluate lambda offspring
        arz = np.random.randn(lambda_, N)
        arx = np.zeros((lambda_, N))
        fit_vals = np.zeros(lambda_)

        for k in range(lambda_):
            arx[k] = xmean + sigma * (B @ (D * arz[k]))
            fit_vals[k] = rosenbrock(arx[k])

            if fit_vals[k] < best_val:
                best_val = fit_vals[k]
                best_x = arx[k].copy()

        # Sort by fitness and compute weighted mean into xmean
        arindex = np.argsort(fit_vals)
        xold = xmean.copy()
        xmean = np.dot(weights, arx[arindex[0:mu]])

        # Cumulation: Update evolution paths
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (xmean - xold) / sigma
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) / chiN < 1.4 + 2 / (N + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma

        # Adapt covariance matrix C
        artmp = (1 / sigma) * (arx[arindex[0:mu]] - xold)
        C = (1 - c1 - cmu) * C \
            + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
            + cmu * artmp.T @ np.diag(weights) @ artmp

        # Adapt step size sigma
        sigma = sigma * np.exp((cs / 1.0) * (np.linalg.norm(ps) / chiN - 1)) # damping d_sigma = 1.0

        # Decomposition of C into B*diag(D.^2)*B' (diagonalization)
        if gen - eigeneval > lambda_ / (c1 + cmu) / N / 10:
            eigeneval = gen
            C = np.triu(C) + np.triu(C, 1).T  # Enforce symmetry
            D, B = np.linalg.eigh(C)          # Eigen decomposition, B==normalized eigenvectors
            D = np.sqrt(D)                    # D is a vector of standard deviations now
            invsqrtC = B @ np.diag(D**-1) @ B.T

    print(f"Final best fitness on Rosenbrock function: {best_val:.6f}")
    print(f"Final best parameters: {best_x}")
    return best_val

if __name__ == "__main__":
    np.random.seed(42)
    train_cma_es()
