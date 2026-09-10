import numpy as np
import time

def dmd(X, Xprime, r):
    # Step 1: SVD of X
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    # Truncate to rank r
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vh[:r, :].conj().T

    # Step 2: Compute Atilde (the reduced dynamics matrix)
    Atilde = Ur.conj().T @ Xprime @ Vr @ np.linalg.inv(Sr)

    # Step 3: Eigen-decomposition of Atilde
    Lambda, W = np.linalg.eig(Atilde)

    # Step 4: DMD modes
    Phi = Xprime @ Vr @ np.linalg.inv(Sr) @ W

    return Phi, Lambda

# To test it, let's create some synthetic data of a dynamical system
def test_dmd():
    t = np.linspace(0, 10, 100)
    dt = t[1] - t[0]

    # Space
    x = np.linspace(-5, 5, 200)
    X, T = np.meshgrid(x, t)

    # Two spatio-temporal signals
    f1 = np.exp(-0.5 * (X + 2)**2) * np.exp(1j * 2 * T)
    f2 = np.exp(-0.5 * (X - 2)**2) * np.exp(1j * 3 * T)

    # Combine signals
    data = (f1 + f2).T # Shape (200, 100)

    # Create time-shifted matrices
    X_mat = data[:, :-1]
    Xprime_mat = data[:, 1:]

    r = 2 # rank

    # DMD
    U, S, Vh = np.linalg.svd(X_mat, full_matrices=False)
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vh[:r, :].conj().T

    Atilde = Ur.conj().T @ Xprime_mat @ Vr @ np.linalg.inv(Sr)
    Lambda, W = np.linalg.eig(Atilde)
    Phi = Xprime_mat @ Vr @ np.linalg.inv(Sr) @ W

    # Calculate continuous time eigenvalues
    omega = np.log(Lambda) / dt

    print("DMD execution successful.")
    print(f"Discrete time eigenvalues: {Lambda}")
    print(f"Continuous time eigenvalues (imaginary part should be close to 2 and 3): {np.imag(omega)}")

if __name__ == "__main__":
    start_time = time.time()
    test_dmd()
    print(f"Completed in {time.time() - start_time:.4f} seconds.")
