import numpy as np
from train_riemannian_gradient_descent_component import train_riemannian_gradient_descent

def test_riemannian_gradient_descent():
    best_X, best_val = train_riemannian_gradient_descent(epochs=100)
    assert best_val < -2.9
    print("Riemannian Gradient Descent test passed.")

if __name__ == "__main__":
    np.random.seed(42)
    test_riemannian_gradient_descent()
