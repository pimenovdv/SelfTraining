import numpy as np

def kl_divergence(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q), axis=-1)

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def js_divergence_derivative(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * np.log(p / m)

def test_component():
    print("Testing Jensen-Shannon Divergence Component...")
    p = np.array([[0.1, 0.9], [0.8, 0.2]])
    q = np.array([[0.2, 0.8], [0.7, 0.3]])

    js_val = js_divergence(p, q)
    print(f"JS Divergence: {js_val}")

    grad = js_divergence_derivative(p, q)
    print(f"Gradient wrt p:\n{grad}")

    epsilon = 1e-5
    p_plus = np.array([[0.1 + epsilon, 0.9], [0.8, 0.2]])
    js_val_plus = js_divergence(p_plus, q)
    num_grad = (js_val_plus[0] - js_val[0]) / epsilon
    print(f"Numerical Gradient wrt p[0,0]: {num_grad}")
    print(f"Analytical Gradient wrt p[0,0]: {grad[0,0]}")

    if np.abs(num_grad - grad[0,0]) < 1e-4:
        print("Gradient check passed!")
        print("Success")
    else:
        print("Gradient check failed!")

if __name__ == "__main__":
    test_component()
