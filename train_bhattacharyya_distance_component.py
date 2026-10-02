import numpy as np
import os
import json

def bhattacharyya_distance(p, q, eps=1e-8):
    """
    Computes the Bhattacharyya distance between two probability distributions p and q.
    """
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    bc = np.sum(np.sqrt(p * q), axis=-1)
    distance = -np.log(bc)
    return distance

def bhattacharyya_distance_gradient(p, q, eps=1e-8):
    """
    Computes the gradient of the Bhattacharyya distance with respect to p.
    """
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    bc = np.sum(np.sqrt(p * q), axis=-1, keepdims=True)
    grad = -0.5 * np.sqrt(q / p) / bc
    return grad

def test_component():
    np.random.seed(42)
    # Generate two random probability distributions
    p = np.random.rand(16, 5)
    p = p / np.sum(p, axis=-1, keepdims=True)

    q = np.random.rand(16, 5)
    q = q / np.sum(q, axis=-1, keepdims=True)

    distance = bhattacharyya_distance(p, q)
    grad = bhattacharyya_distance_gradient(p, q)

    mean_distance = np.mean(distance)

    print("Bhattacharyya Distance Shape:", distance.shape)
    print("Bhattacharyya Gradient Shape:", grad.shape)
    print(f"Mean Distance: {mean_distance:.4f}")

    assert distance.shape == (16,)
    assert grad.shape == (16, 5)
    print("Test passed!")

    os.makedirs("results", exist_ok=True)
    with open("results/bhattacharyya_distance_results.json", "w") as f:
        json.dump({"mean_distance": float(mean_distance)}, f)

if __name__ == "__main__":
    test_component()
