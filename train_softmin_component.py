import numpy as np
import json
import os

def softmin(x):
    """
    Computes the Softmin of x.
    Softmin(x) = Softmax(-x) = exp(-x) / sum(exp(-x))
    """
    # Subtracting the max of -x for numerical stability
    neg_x = -x
    exp_x = np.exp(neg_x - np.max(neg_x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def test_softmin():
    # Test 1: Basic 1D array
    x1 = np.array([1.0, 2.0, 3.0])
    s1 = softmin(x1)

    # Test 2: 2D array
    x2 = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    s2 = softmin(x2)

    print("Test 1 Input:", x1)
    print("Test 1 Softmin:", s1)
    print("Sum:", np.sum(s1))

    print("\nTest 2 Input:\n", x2)
    print("Test 2 Softmin:\n", s2)
    print("Sums:", np.sum(s2, axis=-1))

    # Save results
    os.makedirs('results', exist_ok=True)
    results = {
        'test_1_input': x1.tolist(),
        'test_1_output': s1.tolist(),
        'test_2_input': x2.tolist(),
        'test_2_output': s2.tolist()
    }
    with open('results/softmin_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nSoftmin Component test successful!")

if __name__ == '__main__':
    test_softmin()
