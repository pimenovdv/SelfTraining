import numpy as np

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    x is expected to be of shape (batch_size, num_classes)
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def softmax_derivative(s, grad_output):
    """
    Compute the derivative of the softmax function.
    s is the output of the softmax function of shape (batch_size, num_classes)
    grad_output is the gradient from the next layer of shape (batch_size, num_classes)
    """
    grad_input = np.zeros_like(s)
    for i in range(s.shape[0]):
        # s[i] is a 1D array of shape (num_classes,)
        s_i = s[i].reshape(-1, 1)
        # Jacobian matrix for softmax of shape (num_classes, num_classes)
        jacobian_m = np.diagflat(s_i) - np.dot(s_i, s_i.T)
        # Gradient input is the dot product of the jacobian matrix and the gradient output
        grad_input[i] = np.dot(jacobian_m, grad_output[i])
    return grad_input

def test_softmax_component():
    np.random.seed(42)

    # 1. Forward Pass Test
    x = np.array([[1.0, 2.0, 3.0],
                  [-1.0, -2.0, -3.0],
                  [1000.0, 1000.0, 1000.0]]) # Test stability

    print("Testing Softmax Component")
    print("-------------------------")
    print(f"Input x:\n{x}")

    out = softmax(x)
    print(f"\nSoftmax output:\n{out}")

    # Check if probabilities sum to 1
    sums = np.sum(out, axis=1)
    print(f"\nSums of probabilities (should be 1s): {sums}")
    assert np.allclose(sums, 1.0), "Probabilities do not sum to 1!"

    # 2. Backward Pass Test
    # Simulate gradient from next layer (e.g., cross entropy)
    grad_output = np.array([[0.1, 0.2, -0.3],
                            [-0.1, 0.1, 0.0],
                            [0.0, 0.0, 0.0]])

    grad_input = softmax_derivative(out, grad_output)
    print(f"\nSimulated grad_output:\n{grad_output}")
    print(f"\nGradient input (backprop):\n{grad_input}")

    print("\nSoftmax component successfully passed all tests.")

if __name__ == '__main__':
    test_softmax_component()
