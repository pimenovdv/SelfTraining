import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def mgu_forward(x, h_prev, W_f, U_f, b_f, W_h, U_h, b_h):
    f_t = sigmoid(np.dot(x, W_f) + np.dot(h_prev, U_f) + b_f)
    h_tilde = tanh(np.dot(x, W_h) + np.dot(f_t * h_prev, U_h) + b_h)
    h_t = (1 - f_t) * h_prev + f_t * h_tilde
    return h_t, f_t, h_tilde

def test_mgu():
    np.random.seed(42)
    input_size = 4
    hidden_size = 3

    x = np.random.randn(1, input_size)
    h_prev = np.random.randn(1, hidden_size)

    W_f = np.random.randn(input_size, hidden_size)
    U_f = np.random.randn(hidden_size, hidden_size)
    b_f = np.random.randn(hidden_size)

    W_h = np.random.randn(input_size, hidden_size)
    U_h = np.random.randn(hidden_size, hidden_size)
    b_h = np.random.randn(hidden_size)

    h_t, f_t, h_tilde = mgu_forward(x, h_prev, W_f, U_f, b_f, W_h, U_h, b_h)

    print("MGU forward pass successful.")
    print("h_t:", h_t)
    assert h_t.shape == (1, hidden_size)

if __name__ == "__main__":
    test_mgu()
