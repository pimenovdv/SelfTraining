import numpy as np
import json
import os

def log_softmax(x):
    # x shape: (batch_size, num_classes)
    c = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    return (x - c) - np.log(sum_exp_x)

def log_softmax_backward(dout, x):
    # dout shape: (batch_size, num_classes)
    # x shape: (batch_size, num_classes)
    c = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    softmax_x = exp_x / sum_exp_x

    # \frac{\partial L}{\partial x_i} = dout_i - softmax(x)_i * \sum_j dout_j
    sum_dout = np.sum(dout, axis=-1, keepdims=True)
    dx = dout - softmax_x * sum_dout
    return dx

def test_log_softmax():
    print("Testing LogSoftmax Component...")
    np.random.seed(42)
    x = np.random.randn(10, 5)
    y_true = np.random.randint(0, 5, size=(10,))
    y_one_hot = np.zeros((10, 5))
    y_one_hot[np.arange(10), y_true] = 1.0

    # Let's train a simple linear model with LogSoftmax + NLLLoss
    w = np.random.randn(5, 5) * 0.1
    b = np.zeros((5,))

    epochs = 100
    learning_rate = 0.1
    losses = []

    for epoch in range(epochs):
        # Forward
        logits = np.dot(x, w) + b
        log_probs = log_softmax(logits)

        # NLL Loss
        loss = -np.sum(log_probs * y_one_hot) / x.shape[0]
        losses.append(loss)

        # Backward
        # derivative of NLL Loss with respect to log_probs
        dlog_probs = -y_one_hot / x.shape[0]

        dlogits = log_softmax_backward(dlog_probs, logits)

        dw = np.dot(x.T, dlogits)
        db = np.sum(dlogits, axis=0)

        # Update
        w -= learning_rate * dw
        b -= learning_rate * db

    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")

    if losses[-1] < losses[0]:
        print("Model successfully learned!")
    else:
        print("Model failed to learn.")

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Save results
    results = {
        'component': 'LogSoftmax',
        'initial_loss': float(losses[0]),
        'final_loss': float(losses[-1]),
        'success': bool(losses[-1] < losses[0])
    }

    with open('results/log_softmax_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    test_log_softmax()
