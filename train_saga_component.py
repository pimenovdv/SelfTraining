import numpy as np

class SAGAOptimizer:
    def __init__(self, num_samples, num_features, lr=0.01):
        self.num_samples = num_samples
        self.lr = lr
        self.weights = np.zeros(num_features)
        self.gradients_memory = np.zeros((num_samples, num_features))
        self.average_gradient = np.zeros(num_features)

    def step(self, i, x_i, y_i, compute_gradient_fn):
        # Compute current gradient for the sample
        grad_i = compute_gradient_fn(self.weights, x_i, y_i)

        # SAGA update rule
        # w = w - lr * (grad_i - grad_memory_i + average_gradient)
        update = grad_i - self.gradients_memory[i] + self.average_gradient
        self.weights -= self.lr * update

        # Update memory and average gradient
        self.average_gradient += (grad_i - self.gradients_memory[i]) / self.num_samples
        self.gradients_memory[i] = grad_i

def compute_logistic_gradient(w, x, y):
    z = np.dot(x, w)
    prediction = 1 / (1 + np.exp(-z))
    return (prediction - y) * x

def test_saga_optimizer():
    np.random.seed(42)

    # Generate dummy data
    num_samples = 100
    num_features = 5
    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(0, 2, num_samples)

    # Initialize optimizer
    optimizer = SAGAOptimizer(num_samples, num_features, lr=0.05)

    # Train
    epochs = 100
    losses = []

    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(num_samples)
        epoch_loss = 0

        # Calculate full loss at the start of epoch
        for j in range(num_samples):
             z_j = np.dot(X[j], optimizer.weights)
             pred_j = 1 / (1 + np.exp(-z_j))
             pred_j = np.clip(pred_j, 1e-15, 1 - 1e-15)
             epoch_loss += -y[j] * np.log(pred_j) - (1 - y[j]) * np.log(1 - pred_j)
        losses.append(epoch_loss / num_samples)

        for i in indices:
            x_i = X[i]
            y_i = y[i]
            optimizer.step(i, x_i, y_i, compute_logistic_gradient)

    # Compute final loss
    final_loss = 0
    for j in range(num_samples):
        z_j = np.dot(X[j], optimizer.weights)
        pred_j = 1 / (1 + np.exp(-z_j))
        pred_j = np.clip(pred_j, 1e-15, 1 - 1e-15)
        final_loss += -y[j] * np.log(pred_j) - (1 - y[j]) * np.log(1 - pred_j)
    final_loss /= num_samples
    losses.append(final_loss)

    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")

    assert losses[-1] < losses[0], "Loss did not decrease"
    print("SAGA Optimizer mathematical verification passed.")

if __name__ == "__main__":
    test_saga_optimizer()
