import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

class SiameseNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return z1, a1, z2

    def backward(self, x, z1, a1, grad_output):
        grad_a1 = np.dot(grad_output, self.W2.T)
        grad_W2 = np.dot(a1.T, grad_output)
        grad_b2 = np.sum(grad_output, axis=0, keepdims=True)

        grad_z1 = grad_a1 * relu_deriv(z1)
        grad_W1 = np.dot(x.T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

        return grad_W1, grad_b1, grad_W2, grad_b2

    def update(self, grads, lr):
        grad_W1, grad_b1, grad_W2, grad_b2 = grads
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2

def triplet_loss(anchor, positive, negative, margin=1.0):
    dist_pos = np.sum((anchor - positive)**2, axis=1, keepdims=True)
    dist_neg = np.sum((anchor - negative)**2, axis=1, keepdims=True)

    losses = np.maximum(0, dist_pos - dist_neg + margin)
    loss = np.mean(losses)

    grad_anchor = np.zeros_like(anchor)
    grad_positive = np.zeros_like(positive)
    grad_negative = np.zeros_like(negative)

    N = anchor.shape[0]
    for i in range(N):
        if losses[i] > 0:
            grad_anchor[i] = 2 * (anchor[i] - positive[i]) / N - 2 * (anchor[i] - negative[i]) / N
            grad_positive[i] = -2 * (anchor[i] - positive[i]) / N
            grad_negative[i] = 2 * (anchor[i] - negative[i]) / N

    return loss, grad_anchor, grad_positive, grad_negative

def main():
    np.random.seed(42)
    input_dim = 10
    hidden_dim = 32
    output_dim = 16

    net = SiameseNetwork(input_dim, hidden_dim, output_dim)

    # Generate dummy data
    N = 256
    anchors = np.random.randn(N, input_dim)
    positives = anchors + np.random.randn(N, input_dim) * 0.1
    # Generate negatives such that they are far from anchors but close enough to trigger margin loss initially
    negatives = anchors + np.random.randn(N, input_dim) * 0.5

    lr = 0.1
    epochs = 500

    print("Starting training Siamese Network with Triplet Loss...")
    for epoch in range(epochs):
        # Forward pass
        z1_a, a1_a, out_a = net.forward(anchors)
        z1_p, a1_p, out_p = net.forward(positives)
        z1_n, a1_n, out_n = net.forward(negatives)

        # Loss
        loss, grad_a, grad_p, grad_n = triplet_loss(out_a, out_p, out_n, margin=1.0)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

        # Backward pass
        g_W1_a, g_b1_a, g_W2_a, g_b2_a = net.backward(anchors, z1_a, a1_a, grad_a)
        g_W1_p, g_b1_p, g_W2_p, g_b2_p = net.backward(positives, z1_p, a1_p, grad_p)
        g_W1_n, g_b1_n, g_W2_n, g_b2_n = net.backward(negatives, z1_n, a1_n, grad_n)

        grads = (
            g_W1_a + g_W1_p + g_W1_n,
            g_b1_a + g_b1_p + g_b1_n,
            g_W2_a + g_W2_p + g_W2_n,
            g_b2_a + g_b2_p + g_b2_n
        )

        net.update(grads, lr)

    print(f"Final Loss: {loss:.4f}")
    if loss < 0.1:
        print("Success: Siamese Network successfully converged.")
    else:
        print("Failure: Siamese Network failed to converge.")

if __name__ == "__main__":
    main()
