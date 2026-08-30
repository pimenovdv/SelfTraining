import numpy as np
from train_siamese_network_component import SiameseNetwork, triplet_loss

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

    print("Starting testing Siamese Network with Triplet Loss...")
    for epoch in range(epochs):
        # Forward pass
        z1_a, a1_a, out_a = net.forward(anchors)
        z1_p, a1_p, out_p = net.forward(positives)
        z1_n, a1_n, out_n = net.forward(negatives)

        # Loss
        loss, grad_a, grad_p, grad_n = triplet_loss(out_a, out_p, out_n, margin=1.0)

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
