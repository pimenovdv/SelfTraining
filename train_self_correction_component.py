import numpy as np

def train_self_correction():
    print("Initializing Self-Correction Loop mathematically...")
    np.random.seed(42)

    W = np.array([[0.0]])
    W_target = np.array([[3.14]])

    # Critic parameters
    W_c1 = np.random.randn(1, 32) * 0.1
    b_c1 = np.zeros((1, 32))
    W_c2 = np.random.randn(32, 1) * 0.1
    b_c2 = np.zeros((1, 1))

    lr_actor = 0.05
    lr_critic = 0.01

    print("Pre-training critic...")
    for _ in range(10000):
        w_sample = np.random.rand(1, 1) * 10.0 - 2.0
        true_r = -np.mean((w_sample - W_target)**2)

        h = w_sample @ W_c1 + b_c1
        h_relu = np.maximum(0, h)
        pred_r = h_relu @ W_c2 + b_c2

        grad_pred_r = 2 * (pred_r - true_r)

        grad_W_c2 = h_relu.T @ grad_pred_r
        grad_b_c2 = np.sum(grad_pred_r, axis=0, keepdims=True)
        grad_h_relu = grad_pred_r @ W_c2.T
        grad_h = grad_h_relu * (h > 0)
        grad_W_c1 = w_sample.T @ grad_h
        grad_b_c1 = np.sum(grad_h, axis=0, keepdims=True)

        W_c1 -= lr_critic * grad_W_c1
        b_c1 -= lr_critic * grad_b_c1
        W_c2 -= lr_critic * grad_W_c2
        b_c2 -= lr_critic * grad_b_c2

    print("Self-correction loop...")
    for epoch in range(1000):
        w_noisy = W + np.random.randn(1, 1) * 0.5
        true_r = -np.mean((w_noisy - W_target)**2)

        h = w_noisy @ W_c1 + b_c1
        h_relu = np.maximum(0, h)
        pred_r = h_relu @ W_c2 + b_c2

        loss = (pred_r - true_r)**2
        grad_pred_r = 2 * (pred_r - true_r)

        grad_W_c2 = h_relu.T @ grad_pred_r
        grad_b_c2 = np.sum(grad_pred_r, axis=0, keepdims=True)
        grad_h_relu = grad_pred_r @ W_c2.T
        grad_h = grad_h_relu * (h > 0)
        grad_W_c1 = w_noisy.T @ grad_h
        grad_b_c1 = np.sum(grad_h, axis=0, keepdims=True)

        W_c1 -= lr_critic * grad_W_c1
        b_c1 -= lr_critic * grad_b_c1
        W_c2 -= lr_critic * grad_W_c2
        b_c2 -= lr_critic * grad_b_c2

        h_actor = W @ W_c1 + b_c1
        h_actor_relu = np.maximum(0, h_actor)

        grad_pred_r_maximize = np.array([[1.0]])
        grad_h_relu_actor = grad_pred_r_maximize @ W_c2.T
        grad_h_actor = grad_h_relu_actor * (h_actor > 0)
        grad_W_actor = grad_h_actor @ W_c1.T

        W += lr_actor * grad_W_actor

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | W: {W[0,0]:.4f} | Critic Loss: {loss.item():.4f}")

    final_error = np.abs(W[0,0] - W_target[0,0])
    print(f"Final W: {W[0,0]:.4f} (Target: {W_target[0,0]:.4f})")
    assert final_error < 0.5, "Self-correction loop failed to converge to target."
    print("Success: System successfully self-corrected using learned critic.")

if __name__ == "__main__":
    train_self_correction()
