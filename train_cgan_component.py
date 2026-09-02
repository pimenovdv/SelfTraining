import numpy as np
import os
import argparse

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def get_real_data(batch_size, class_label):
    # Class 0: mean=4.0, std=1.2
    # Class 1: mean=-4.0, std=1.2
    mu = 4.0 if class_label == 0 else -4.0
    return np.random.normal(mu, 1.2, (batch_size, 1))

def get_noise(batch_size, latent_dim=1):
    return np.random.normal(0, 1, (batch_size, latent_dim))

def train_cgan(epochs, batch_size, lr_d, lr_g, hidden_dim):
    np.random.seed(42)
    # Generator: Takes latent + condition (1+1=2 dims)
    input_dim_g = 2
    W_g1 = np.random.randn(input_dim_g, hidden_dim) * np.sqrt(2.0/input_dim_g)
    b_g1 = np.zeros((1, hidden_dim))
    W_g2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0/hidden_dim)
    b_g2 = np.zeros((1, 1))

    # Discriminator: Takes data + condition (1+1=2 dims)
    input_dim_d = 2
    W_d1 = np.random.randn(input_dim_d, hidden_dim) * np.sqrt(2.0/input_dim_d)
    b_d1 = np.zeros((1, hidden_dim))
    W_d2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0/hidden_dim)
    b_d2 = np.zeros((1, 1))

    for epoch in range(epochs):
        # 1. Train Discriminator
        labels = np.random.randint(0, 2, (batch_size, 1))

        # Real data
        X_real = np.zeros((batch_size, 1))
        for i in range(batch_size):
            X_real[i, 0] = get_real_data(1, labels[i, 0])[0, 0]

        # Concat condition for D real
        X_real_cond = np.hstack([X_real, labels])

        # Forward pass D (real)
        Z_d1_real = np.dot(X_real_cond, W_d1) + b_d1
        A_d1_real = relu(Z_d1_real)
        Z_d2_real = np.dot(A_d1_real, W_d2) + b_d2
        pred_real = sigmoid(Z_d2_real)

        # Fake data
        Z_noise = get_noise(batch_size)
        Z_noise_cond = np.hstack([Z_noise, labels])

        # Forward pass G
        Z_g1 = np.dot(Z_noise_cond, W_g1) + b_g1
        A_g1 = relu(Z_g1)
        X_fake = np.dot(A_g1, W_g2) + b_g2

        # Concat condition for D fake
        X_fake_cond = np.hstack([X_fake, labels])

        # Forward pass D (fake)
        Z_d1_fake = np.dot(X_fake_cond, W_d1) + b_d1
        A_d1_fake = relu(Z_d1_fake)
        Z_d2_fake = np.dot(A_d1_fake, W_d2) + b_d2
        pred_fake = sigmoid(Z_d2_fake)

        # Loss D
        loss_d_real = -np.mean(np.log(pred_real + 1e-8))
        loss_d_fake = -np.mean(np.log(1 - pred_fake + 1e-8))
        loss_d = loss_d_real + loss_d_fake

        # Backward pass D (real)
        dZ_d2_real = (pred_real - 1) / batch_size
        dW_d2_real = np.dot(A_d1_real.T, dZ_d2_real)
        db_d2_real = np.sum(dZ_d2_real, axis=0, keepdims=True)

        dA_d1_real = np.dot(dZ_d2_real, W_d2.T)
        dZ_d1_real = dA_d1_real * relu_deriv(Z_d1_real)
        dW_d1_real = np.dot(X_real_cond.T, dZ_d1_real)
        db_d1_real = np.sum(dZ_d1_real, axis=0, keepdims=True)

        # Backward pass D (fake)
        dZ_d2_fake = (pred_fake - 0) / batch_size
        dW_d2_fake = np.dot(A_d1_fake.T, dZ_d2_fake)
        db_d2_fake = np.sum(dZ_d2_fake, axis=0, keepdims=True)

        dA_d1_fake = np.dot(dZ_d2_fake, W_d2.T)
        dZ_d1_fake = dA_d1_fake * relu_deriv(Z_d1_fake)
        dW_d1_fake = np.dot(X_fake_cond.T, dZ_d1_fake)
        db_d1_fake = np.sum(dZ_d1_fake, axis=0, keepdims=True)

        # Update D
        W_d1 -= lr_d * (dW_d1_real + dW_d1_fake)
        b_d1 -= lr_d * (db_d1_real + db_d1_fake)
        W_d2 -= lr_d * (dW_d2_real + dW_d2_fake)
        b_d2 -= lr_d * (db_d2_real + db_d2_fake)

        # 2. Train Generator
        Z_noise_g = get_noise(batch_size)
        labels_g = np.random.randint(0, 2, (batch_size, 1))
        Z_noise_cond_g = np.hstack([Z_noise_g, labels_g])

        # Forward G
        Z_g1 = np.dot(Z_noise_cond_g, W_g1) + b_g1
        A_g1 = relu(Z_g1)
        X_fake_g = np.dot(A_g1, W_g2) + b_g2

        # Concat condition for D
        X_fake_cond_g = np.hstack([X_fake_g, labels_g])

        # Forward D (fake)
        Z_d1_g = np.dot(X_fake_cond_g, W_d1) + b_d1
        A_d1_g = relu(Z_d1_g)
        Z_d2_g = np.dot(A_d1_g, W_d2) + b_d2
        pred_fake_g = sigmoid(Z_d2_g)

        # Loss G
        loss_g = -np.mean(np.log(pred_fake_g + 1e-8))

        # Backward G (through D)
        dZ_d2_g = (pred_fake_g - 1) / batch_size

        dA_d1_g = np.dot(dZ_d2_g, W_d2.T)
        dZ_d1_g = dA_d1_g * relu_deriv(Z_d1_g)
        dX_fake_cond_g = np.dot(dZ_d1_g, W_d1.T)

        # Extract gradient for just the generated data, ignore condition
        dX_fake_g = dX_fake_cond_g[:, 0:1]

        dZ_g2 = dX_fake_g
        dW_g2 = np.dot(A_g1.T, dZ_g2)
        db_g2 = np.sum(dZ_g2, axis=0, keepdims=True)

        dA_g1 = np.dot(dZ_g2, W_g2.T)
        dZ_g1 = dA_g1 * relu_deriv(Z_g1)
        dW_g1 = np.dot(Z_noise_cond_g.T, dZ_g1)
        db_g1 = np.sum(dZ_g1, axis=0, keepdims=True)

        # Update G
        W_g1 -= lr_g * dW_g1
        b_g1 -= lr_g * db_g1
        W_g2 -= lr_g * dW_g2
        b_g2 -= lr_g * db_g2

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d}: Loss D = {loss_d:.4f}, Loss G = {loss_g:.4f}")

    # Evaluate
    Z_test = get_noise(1000)
    labels_test_0 = np.zeros((1000, 1))
    Z_cond_0 = np.hstack([Z_test, labels_test_0])
    A_g1_0 = relu(np.dot(Z_cond_0, W_g1) + b_g1)
    X_fake_0 = np.dot(A_g1_0, W_g2) + b_g2

    labels_test_1 = np.ones((1000, 1))
    Z_cond_1 = np.hstack([Z_test, labels_test_1])
    A_g1_1 = relu(np.dot(Z_cond_1, W_g1) + b_g1)
    X_fake_1 = np.dot(A_g1_1, W_g2) + b_g2

    return np.mean(X_fake_0), np.std(X_fake_0), np.mean(X_fake_1), np.std(X_fake_1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_d", type=float, default=0.01)
    parser.add_argument("--lr_g", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=16)
    args = parser.parse_args()

    m0, s0, m1, s1 = train_cgan(args.epochs, args.batch_size, args.lr_d, args.lr_g, args.hidden_dim)

    print("\nTraining Complete.")
    print(f"Class 0 (Target: 4.0, 1.2) -> Mean: {m0:.4f}, Std: {s0:.4f}")
    print(f"Class 1 (Target: -4.0, 1.2) -> Mean: {m1:.4f}, Std: {s1:.4f}")

if __name__ == "__main__":
    main()
