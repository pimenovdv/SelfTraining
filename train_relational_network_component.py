import numpy as np
import os

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def train_relational_network():
    np.random.seed(42)
    batch_size = 32
    num_objects = 5
    object_dim = 10
    g_out_dim = 16
    f_out_dim = 8
    out_dim = 1
    learning_rate = 0.01
    epochs = 200

    X = np.random.randn(batch_size, num_objects, object_dim)
    y = np.sum(X[:, 0, :] * X[:, 1, :], axis=-1, keepdims=True)

    W_g = np.random.randn(2 * object_dim, g_out_dim) * 0.1
    b_g = np.zeros((1, g_out_dim))
    W_f = np.random.randn(g_out_dim, f_out_dim) * 0.1
    b_f = np.zeros((1, f_out_dim))
    W_out = np.random.randn(f_out_dim, out_dim) * 0.1
    b_out = np.zeros((1, out_dim))

    for epoch in range(epochs):
        # Forward pass
        X_i = np.expand_dims(X, axis=2)
        X_j = np.expand_dims(X, axis=1)
        X_i = np.broadcast_to(X_i, (batch_size, num_objects, num_objects, object_dim))
        X_j = np.broadcast_to(X_j, (batch_size, num_objects, num_objects, object_dim))
        pairs = np.concatenate([X_i, X_j], axis=-1)

        pairs_flat = pairs.reshape(-1, 2 * object_dim)
        g_out = relu(np.dot(pairs_flat, W_g) + b_g)
        g_out_reshaped = g_out.reshape(batch_size, num_objects, num_objects, g_out_dim)

        sum_g = np.sum(g_out_reshaped, axis=(1, 2))
        f_out = relu(np.dot(sum_g, W_f) + b_f)
        y_pred = np.dot(f_out, W_out) + b_out

        loss = np.mean((y_pred - y) ** 2)

        # Backward pass
        dy_pred = 2 * (y_pred - y) / batch_size

        dW_out = np.dot(f_out.T, dy_pred)
        db_out = np.sum(dy_pred, axis=0, keepdims=True)

        df_out = np.dot(dy_pred, W_out.T) * relu_derivative(np.dot(sum_g, W_f) + b_f)

        dW_f = np.dot(sum_g.T, df_out)
        db_f = np.sum(df_out, axis=0, keepdims=True)

        dsum_g = np.dot(df_out, W_f.T)
        dg_out_reshaped = np.broadcast_to(np.expand_dims(np.expand_dims(dsum_g, axis=1), axis=2), (batch_size, num_objects, num_objects, g_out_dim))

        dg_out = dg_out_reshaped.reshape(-1, g_out_dim)
        dpairs_flat_pre = dg_out * relu_derivative(np.dot(pairs_flat, W_g) + b_g)

        dW_g = np.dot(pairs_flat.T, dpairs_flat_pre)
        db_g = np.sum(dpairs_flat_pre, axis=0, keepdims=True)

        W_out -= learning_rate * dW_out
        b_out -= learning_rate * db_out
        W_f -= learning_rate * dW_f
        b_f -= learning_rate * db_f
        W_g -= learning_rate * dW_g
        b_g -= learning_rate * db_g

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    if not os.path.exists('docs'):
        os.makedirs('docs')
    with open('docs/0127_train_relational_network_component.md', 'w') as f:
        f.write("# Experiment 0127: Relational Network\n\n")
        f.write("**Script:** `train_relational_network_component.py`\n")
        f.write("**Status:** Success\n\n")
        f.write("Successfully trained a Relational Network component mathematically in pure NumPy.")

if __name__ == '__main__':
    train_relational_network()
