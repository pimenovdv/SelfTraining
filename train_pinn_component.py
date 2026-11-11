import numpy as np

def main():
    print("Training Physics-Informed Neural Network (PINN) Component...")

    np.random.seed(42)

    # Simple ODE: df/dt = -f(t), f(0) = 1
    # Analytical solution: f(t) = exp(-t)

    # Network architecture
    input_size = 1
    hidden_size = 16
    output_size = 1

    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros(output_size)

    learning_rate = 0.01
    epochs = 5000

    t_physics = np.linspace(0, 2, 50).reshape(-1, 1) # Collocation points
    t_data = np.array([[0.0]])
    y_data = np.array([[1.0]])

    for epoch in range(epochs):
        # --- Forward pass for Data Loss ---
        h_data = np.tanh(np.dot(t_data, W1) + b1)
        y_pred_data = np.dot(h_data, W2) + b2

        loss_data = np.mean((y_pred_data - y_data) ** 2)

        # Backward for Data Loss
        dy_pred_data = 2 * (y_pred_data - y_data) / len(t_data)
        dW2_data = np.dot(h_data.T, dy_pred_data)
        db2_data = np.sum(dy_pred_data, axis=0)
        dh_data = np.dot(dy_pred_data, W2.T)
        dz1_data = dh_data * (1 - h_data ** 2)
        dW1_data = np.dot(t_data.T, dz1_data)
        db1_data = np.sum(dz1_data, axis=0)

        # --- Forward pass for Physics Loss ---
        # f(t)
        h_phys = np.tanh(np.dot(t_physics, W1) + b1)
        y_pred_phys = np.dot(h_phys, W2) + b2

        # f'(t)
        # dh/dt = (1 - h^2) * W1
        # dy/dt = W2 * dh/dt
        dh_dt = (1 - h_phys ** 2) * W1 # shape (N, hidden_size)
        dy_dt = np.dot(dh_dt, W2) # shape (N, output_size)

        # Physics residual: df/dt + f(t) = 0
        residual = dy_dt + y_pred_phys
        loss_physics = np.mean(residual ** 2)

        # Backward for Physics Loss
        dresidual = 2 * residual / len(t_physics)

        # dy_pred_phys part of residual
        dy_pred_phys = dresidual

        # dy_dt part of residual
        ddy_dt = dresidual

        # Gradients from y_pred_phys
        dW2_phys_y = np.dot(h_phys.T, dy_pred_phys)
        db2_phys_y = np.sum(dy_pred_phys, axis=0)
        dh_phys = np.dot(dy_pred_phys, W2.T)
        dz1_phys = dh_phys * (1 - h_phys ** 2)
        dW1_phys_y = np.dot(t_physics.T, dz1_phys)
        db1_phys_y = np.sum(dz1_phys, axis=0)

        # Gradients from dy_dt
        # dy_dt = np.dot(dh_dt, W2) -> shape (N, 1)
        # d(dy_dt) = ddy_dt
        dW2_phys_dy = np.dot(dh_dt.T, ddy_dt)
        # No b2 in dy_dt

        ddh_dt = np.dot(ddy_dt, W2.T) # shape (N, hidden_size)

        # dh_dt = (1 - h_phys ** 2) * W1
        # product rule for ddh_dt:
        # wrt W1 directly:
        dW1_phys_dy_1 = np.sum(ddh_dt * (1 - h_phys ** 2) * t_physics ** 0, axis=0, keepdims=True)
        # Wait, dh_dt = (1 - h_phys^2) * W1 (using broadcasting)
        # dh_dt[i, j] = (1 - h_phys[i, j]^2) * W1[0, j]

        # wrt h_phys:
        dh_phys_from_dh_dt = ddh_dt * W1 * (-2 * h_phys)

        # Now backprop dh_phys_from_dh_dt through h_phys = tanh(...)
        dz1_phys_from_dy_dt = dh_phys_from_dh_dt * (1 - h_phys ** 2)
        dW1_phys_dy_2 = np.dot(t_physics.T, dz1_phys_from_dy_dt)
        db1_phys_dy = np.sum(dz1_phys_from_dy_dt, axis=0)

        dW1_phys_dy = dW1_phys_dy_1 + dW1_phys_dy_2

        dW1_phys = dW1_phys_y + dW1_phys_dy
        db1_phys = db1_phys_y + db1_phys_dy
        dW2_phys = dW2_phys_y + dW2_phys_dy
        db2_phys = db2_phys_y

        # Combine gradients
        dW1 = dW1_data + dW1_phys
        db1 = db1_data + db1_phys
        dW2 = dW2_data + dW2_phys
        db2 = db2_data + db2_phys

        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss Data: {loss_data:.6f} | Loss Physics: {loss_physics:.6f}")

    # Evaluation
    t_test = np.linspace(0, 2, 10).reshape(-1, 1)
    y_true = np.exp(-t_test)
    h_test = np.tanh(np.dot(t_test, W1) + b1)
    y_pred = np.dot(h_test, W2) + b2

    mse = np.mean((y_pred - y_true) ** 2)
    print(f"Final Evaluation MSE against analytical solution: {mse:.6f}")
    if mse < 0.05:
        print("PINN successfully learned the dynamics of the ODE.")
    else:
        print("PINN failed to learn the dynamics accurately.")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
