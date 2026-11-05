import numpy as np

# Simple 1D regression meta-learning setup
# Task: f(x) = A * x + B, where A, B are sampled for each task

def generate_task():
    A = np.random.uniform(-1, 1)
    B = np.random.uniform(-1, 1)
    return A, B

def sample_data(A, B, num_samples=10):
    x = np.random.uniform(-5, 5, size=(num_samples, 1))
    y = A * x + B + np.random.normal(0, 0.05, size=(num_samples, 1))
    return x, y

def forward(theta, x):
    # theta = [w, b]
    return x * theta[0] + theta[1]

def mse_loss_and_grad(theta, x, y):
    pred = forward(theta, x)
    err = pred - y
    loss = np.mean(err**2)

    # Gradients
    grad_w = np.mean(2 * err * x)
    grad_b = np.mean(2 * err)

    return loss, np.array([grad_w, grad_b])

def train_fomaml(meta_epochs=3000, alpha=0.01, beta=0.01, inner_steps=5):
    # Initialize meta-parameters theta
    theta = np.array([0.0, 0.0])

    print("Starting First-Order MAML (FOMAML) training...")
    for epoch in range(meta_epochs):
        meta_grad = np.zeros_like(theta)

        # Sample batch of tasks
        tasks = [generate_task() for _ in range(8)]

        epoch_loss = 0
        for A, B in tasks:
            # 1. Sample support set for inner update
            x_s, y_s = sample_data(A, B, num_samples=10)

            # 2. Inner loop update (task-specific parameters)
            theta_i = np.copy(theta)
            for _ in range(inner_steps):
                _, grad_s = mse_loss_and_grad(theta_i, x_s, y_s)
                theta_i = theta_i - alpha * grad_s

            # 3. Sample query set for meta update
            x_q, y_q = sample_data(A, B, num_samples=10)
            loss_q, grad_q = mse_loss_and_grad(theta_i, x_q, y_q)

            epoch_loss += loss_q

            # FOMAML: Use the gradient on the query set evaluated at theta_i
            # directly as an approximation for the meta-gradient,
            # ignoring second-order derivatives (Hessian).
            meta_grad += grad_q

        # Meta-update
        theta = theta - beta * (meta_grad / len(tasks))

        if (epoch + 1) % 500 == 0:
            print(f"Meta-Epoch {epoch+1:04d} | Meta Loss: {epoch_loss / len(tasks):.4f}")

    return theta, epoch_loss / len(tasks)

if __name__ == "__main__":
    np.random.seed(42)
    final_theta, final_loss = train_fomaml()

    print("Training complete and successful.")
    print(f"Final Meta Loss: {final_loss:.4f}")
    assert final_loss < 2.5, "FOMAML failed to converge to a reasonable prior."
