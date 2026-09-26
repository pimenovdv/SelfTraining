import numpy as np

def meta_learning(tasks, num_iterations=100, inner_lr=0.01, outer_lr=0.01):
    # Model: y = w * x
    w = np.random.randn()

    for iteration in range(num_iterations):
        meta_grad = 0
        for task in tasks:
            # Task format: (x_train, y_train), (x_val, y_val)
            x_train, y_train = task[0]
            x_val, y_val = task[1]

            # Inner loop (adapt to task)
            w_adapted = w - inner_lr * 2 * x_train * (w * x_train - y_train)

            # Outer loop (evaluate on val and accumulate meta gradient)
            meta_grad += 2 * x_val * (w_adapted * x_val - y_val) * (1 - inner_lr * 2 * x_train**2)

        # Update meta-parameters
        w = w - outer_lr * meta_grad / len(tasks)

        if iteration % 20 == 0:
            print(f"Iteration {iteration}, w: {w:.4f}")

    return w

def test_component():
    # True w values for different tasks: 2, 3, 4
    tasks = [
        ((1.0, 2.0), (2.0, 4.0)),
        ((1.0, 3.0), (2.0, 6.0)),
        ((1.0, 4.0), (2.0, 8.0))
    ]
    print("Training Meta-Learning Component (MAML-like)...")
    final_w = meta_learning(tasks)
    print(f"Final meta-learned w: {final_w:.4f}")

if __name__ == '__main__':
    test_component()
