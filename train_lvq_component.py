import numpy as np
import time

def train_lvq():
    print("Starting Learning Vector Quantization (LVQ) component training...")
    # Synthetic data for 2 classes
    np.random.seed(42)

    # Class 0: centered at (2, 2)
    X0 = np.random.randn(50, 2) + np.array([2, 2])
    y0 = np.zeros(50)

    # Class 1: centered at (-2, -2)
    X1 = np.random.randn(50, 2) + np.array([-2, -2])
    y1 = np.ones(50)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Initialize prototypes
    # We will use 2 prototypes per class
    num_prototypes_per_class = 2
    prototypes = []
    prototype_labels = []
    for c in [0, 1]:
        c_indices = np.where(y == c)[0]
        chosen = np.random.choice(c_indices, num_prototypes_per_class, replace=False)
        for idx in chosen:
            prototypes.append(X[idx])
            prototype_labels.append(c)

    prototypes = np.array(prototypes)
    prototype_labels = np.array(prototype_labels)

    epochs = 100
    lr = 0.1

    start_time = time.time()

    for epoch in range(epochs):
        errors = 0
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]

            # Find Best Matching Unit (BMU)
            distances = np.linalg.norm(prototypes - x_i, axis=1)
            bmu_idx = np.argmin(distances)

            # Update BMU
            if prototype_labels[bmu_idx] == y_i:
                prototypes[bmu_idx] += lr * (x_i - prototypes[bmu_idx])
            else:
                prototypes[bmu_idx] -= lr * (x_i - prototypes[bmu_idx])
                errors += 1

        # Learning rate decay
        lr *= 0.99

        if (epoch + 1) % 20 == 0:
            accuracy = 1.0 - errors / len(X)
            print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy:.4f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.4f} seconds.")
    print("Final Prototypes:")
    print(prototypes)
    print("Prototype Labels:")
    print(prototype_labels)

if __name__ == "__main__":
    train_lvq()
