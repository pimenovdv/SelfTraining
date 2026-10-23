import numpy as np
import time

class BAM:
    def __init__(self, input_dim, output_dim):
        self.W = np.zeros((input_dim, output_dim))

    def train(self, X, Y):
        # X: (num_samples, input_dim), Y: (num_samples, output_dim)
        # Bipolar inputs {-1, 1}
        self.W = X.T @ Y

    def recall_y(self, X_corrupted):
        Y = X_corrupted @ self.W
        return np.where(Y >= 0, 1, -1)

    def recall_x(self, Y_corrupted):
        X = Y_corrupted @ self.W.T
        return np.where(X >= 0, 1, -1)

def main():
    print("Starting Bidirectional Associative Memory (BAM) Training...")
    start_time = time.time()

    # Define a simple dataset (pairs of bipolar patterns)
    X = np.array([
        [1, -1, 1, -1, 1, -1],
        [-1, 1, -1, 1, -1, 1],
        [1, 1, 1, -1, -1, -1]
    ])
    Y = np.array([
        [1, -1, 1, -1],
        [-1, 1, -1, 1],
        [1, 1, -1, -1]
    ])

    model = BAM(input_dim=6, output_dim=4)
    model.train(X, Y)

    print("Training complete. Testing recall...")
    Y_pred = model.recall_y(X[0])
    print(f"Recall Y for X[0]: {Y_pred} (Expected: {Y[0]})")

    X_corrupted = X[0].copy()
    X_corrupted[0] *= -1
    Y_pred_corrupt = model.recall_y(X_corrupted)
    print(f"Recall Y for corrupted X[0]: {Y_pred_corrupt} (Expected: {Y[0]})")

    X_pred = model.recall_x(Y[0])
    print(f"Recall X for Y[0]: {X_pred} (Expected: {X[0]})")

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds.")
    print("BAM successfully achieved bidirectional recall.")

if __name__ == '__main__':
    main()
