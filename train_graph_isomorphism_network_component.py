import numpy as np

class GINLayer:
    def __init__(self, in_features, out_features, epsilon=0.0):
        self.epsilon = epsilon
        self.W1 = np.random.randn(in_features, out_features) * 0.1
        self.b1 = np.zeros(out_features)
        self.W2 = np.random.randn(out_features, out_features) * 0.1
        self.b2 = np.zeros(out_features)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X, A):
        # A: Adjacency matrix (N x N)
        # X: Node features (N x in_features)

        # Aggregate neighbor features and add self features with epsilon
        agg_features = A @ X + (1 + self.epsilon) * X

        # Apply 2-layer MLP
        h1 = self.relu(agg_features @ self.W1 + self.b1)
        out = h1 @ self.W2 + self.b2
        return out

if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=float)
    A = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ], dtype=float)

    gin = GINLayer(in_features=2, out_features=4)
    out = gin.forward(X, A)
    print("Graph Isomorphism Network component ran successfully.")
    print(f"Output shape: {out.shape}")
