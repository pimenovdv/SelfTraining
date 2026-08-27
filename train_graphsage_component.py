import numpy as np
import argparse

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def create_synthetic_graph(num_nodes=100, num_features=16, num_classes=2):
    np.random.seed(42)
    X = np.random.randn(num_nodes, num_features)
    y = np.random.randint(0, num_classes, num_nodes)

    X[y == 0] += 2.0
    X[y == 1] -= 2.0

    adj_list = {i: [] for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            prob = 0.3 if y[i] == y[j] else 0.05
            if np.random.rand() < prob:
                adj_list[i].append(j)
                adj_list[j].append(i)

    return adj_list, X, y

class GraphSAGE:
    def __init__(self, num_features, hidden_dim, num_classes, max_degree=5):
        self.W1 = np.random.randn(num_features * 2, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.randn(hidden_dim, num_classes) * 0.1
        self.b2 = np.zeros(num_classes)

        self.max_degree = max_degree

    def sample_neighbors(self, adj_list, nodes):
        sampled = []
        for node in nodes:
            neighbors = adj_list[node]
            if len(neighbors) == 0:
                sampled.append([node] * self.max_degree)
            elif len(neighbors) >= self.max_degree:
                sampled.append(np.random.choice(neighbors, self.max_degree, replace=False).tolist())
            else:
                sampled.append(np.random.choice(neighbors, self.max_degree, replace=True).tolist())
        return np.array(sampled)

    def forward(self, X, adj_list, nodes):
        # sample neighbors
        neighbors = self.sample_neighbors(adj_list, nodes)

        # aggregate neighbors (Mean aggregator)
        neighbor_feats = X[neighbors] # (batch_size, max_degree, num_features)
        h_N = np.mean(neighbor_feats, axis=1) # (batch_size, num_features)

        # concatenate self features
        h_v = X[nodes] # (batch_size, num_features)
        h_concat = np.concatenate([h_v, h_N], axis=1) # (batch_size, num_features * 2)

        # layer 1
        z1 = np.dot(h_concat, self.W1) + self.b1
        a1 = relu(z1)

        # layer 2 (output)
        logits = np.dot(a1, self.W2) + self.b2

        # softmax
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        # Cache for backprop
        self.cache = (h_concat, z1, a1, probs, nodes, neighbors)
        return probs

    def backward(self, X, y_true, lr=0.01):
        h_concat, z1, a1, probs, nodes, neighbors = self.cache
        batch_size = len(nodes)

        # Gradient of cross-entropy loss with softmax
        dlogits = probs.copy()
        dlogits[np.arange(batch_size), y_true] -= 1
        dlogits /= batch_size

        dW2 = np.dot(a1.T, dlogits)
        db2 = np.sum(dlogits, axis=0)

        da1 = np.dot(dlogits, self.W2.T)
        dz1 = da1 * relu_deriv(z1)

        dW1 = np.dot(h_concat.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # Update weights
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

def train():
    adj_list, X, y = create_synthetic_graph()
    nodes = np.arange(len(y))
    np.random.seed(42)
    np.random.shuffle(nodes)

    train_nodes = nodes[:80]
    test_nodes = nodes[80:]

    model = GraphSAGE(num_features=16, hidden_dim=8, num_classes=2)

    epochs = 100
    for epoch in range(epochs):
        probs = model.forward(X, adj_list, train_nodes)
        model.backward(X, y[train_nodes], lr=0.1)

        if epoch % 10 == 0:
            preds = np.argmax(probs, axis=1)
            acc = np.mean(preds == y[train_nodes])
            print(f"Epoch {epoch}, Train Accuracy: {acc:.4f}")

    # Test
    test_probs = model.forward(X, adj_list, test_nodes)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = np.mean(test_preds == y[test_nodes])
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GraphSAGE component")
    args = parser.parse_args()
    print("Starting GraphSAGE training...")
    train()
    print("Training complete.")
