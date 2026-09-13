import numpy as np

def sigmoid(x):
    # Clip x to prevent overflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class Node2Vec:
    def __init__(self, num_nodes, embedding_dim=16, learning_rate=0.01):
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        # Initialize embeddings for nodes (center) and contexts
        np.random.seed(42)
        self.W_in = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (num_nodes, embedding_dim))
        self.W_out = np.zeros((num_nodes, embedding_dim))

    def train_step(self, center_node, context_nodes, negative_nodes):
        # Center node embedding
        v_c = self.W_in[center_node]

        # Positive context nodes
        for ctx_node in context_nodes:
            v_ctx = self.W_out[ctx_node]
            z = np.dot(v_c, v_ctx)
            p = sigmoid(z)

            # Gradients
            err = p - 1.0
            grad_in = err * v_ctx
            grad_out = err * v_c

            # Update
            self.W_in[center_node] -= self.learning_rate * grad_in
            self.W_out[ctx_node] -= self.learning_rate * grad_out

        # Negative context nodes
        for neg_node in negative_nodes:
            v_neg = self.W_out[neg_node]
            z = np.dot(v_c, v_neg)
            p = sigmoid(z)

            # Gradients
            err = p - 0.0
            grad_in = err * v_neg
            grad_out = err * v_c

            # Update
            self.W_in[center_node] -= self.learning_rate * grad_in
            self.W_out[neg_node] -= self.learning_rate * grad_out

def simulate_random_walks(adj_list, num_walks, walk_length, p=1.0, q=1.0):
    # Simplified node2vec walk (DeepWalk if p=1, q=1)
    walks = []
    nodes = list(adj_list.keys())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                curr = walk[-1]
                neighbors = adj_list[curr]
                if len(neighbors) == 0:
                    break
                next_node = np.random.choice(neighbors)
                walk.append(next_node)
            walks.append(walk)
    return walks

def generate_training_data(walks, window_size, num_nodes, num_neg_samples=5):
    data = []
    for walk in walks:
        for i, center in enumerate(walk):
            context = []
            for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                if i != j:
                    context.append(walk[j])

            if context:
                negatives = []
                while len(negatives) < num_neg_samples * len(context):
                    neg = np.random.randint(0, num_nodes)
                    if neg != center and neg not in context:
                        negatives.append(neg)
                data.append((center, context, negatives))
    return data

def test_component():
    print("Testing Node2Vec Component...")

    # Simple barbell graph: two cliques of 4 nodes connected by a bridge
    # Clique 1: 0, 1, 2, 3
    # Clique 2: 4, 5, 6, 7
    # Bridge: 3-4
    adj_list = {
        0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2, 4],
        4: [3, 5, 6, 7], 5: [4, 6, 7], 6: [4, 5, 7], 7: [4, 5, 6]
    }

    num_nodes = 8

    # Simulate random walks
    walks = simulate_random_walks(adj_list, num_walks=10, walk_length=8)

    # Generate training data (skip-gram with negative sampling)
    training_data = generate_training_data(walks, window_size=2, num_nodes=num_nodes, num_neg_samples=2)

    # Initialize Node2Vec model
    model = Node2Vec(num_nodes=num_nodes, embedding_dim=2, learning_rate=0.05)

    # Train
    epochs = 20
    for epoch in range(epochs):
        np.random.shuffle(training_data)
        for center, context, negatives in training_data:
            model.train_step(center, context, negatives)

    # Evaluate
    embeddings = model.W_in

    # Check if nodes in same clique are closer to each other than to other clique
    # Clique 1: 0, 1, 2
    # Clique 2: 5, 6, 7
    dist_0_1 = np.linalg.norm(embeddings[0] - embeddings[1])
    dist_0_2 = np.linalg.norm(embeddings[0] - embeddings[2])
    dist_0_7 = np.linalg.norm(embeddings[0] - embeddings[7])

    print(f"Distance 0 to 1 (same clique): {dist_0_1:.4f}")
    print(f"Distance 0 to 2 (same clique): {dist_0_2:.4f}")
    print(f"Distance 0 to 7 (diff clique): {dist_0_7:.4f}")

    if dist_0_1 < dist_0_7 and dist_0_2 < dist_0_7:
        print("Success: Node2Vec embeddings capture the graph structure.")
    else:
        print("Failure: Node2Vec embeddings do not capture the graph structure well.")

if __name__ == "__main__":
    test_component()
