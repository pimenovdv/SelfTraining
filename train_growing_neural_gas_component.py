import numpy as np

class GNGNode:
    def __init__(self, position):
        self.position = np.array(position)
        self.error = 0.0

class GNGEdge:
    def __init__(self, age=0):
        self.age = age

class GrowingNeuralGas:
    def __init__(self, max_nodes=100, e_b=0.2, e_n=0.006, a_max=50, alpha=0.5, d=0.995, lambda_param=100):
        self.nodes = []
        self.edges = {} # dict of dicts: edges[i][j] = GNGEdge

        self.max_nodes = max_nodes
        self.e_b = e_b # learning rate for winner
        self.e_n = e_n # learning rate for topological neighbors
        self.a_max = a_max # max age for edges
        self.alpha = alpha # error reduction parameter after node insertion
        self.d = d # error decay parameter
        self.lambda_param = lambda_param # interval for node insertion
        self.iter = 0

    def fit_step(self, x):
        if len(self.nodes) < 2:
            self.nodes.append(GNGNode(x))
            if len(self.nodes) == 2:
                self.edges[0] = {1: GNGEdge()}
                self.edges[1] = {0: GNGEdge()}
            return

        # 1. Find two nearest nodes
        dists = [np.linalg.norm(x - n.position) for n in self.nodes]
        s1_idx = np.argmin(dists)
        dists[s1_idx] = np.inf
        s2_idx = np.argmin(dists)

        # 2. Update age of all edges emanating from s1
        if s1_idx in self.edges:
            for j in self.edges[s1_idx]:
                self.edges[s1_idx][j].age += 1
                self.edges[j][s1_idx].age += 1 # symmetric

        # 3. Add squared distance to error of s1
        s1_dist = np.linalg.norm(x - self.nodes[s1_idx].position)
        self.nodes[s1_idx].error += s1_dist**2

        # 4. Move s1 and its topological neighbors towards x
        self.nodes[s1_idx].position += self.e_b * (x - self.nodes[s1_idx].position)
        if s1_idx in self.edges:
            for j in self.edges[s1_idx]:
                self.nodes[j].position += self.e_n * (x - self.nodes[j].position)

        # 5. Create edge between s1 and s2 if it doesn't exist, else set age to 0
        if s1_idx not in self.edges: self.edges[s1_idx] = {}
        if s2_idx not in self.edges: self.edges[s2_idx] = {}
        self.edges[s1_idx][s2_idx] = GNGEdge(age=0)
        self.edges[s2_idx][s1_idx] = GNGEdge(age=0)

        # 6. Remove edges with age > a_max and nodes with no edges
        nodes_to_remove = set()
        for i in list(self.edges.keys()):
            for j in list(self.edges[i].keys()):
                if self.edges[i][j].age > self.a_max:
                    del self.edges[i][j]
                    if i in self.edges[j]: del self.edges[j][i]

        for i in range(len(self.nodes)):
            if i not in self.edges or len(self.edges[i]) == 0:
                nodes_to_remove.add(i)

        if nodes_to_remove:
            # We don't remove nodes to keep indices simple, or we can rebuild.
            # For simplicity in this purely mathematical demonstration, we'll keep them as isolated,
            # or handle deletion by rebuilding the network. A full GNG removes them.
            # To avoid index shifting complexity, we'll just leave them for this short script.
            pass

        # 7. Insert new node every lambda iterations
        self.iter += 1
        if self.iter % self.lambda_param == 0 and len(self.nodes) < self.max_nodes:
            # Find node q with max error
            errors = [n.error for n in self.nodes]
            q_idx = np.argmax(errors)

            # Find neighbor f of q with max error
            if q_idx in self.edges and len(self.edges[q_idx]) > 0:
                neighbors = list(self.edges[q_idx].keys())
                neighbor_errors = [self.nodes[n_idx].error for n_idx in neighbors]
                f_idx = neighbors[np.argmax(neighbor_errors)]

                # Insert r halfway between q and f
                r_pos = 0.5 * (self.nodes[q_idx].position + self.nodes[f_idx].position)
                r_idx = len(self.nodes)
                self.nodes.append(GNGNode(r_pos))

                # Remove edge q-f, add q-r and f-r
                del self.edges[q_idx][f_idx]
                del self.edges[f_idx][q_idx]

                self.edges[r_idx] = {}
                self.edges[q_idx][r_idx] = GNGEdge(age=0)
                self.edges[r_idx][q_idx] = GNGEdge(age=0)

                self.edges[f_idx][r_idx] = GNGEdge(age=0)
                self.edges[r_idx][f_idx] = GNGEdge(age=0)

                # Update errors
                self.nodes[q_idx].error *= self.alpha
                self.nodes[f_idx].error *= self.alpha
                self.nodes[r_idx].error = self.nodes[q_idx].error

        # 8. Decrease all errors
        for n in self.nodes:
            n.error *= self.d

def test_growing_neural_gas():
    print("Testing Growing Neural Gas (GNG) Component...")
    np.random.seed(42)

    # Create simple 2-cluster data
    cluster1 = np.random.randn(200, 2) + np.array([5, 5])
    cluster2 = np.random.randn(200, 2) + np.array([-5, -5])
    data = np.vstack([cluster1, cluster2])
    np.random.shuffle(data)

    gng = GrowingNeuralGas(max_nodes=10)
    for x in data:
        gng.fit_step(x)

    print(f"Number of nodes after training: {len(gng.nodes)}")
    positions = [n.position for n in gng.nodes]
    print(f"Node positions:\n{np.array(positions)}")

    # Basic assertions
    assert len(gng.nodes) > 2, "GNG should grow nodes."

    # Check if nodes generally represent the two clusters
    # Nodes should roughly be around (5,5) or (-5,-5)
    in_cluster1 = sum(1 for p in positions if np.linalg.norm(p - np.array([5, 5])) < 3)
    in_cluster2 = sum(1 for p in positions if np.linalg.norm(p - np.array([-5, -5])) < 3)

    print(f"Nodes near cluster 1: {in_cluster1}")
    print(f"Nodes near cluster 2: {in_cluster2}")

    assert in_cluster1 > 0 and in_cluster2 > 0, "Nodes should represent both data clusters."
    print("Growing Neural Gas Component test passed successfully.")

if __name__ == "__main__":
    test_growing_neural_gas()
