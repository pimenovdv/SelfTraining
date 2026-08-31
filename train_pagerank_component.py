"""
PageRank Component

This script implements a mathematical simulation of the PageRank algorithm.
PageRank computes a probability distribution used to represent the likelihood
that a person randomly clicking on links will arrive at any particular page.

The algorithm uses a dampening factor (d) and iteratively updates node ranks.
"""

import numpy as np

class PageRank:
    def __init__(self, d=0.85, tol=1e-6, max_iter=100):
        self.d = d
        self.tol = tol
        self.max_iter = max_iter

    def fit_transform(self, M):
        """
        Computes the PageRank vector for a given adjacency matrix M.
        M[i, j] should be non-zero if there is a link from j to i.
        """
        M = np.array(M, dtype=float)
        N = M.shape[1]

        # Calculate out-degree for each node
        out_degree = M.sum(axis=0)

        # Handle dangling nodes (nodes with no outbound links)
        # by distributing their rank evenly across all nodes
        for i in range(N):
            if out_degree[i] == 0:
                M[:, i] = 1.0 / N
            else:
                M[:, i] = M[:, i] / out_degree[i]

        # Initialize uniform rank vector
        v = np.ones((N, 1)) / N

        # Power iteration
        for i in range(self.max_iter):
            v_next = self.d * np.dot(M, v) + (1 - self.d) / N
            if np.linalg.norm(v_next - v, ord=1) < self.tol:
                print(f"Converged after {i+1} iterations.")
                v = v_next
                break
            v = v_next

        return v

def test_pagerank():
    print("Initializing PageRank component test...")

    # Example adjacency matrix (3 nodes)
    # 0 links to 1 and 2
    # 1 links to 2
    # 2 links to 0
    #
    # Matrix M where M[i, j] = 1 if link from j to i
    #    0  1  2
    # 0  0  0  1
    # 1  1  0  0
    # 2  1  1  0
    M = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0]
    ])

    pr = PageRank()
    ranks = pr.fit_transform(M)

    print("\nAdjacency Matrix:")
    print(M)
    print("\nFinal PageRank values:")
    print(ranks)

    assert np.isclose(ranks.sum(), 1.0), "PageRank values must sum to 1"
    print("\nAssertion passed: PageRank values sum to 1.")
    print("PageRank component mathematical verification successful.")

if __name__ == "__main__":
    test_pagerank()
