import numpy as np

def hits(adj_matrix, max_iter=100, tol=1e-6):
    """
    Computes HITS (Hyperlink-Induced Topic Search) hub and authority scores.
    """
    n = adj_matrix.shape[0]
    h = np.ones(n)
    a = np.ones(n)

    for _ in range(max_iter):
        h_old = h.copy()
        a_old = a.copy()

        # Update authority: a_i = sum_{j: j->i} h_j
        a = adj_matrix.T.dot(h)
        # Update hub: h_i = sum_{j: i->j} a_j
        h = adj_matrix.dot(a)

        # Normalize
        a = a / np.linalg.norm(a, ord=2)
        h = h / np.linalg.norm(h, ord=2)

        # Check convergence
        if np.linalg.norm(a - a_old) < tol and np.linalg.norm(h - h_old) < tol:
            break

    return h, a

if __name__ == "__main__":
    print("Testing HITS Algorithm...")
    # Directed graph adjacency matrix
    # A[i,j] = 1 if edge from i to j
    adj_matrix = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 0]
    ])

    h, a = hits(adj_matrix)
    print("Hub scores:", h)
    print("Authority scores:", a)

    assert np.all(h >= 0)
    assert np.all(a >= 0)
    assert np.isclose(np.linalg.norm(h), 1.0)
    assert np.isclose(np.linalg.norm(a), 1.0)
    print("HITS algorithm mathematical verification passed successfully.")
