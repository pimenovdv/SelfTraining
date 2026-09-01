import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class LSH:
    def __init__(self, num_planes, dim):
        self.num_planes = num_planes
        self.planes = np.random.randn(num_planes, dim)

    def hash_vector(self, v):
        # Dot product with random planes
        projections = np.dot(self.planes, v)
        # Binarize
        return tuple(projections > 0)

    def add(self, idx, v):
        h = self.hash_vector(v)
        if h not in self.hash_table:
            self.hash_table[h] = []
        self.hash_table[h].append(idx)
        self.data[idx] = v

    def reset(self):
        self.hash_table = {}
        self.data = {}

    def query(self, v):
        h = self.hash_vector(v)
        return self.hash_table.get(h, [])

def test_lsh():
    np.random.seed(42)
    dim = 50
    num_points = 1000
    num_planes = 10

    # Generate random points
    points = np.random.randn(num_points, dim)

    # Create LSH and insert points
    lsh = LSH(num_planes, dim)
    lsh.reset()
    for i, p in enumerate(points):
        lsh.add(i, p)

    # Query point
    q_idx = 0
    q = points[q_idx]

    candidates = lsh.query(q)
    assert q_idx in candidates, "Query point should always be in its own bucket."

    # Check that candidates have higher average cosine similarity than non-candidates
    candidate_sims = [cosine_similarity(q, points[c]) for c in candidates if c != q_idx]

    all_sims = [cosine_similarity(q, p) for p in points]
    mean_sim_all = np.mean(all_sims)

    if len(candidate_sims) > 0:
        mean_sim_candidates = np.mean(candidate_sims)
        print(f"Mean similarity of all points: {mean_sim_all:.4f}")
        print(f"Mean similarity of candidates: {mean_sim_candidates:.4f}")
        assert mean_sim_candidates > mean_sim_all, "LSH candidates should be more similar on average."

    print("LSH component mathematical verification passed.")

if __name__ == "__main__":
    test_lsh()
