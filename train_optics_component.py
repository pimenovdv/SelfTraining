import numpy as np

class OPTICS:
    """
    Ordering Points To Identify the Clustering Structure (OPTICS).
    Finds core-distance and reachability-distance for all points.
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        n_samples = X.shape[0]
        self.reachability_ = np.full(n_samples, np.inf)
        self.core_distances_ = np.full(n_samples, np.inf)
        self.ordering_ = []

        processed = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            if processed[i]:
                continue

            # Find neighbors
            neighbors = self._region_query(X, i)
            processed[i] = True
            self.ordering_.append(i)

            core_dist = self._core_distance(X, i, neighbors)
            self.core_distances_[i] = core_dist

            if core_dist <= self.eps:
                seeds = {}
                self._update(X, i, neighbors, processed, seeds)

                while seeds:
                    # Extract min
                    q = min(seeds, key=seeds.get)
                    del seeds[q]

                    neighbors_q = self._region_query(X, q)
                    processed[q] = True
                    self.ordering_.append(q)

                    core_dist_q = self._core_distance(X, q, neighbors_q)
                    self.core_distances_[q] = core_dist_q

                    if core_dist_q <= self.eps:
                        self._update(X, q, neighbors_q, processed, seeds)

        return self

    def _region_query(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0].tolist()

    def _core_distance(self, X, point_idx, neighbors):
        if len(neighbors) < self.min_samples:
            return np.inf

        distances = np.linalg.norm(X[neighbors] - X[point_idx], axis=1)
        # Sort distances and get the min_samples-th smallest
        distances.sort()
        return distances[self.min_samples - 1]

    def _update(self, X, point_idx, neighbors, processed, seeds):
        core_dist = self.core_distances_[point_idx]
        for neighbor_idx in neighbors:
            if not processed[neighbor_idx]:
                dist = np.linalg.norm(X[point_idx] - X[neighbor_idx])
                new_reach_dist = max(core_dist, dist)

                if np.isinf(self.reachability_[neighbor_idx]):
                    self.reachability_[neighbor_idx] = new_reach_dist
                    seeds[neighbor_idx] = new_reach_dist
                elif new_reach_dist < self.reachability_[neighbor_idx]:
                    self.reachability_[neighbor_idx] = new_reach_dist
                    seeds[neighbor_idx] = new_reach_dist

def generate_blobs_varying_density():
    np.random.seed(42)
    # Dense blob
    X1 = np.random.normal(loc=[0, 0], scale=0.1, size=(50, 2))
    # Sparse blob
    X2 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    # Noise
    X3 = np.random.uniform(low=-2, high=4, size=(20, 2))
    return np.vstack([X1, X2, X3])

if __name__ == "__main__":
    print("Generating data with varying density...")
    X = generate_blobs_varying_density()

    print("Training OPTICS model...")
    optics = OPTICS(eps=2.0, min_samples=5)
    optics.fit(X)

    print(f"Number of points ordered: {len(optics.ordering_)}")

    # Check if we successfully captured reachability distances
    reach_dists = optics.reachability_[optics.ordering_]

    # Valleys in reachability plot indicate clusters
    valleys = np.sum(reach_dists < 0.3)
    print(f"Number of points with low reachability distance (< 0.3) (indicative of dense cluster): {valleys}")

    if valleys > 10 and len(optics.ordering_) == X.shape[0]:
        print("Success: OPTICS successfully processed points and calculated reachability distances.")
    else:
        print("Warning: Model might not have performed as expected.")
