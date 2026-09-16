import numpy as np
import os

def wasserstein_distance_1d(u_values, v_values, u_weights=None, v_weights=None):
    """
    Computes the 1st Wasserstein distance between two 1D distributions.
    This implementation follows the empirical CDF matching method.
    """
    if u_weights is None:
        u_weights = np.ones_like(u_values) / len(u_values)
    if v_weights is None:
        v_weights = np.ones_like(v_values) / len(v_values)

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort()

    deltas = np.diff(all_values)

    u_cdf = np.zeros_like(all_values)
    v_cdf = np.zeros_like(all_values)

    u_idx = np.searchsorted(u_values[u_sorter], all_values[:-1], side='right')
    v_idx = np.searchsorted(v_values[v_sorter], all_values[:-1], side='right')

    u_cdf_vals = np.cumsum(u_weights[u_sorter])
    v_cdf_vals = np.cumsum(v_weights[v_sorter])

    # Prepend 0 for searchsorted index 0
    u_cdf_vals = np.insert(u_cdf_vals, 0, 0)
    v_cdf_vals = np.insert(v_cdf_vals, 0, 0)

    u_cdf[:-1] = u_cdf_vals[u_idx]
    v_cdf[:-1] = v_cdf_vals[v_idx]

    return np.sum(np.abs(u_cdf[:-1] - v_cdf[:-1]) * deltas)

def test_component():
    print("Testing Wasserstein Distance (Earth Mover's Distance) component...")

    # Two identical distributions
    u = np.array([1, 2, 3])
    v = np.array([1, 2, 3])
    wd_identical = wasserstein_distance_1d(u, v)
    print(f"Distributions U={u}, V={v}")
    print(f"Wasserstein distance (identical): {wd_identical:.6f}")
    assert np.isclose(wd_identical, 0.0), "Distance between identical distributions must be 0."

    # Two distributions with a shift
    u = np.array([1, 2, 3])
    v = np.array([2, 3, 4])
    wd_shifted = wasserstein_distance_1d(u, v)
    print(f"\nDistributions U={u}, V={v}")
    print(f"Wasserstein distance (shifted by 1): {wd_shifted:.6f}")
    assert np.isclose(wd_shifted, 1.0), "Distance for shift of 1 should be 1."

    # Two distinct distributions
    u = np.array([1, 5])
    v = np.array([2, 6])
    wd_distinct = wasserstein_distance_1d(u, v)
    print(f"\nDistributions U={u}, V={v}")
    print(f"Wasserstein distance (distinct): {wd_distinct:.6f}")

    print("\nWasserstein Distance component test passed.")

if __name__ == "__main__":
    os.makedirs("models/wasserstein_distance", exist_ok=True)
    test_component()
