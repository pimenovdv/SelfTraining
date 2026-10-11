import numpy as np

def b_spline_basis(x, grid, k, i):
    """
    Evaluates the i-th B-spline basis function of degree k at points x,
    given a knot sequence 'grid'.
    """
    if k == 0:
        return np.where((x >= grid[i]) & (x < grid[i+1]), 1.0, 0.0)
    else:
        # Prevent division by zero
        den1 = grid[i+k] - grid[i]
        term1 = 0.0
        if den1 > 0:
            term1 = ((x - grid[i]) / den1) * b_spline_basis(x, grid, k-1, i)

        den2 = grid[i+k+1] - grid[i+1]
        term2 = 0.0
        if den2 > 0:
            term2 = ((grid[i+k+1] - x) / den2) * b_spline_basis(x, grid, k-1, i+1)

        return term1 + term2

def test_b_spline():
    # Simple grid [0, 1, 2, 3, 4]
    grid = np.array([0, 1, 2, 3, 4], dtype=float)
    x = np.array([0.5, 1.5, 2.5])

    # Test degree 0
    res0 = b_spline_basis(x, grid, k=0, i=1) # Should be 1 in [1, 2), 0 otherwise
    print("Degree 0:", res0)

    # Test degree 1
    res1 = b_spline_basis(x, grid, k=1, i=1)
    print("Degree 1:", res1)

    assert np.allclose(res0, [0., 1., 0.])

    print("B-Spline Component mathematical validation passed!")

if __name__ == "__main__":
    test_b_spline()
