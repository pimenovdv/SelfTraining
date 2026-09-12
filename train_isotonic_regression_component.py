import numpy as np
import time

def pool_adjacent_violators(y, w=None):
    """
    Isotonic regression using the Pool Adjacent Violators Algorithm (PAVA).
    Args:
        y: 1D numpy array of values to be monotonically increasing.
        w: 1D numpy array of weights.
    Returns:
        1D numpy array of isotonic regression values.
    """
    n = len(y)
    if w is None:
        w = np.ones(n)

    # We will maintain blocks of monotonically increasing values.
    # Each block is represented by (start_index, end_index, value, weight)
    blocks = []

    for i in range(n):
        # Create a new block for the current element
        blocks.append({'start': i, 'end': i, 'val': y[i], 'w': w[i]})

        # Merge backwards if the monotonically increasing condition is violated
        while len(blocks) > 1 and blocks[-2]['val'] >= blocks[-1]['val']:
            # Pool the last two blocks
            b1 = blocks.pop(-2)
            b2 = blocks.pop(-1)

            # Weighted average
            new_w = b1['w'] + b2['w']
            new_val = (b1['val'] * b1['w'] + b2['val'] * b2['w']) / new_w

            blocks.append({
                'start': b1['start'],
                'end': b2['end'],
                'val': new_val,
                'w': new_w
            })

    # Reconstruct the fitted values
    y_fitted = np.zeros(n)
    for b in blocks:
        y_fitted[b['start']:b['end']+1] = b['val']

    return y_fitted

class IsotonicRegression:
    def __init__(self):
        self.x_train_ = None
        self.y_train_ = None
        self.y_fitted_ = None

    def fit(self, X, y):
        """
        X must be 1D. We sort X and y, then apply PAVA.
        """
        # Ensure 1D
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()

        # Sort based on X
        sort_idx = np.argsort(X)
        self.x_train_ = X[sort_idx]
        y_sorted = y[sort_idx]

        self.y_fitted_ = pool_adjacent_violators(y_sorted)

    def predict(self, X):
        """
        Predict by interpolating between the fitted points.
        """
        X = np.asarray(X).flatten()
        # Linear interpolation for values within the training range,
        # and constant extrapolation for out-of-bounds values.
        return np.interp(X, self.x_train_, self.y_fitted_)

def test_isotonic_regression():
    print("Testing Isotonic Regression (PAVA)...")

    # Generate some synthetic non-decreasing data with noise
    np.random.seed(42)
    X = np.random.uniform(0, 10, size=50)
    y = np.log1p(X) + np.random.normal(0, 0.2, size=50)

    start_time = time.time()

    iso_reg = IsotonicRegression()
    iso_reg.fit(X, y)

    y_pred = iso_reg.predict(X)

    end_time = time.time()

    mse = np.mean((y - y_pred)**2)

    print(f"MSE: {mse:.4f}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # Verify monotonicity
    sorted_idx = np.argsort(X)
    y_pred_sorted = y_pred[sorted_idx]
    is_monotonic = np.all(np.diff(y_pred_sorted) >= -1e-8)

    print(f"Predictions are monotonically increasing: {is_monotonic}")
    assert is_monotonic, "Isotonic regression constraint violated."
    print("Isotonic Regression (PAVA) component successfully tested!")

if __name__ == "__main__":
    test_isotonic_regression()
