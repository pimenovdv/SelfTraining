import numpy as np

def ema(x_series, beta=0.9):
    """
    Computes the Exponential Moving Average (EMA) of a sequence of values.
    EMA_t = beta * EMA_{t-1} + (1 - beta) * X_t
    """
    ema_series = np.zeros_like(x_series, dtype=np.float64)
    if x_series.shape[0] == 0:
        return ema_series

    ema_val = x_series[0]
    for t in range(x_series.shape[0]):
        # Update EMA
        ema_val = beta * ema_val + (1 - beta) * x_series[t]
        ema_series[t] = ema_val

    return ema_series

def test_component():
    np.random.seed(42)
    # Generate a random sequence of values
    T = 10
    features = 2
    x = np.random.randn(T, features)

    beta = 0.9

    print("Input Series shape:", x.shape)

    # Calculate EMA
    ema_output = ema(x, beta=beta)

    print("EMA Output shape:", ema_output.shape)

    # Mathematical verification
    expected_t0 = beta * x[0] + (1 - beta) * x[0]
    expected_t1 = beta * expected_t0 + (1 - beta) * x[1]

    assert np.allclose(ema_output[0], expected_t0), "Mismatch at t=0"
    assert np.allclose(ema_output[1], expected_t1), "Mismatch at t=1"

    print("EMA logic verified successfully.")

if __name__ == "__main__":
    test_component()
