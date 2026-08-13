import numpy as np

def test_kalman_filter():
    print("Testing Kalman Filter component...")
    # Initialize random seed
    np.random.seed(42)

    # 1D motion with constant velocity
    # State: [position, velocity]^T
    dt = 1.0

    # State transition model
    F = np.array([[1, dt],
                  [0, 1]], dtype=np.float32)

    # Observation model (we only observe position)
    H = np.array([[1, 0]], dtype=np.float32)

    # Covariance of process noise
    Q = np.array([[1e-4, 0],
                  [0, 1e-4]], dtype=np.float32)

    # Covariance of observation noise
    R = np.array([[1e-1]], dtype=np.float32) # Increased noise to make filter work harder

    # Initial state estimate
    x_hat = np.array([[0], [0]], dtype=np.float32)

    # Initial error covariance
    P = np.array([[1, 0],
                  [0, 1]], dtype=np.float32)

    # Generate some true data and observations
    num_steps = 100
    true_states = []
    observations = []

    current_x = np.array([[0], [1]], dtype=np.float32) # start at pos 0, vel 1
    for _ in range(num_steps):
        # Update true state (with some process noise)
        current_x = F @ current_x + np.random.multivariate_normal([0, 0], Q).reshape(2, 1)
        true_states.append(current_x)

        # Generate observation (with observation noise)
        z = H @ current_x + np.random.normal(0, np.sqrt(R[0, 0]))
        observations.append(z)

    # Run Kalman Filter
    estimated_states = []

    for z in observations:
        # Predict
        x_hat_minus = F @ x_hat
        P_minus = F @ P @ F.T + Q

        # Update
        y = z - H @ x_hat_minus # Measurement residual
        S = H @ P_minus @ H.T + R # Residual covariance
        K = P_minus @ H.T @ np.linalg.inv(S) # Kalman gain

        x_hat = x_hat_minus + K @ y
        P = (np.eye(2) - K @ H) @ P_minus

        estimated_states.append(x_hat)

    # Calculate Mean Squared Error
    true_positions = np.array([x[0, 0] for x in true_states])
    observed_positions = np.array([z[0, 0] for z in observations])
    estimated_positions = np.array([x[0, 0] for x in estimated_states])

    mse_observed = np.mean((true_positions - observed_positions)**2)
    mse_estimated = np.mean((true_positions - estimated_positions)**2)

    print(f"MSE of raw observations: {mse_observed:.4f}")
    print(f"MSE of Kalman Filter estimates: {mse_estimated:.4f}")

    assert mse_estimated < mse_observed, "Kalman Filter should reduce the estimation error!"

    print("Kalman Filter successfully estimated the hidden state and reduced observation noise.")

if __name__ == "__main__":
    test_kalman_filter()
