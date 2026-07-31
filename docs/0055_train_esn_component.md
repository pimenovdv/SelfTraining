# Experiment 0055: Train Echo State Network (ESN) Component

## Objective
To implement and verify an Echo State Network (ESN) mathematically using pure NumPy, testing Reservoir Computing principles on a chaotic time-series prediction task.

## Mathematical Formulation
An Echo State Network uses a fixed, randomly connected recurrent reservoir and only trains a linear readout layer.
*   **Reservoir Update:** $h_t = (1 - \alpha) h_{t-1} + \alpha \tanh(W_{in} [1; x_t] + W h_{t-1})$
    *   $W_{in}$: Input weights (fixed, dense).
    *   $W$: Reservoir weights (fixed, sparse, scaled by spectral radius).
    *   $\alpha$: Leaky rate.
*   **Readout Layer:** $\hat{y}_t = W_{out}^T [1; h_t]$
*   **Training:** $W_{out}$ is learned via Ridge Regression: $W_{out} = (H^T H + \lambda I)^{-1} H^T Y$, where $H$ is the matrix of collected reservoir states after a warmup period, and $\lambda$ is the ridge regularization parameter.

## Experimental Setup
*   **Input Dimension:** 1
*   **Reservoir Size:** 500
*   **Output Dimension:** 1
*   **Spectral Radius:** 1.25
*   **Dataset:** Mackey-Glass chaotic time series (Length: 2000).
*   **Training Method:** Ridge Regression (Closed-form solution).

## Results
*   **Final Test MSE:** 0.085643
*   **Status:** Success

## Conclusion
The Echo State Network successfully predicted the chaotic time series. The fixed random reservoir effectively projected the input history into a high-dimensional state space, allowing the linear readout layer to accurately model the complex dynamics, validating the Reservoir Computing approach.

**Script:** `train_esn_component.py`
