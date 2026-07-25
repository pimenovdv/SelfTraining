# Experiment 0032: Train State Space Model (SSM) Component

## Objective
To implement and train a discrete State Space Model (SSM) component from scratch using pure `numpy`. This serves to mathematically verify the core mechanism behind modern SSM-based architectures (like Mamba) which map continuous sequences to discrete representations via Euler/Zero-Order Hold discretization, learning efficient sequence transformations.

## Setup
*   **Script:** `train_ssm_component.py`
*   **Data:** Synthetic 1D sequence dataset designed for sequential dependency learning.
*   **Hyperparameters:** `state_dim` = 8, `epochs` = 10000, `learning_rate` = 0.1

## Execution
The training script was executed successfully.

## Results
*   **Status:** Success.
*   **Convergence:** The model successfully minimized the Mean Squared Error over 10000 epochs.
*   **Learning:** Backpropagation Through Time (BPTT) effectively computed gradients for the continuous matrices $A$, $B$, $C$, and the step size $\Delta$.
*   **Output:** The predictions closely matched the expected sequential targets.

## Observations & Next Steps
*   This experiment verifies that first-order Euler discretization ($\overline{A} \approx I + \Delta A, \overline{B} \approx \Delta B$) is differentiable and sufficient for learning state transitions on simple sequences.
*   The parameter $\Delta$ controls the continuous-to-discrete step scale, mimicking the learned timescale dynamics seen in HiPPO and S4 models.
*   Next steps could involve implementing data-dependent selective transitions (Selective SSMs / Mamba) where $B$, $C$, and $\Delta$ are functions of the input $X_t$.
