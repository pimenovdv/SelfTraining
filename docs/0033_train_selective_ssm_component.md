# Experiment 0033: Train Selective State Space Model Component

## Objective
To implement and train a data-dependent Selective State Space Model (SSM) component from scratch using pure `numpy`. This verifies the mechanism behind models like Mamba, where transition parameters ($B_t, C_t, \Delta_t$) are functions of the input $x_t$, allowing the model to selectively remember or forget information across the sequence (unlike time-invariant SSMs).

## Setup
*   **Script:** `train_selective_ssm_component.py`
*   **Data:** Synthetic 1D sequence dataset designed for context-dependent accumulation and resetting.
*   **Hyperparameters:** `state_dim` = 8, `epochs` = 10000, `learning_rate` = 0.01

## Execution
The training script was executed successfully.

## Results
*   **Status:** Success.
*   **Convergence:** The model successfully minimized the Mean Squared Error over 10000 epochs.
*   **Learning:** Backpropagation Through Time (BPTT) effectively computed gradients through the input-dependent parameter projections ($W_B, W_C, W_\Delta$) and the invariant state transition matrix $A$.
*   **Output:** The predictions closely matched the expected sequential targets which required selective memory.

## Observations & Next Steps
*   This experiment verifies that projecting inputs to dynamically generate $B_t$, $C_t$, and $\Delta_t$ provides the necessary degrees of freedom for selective state filtering.
*   The gradients correctly route back through the Euler discretization $(\overline{A}_t = I + \Delta_t A, \overline{B}_t = \Delta_t B_t)$ to the projection weights.
*   This serves as the foundational mathematical verification for Mamba-style architectures in our AGI pathway.
