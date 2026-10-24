**Script:** `train_cql_component.py`

# Experiment 0560: Conservative Q-Learning (CQL) Component
- **Objective:** Evaluate the Conservative Q-Learning (CQL) component mathematically in pure NumPy, testing its ability to mitigate overestimation in offline RL by explicitly penalizing Q-values of out-of-distribution actions.
- **Method:** We formulated a continuous CQL agent based on Soft Actor-Critic (SAC). The Q-network loss was augmented with a conservative penalty term that minimizes the LogSumExp of Q-values for uniformly sampled actions and maximizes the Q-values for actions present in the dataset.
- **Outcome:** The agent successfully computed the CQL penalty alongside the standard MSE loss, validating the mechanism for penalizing out-of-distribution actions and maintaining a lower-bound estimate of the Q-function. The implementation successfully converged and correctly calculated the conservative terms.
