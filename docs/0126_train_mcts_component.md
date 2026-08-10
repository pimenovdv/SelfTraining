# Experiment: Train MCTS Component

**Script:** `train_mcts_component.py`

## Objective
To implement and verify a mathematical model of Monte Carlo Tree Search (MCTS) combined with a neural network for policy and value evaluation, simulating core elements of AlphaZero-style planning.

## Implementation Details
The implementation constructs a `Node` structure for tree search and an `AlphaZeroNet` for state evaluation.
- The `MCTS` algorithm simulates trajectories from the current state, using the neural network to evaluate leaf nodes and provide priors for actions.
- Action selection uses a variant of the PUCT formula, balancing exploration (driven by network priors and visit counts) with exploitation (empirical Q-values).
- The neural network has two heads: a policy head ($\pi$) outputting a probability distribution over actions, and a value head ($V$) outputting the expected return, constrained via `tanh`.
- During self-play, MCTS produces an improved policy target (visit distribution). The network is trained to minimize the cross-entropy with the MCTS policy and the mean squared error with the eventual return.

## Results
The model successfully learned to navigate the simple 1D grid world environment to the target state with a positive reward.

- State 1 (Left adjacent to trap): Network learned high value and a policy directing away from the trap.
- State 2 (Start): Network learned a positive value and a policy biased towards the goal.
- State 3 (Right adjacent to goal): Network learned high value and a policy directing towards the goal.

## Conclusion
The successful training of the MCTS component mathematically validates the AlphaZero mechanism of using search to generate policy improvement operators and training neural representations to internalize the search outcomes.
