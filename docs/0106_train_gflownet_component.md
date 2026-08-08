# Experiment: GFlowNet (Generative Flow Network)

**Script:** `train_gflownet_component.py`
**Description:** Evaluates a GFlowNet agent learning to generate compositional objects with probabilities proportional to a reward function, utilizing manual backpropagation on the Trajectory Balance loss.

## Objective
To test the hypothesis that a Generative Flow Network (GFlowNet) can learn to generate compositional objects (in this case, paths on a grid) such that the probability of generating an object is proportional to a given reward function. This is verified by checking if the learned partition function (log Z) matches the true log partition function, and by utilizing manual backpropagation on the Trajectory Balance loss.

## Methodology
- Implemented a GFlowNet agent navigating a 2D grid.
- Used a forward policy parameterized by a 1-layer MLP to generate trajectories.
- Used a uniform backward policy for simplicity.
- Optimized the Trajectory Balance (TB) loss objective: `(log Z + sum(log P_F) - log R - sum(log P_B))^2`.
- Used pure NumPy to compute the forward passes and manually route gradients back through the TB loss into the MLP and learnable `log Z` parameter.

## Results
- **Training:** The model successfully trained over 2000 epochs with a batch size of 64.
- **Verification:** The TB loss minimized to near zero.
- **Log Z matching:** The learned `log Z` parameter closely approximated the true `log Z` computed by summing rewards over all possible terminal states in the grid.
- **Success:** The manual backpropagation effectively updated both the policy weights and the partition function estimate, confirming the mathematical soundness of the GFlowNet Trajectory Balance objective.

## Conclusion
GFlowNets provide a robust mechanism for generating diverse samples proportional to reward, distinct from standard RL which seeks only the maximum reward. The manual implementation verifies that TB loss gradients correctly guide the generative policy.
