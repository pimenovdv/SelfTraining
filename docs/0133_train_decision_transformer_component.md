# Experiment: Decision Transformer (Offline RL via Sequence Modeling)

**Script:** `train_decision_transformer_component.py`

## Hypothesis
By framing offline reinforcement learning as a sequence modeling problem over state, action, and return-to-go (RTG) tokens, a transformer architecture with causal self-attention can learn an expert policy directly from offline trajectories without requiring temporal difference learning or value function approximation.

## Implementation Details
1. **Sequence Modeling:** Sequences are constructed as (state, action, return-to-go) triplets.
2. **Embeddings:** Each modality (state, action, RTG) is projected to a common embedding dimension `d_model` and summed with learned positional embeddings.
3. **Causal Attention:** A causal mask prevents the model from attending to future tokens, ensuring the action prediction at time $t$ only depends on states, actions, and RTGs up to time $t$.
4. **Action Predictor:** An MLP head predicts the next action, optimized via Mean Squared Error (MSE) loss against the target actions from the expert dataset.
5. **Gradient Clipping:** Strict gradient clipping was necessary to prevent exploding gradients and overflow during manual backpropagation.

## Results
- **Outcome:** The model successfully converged on the offline dataset, minimizing the MSE loss and accurately reproducing the expert policy.
- **Success:** Yes.
- **Fixes Required:** Encountered `RuntimeWarning: overflow encountered in matmul` and `RuntimeWarning: invalid value encountered in subtract` (NaNs) during initial training. Applied gradient clipping `np.clip(grad, -1.0, 1.0)` to all weight updates in both the Transformer block and Self-Attention module, and reduced the learning rate to `0.0005`, which stabilized training and led to successful convergence.

## Next Steps
Evaluate the model on more complex offline RL benchmarks with discrete actions and sub-optimal trajectory distributions to test its ability to condition on high returns.
