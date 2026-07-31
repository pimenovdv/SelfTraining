# Experiment 0065: Train Model-Agnostic Meta-Learning (MAML) Component

## Objective
To implement and evaluate a Model-Agnostic Meta-Learning (MAML) algorithm mathematically. This tests the hypothesis that a model can learn an internal representation (initialization parameters) that is broadly suitable for many tasks, enabling rapid adaptation to new, unseen tasks with only a few gradient steps (few-shot learning). We implement First-Order MAML (FOMAML) which omits second derivatives for computational efficiency.

## Setup
*   **Script:** `train_maml_component.py`
*   **Data:** A family of sine wave regression tasks: $y = A \sin(x + b)$, where amplitude $A \sim U[0.1, 5.0]$ and phase $b \sim U[0, \pi]$.
*   **Hyperparameters:** `epochs` = 1000, `meta_batch_size` = 16, `inner_lr` = 0.01, `outer_lr` = 0.001, `k_support` = 10, `k_query` = 10, `inner_steps` = 1

## Execution
The training script was executed to verify the mathematical formulation of MAML. The inner loop simulates adaptation to a specific task using a support set. The outer loop updates the initial parameters $\theta$ by computing the gradient of the loss on a query set using the adapted parameters $\theta'$. In FOMAML, we approximate $\nabla_\theta \mathcal{L}(\theta') \approx \nabla_{ \theta' } \mathcal{L}(\theta')$. Gradients were computed manually using standard backpropagation for the MLP.

## Results
*   **Status:** Success.
*   **Learning:** The meta-model successfully minimized the meta-loss across the epochs, learning an initialization that allows fast adaptation to new sine wave tasks with a single or few gradient steps.
*   **Evaluation:** The outer loop gradients reliably shifted the model's starting weights into a manifold where task-specific tuning is highly effective.

## Observations & Next Steps
*   The implementation validates the core mechanism of First-Order MAML. By directly differentiating the post-adaptation loss with respect to the pre-adaptation weights (approximated), the network discovers general features of sine waves.
*   Future explorations could implement exact MAML (requiring second-order derivatives or Hessian-vector products) or expand the tasks to few-shot classification (e.g., Omniglot or MiniImageNet analogs).
