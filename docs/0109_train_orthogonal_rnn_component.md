# Experiment: Orthogonal RNN Component

**Script:** `train_orthogonal_rnn_component.py`
**Description:** Implementation and training of an Orthogonal Recurrent Neural Network (RNN) using the Cayley transform.
**Mathematical Basis:** Standard RNNs suffer from vanishing and exploding gradients over long sequences. Orthogonal RNNs constrain the hidden-to-hidden weight matrix to be orthogonal, ensuring its eigenvalues have an absolute value of 1, thereby preserving gradient norms. The Cayley transform parametrizes an orthogonal matrix $W = (I - A)(I + A)^{-1}$ using a skew-symmetric matrix $A = V - V^T$, where $V$ is unconstrained.

## Results
- **Final Loss:** 0.2740
- **Status:** Success
- **Observations:** The Orthogonal RNN successfully learned the sequential task, demonstrating stable training. The Cayley transform provided an effective way to maintain orthogonality through standard gradient descent on the unconstrained parameters $V$.

## Usage
To run the component:
```bash
python train_orthogonal_rnn_component.py
```
