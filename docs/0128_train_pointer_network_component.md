# Experiment 0128: Pointer Network Component

**Objective:** Implement a Pointer Network in pure NumPy to learn conditional probabilities over an input dictionary, addressing tasks where the output vocabulary depends entirely on the input sequence (e.g., sorting).

**Script:** `train_pointer_network_component.py`

**Hypothesis:** By modifying the attention mechanism to output probabilities directly over the input sequence rather than blending encoder states, a neural network can successfully learn to point to input elements, enabling it to solve algorithmic tasks like sorting.

**Methodology:**
- Built an RNN encoder-decoder architecture.
- Implemented the pointer attention mechanism $u_i^t = v^T \tanh(W_1 h_i + W_2 d_t)$.
- Applied softmax to $u^t$ to produce a probability distribution over the input sequence.
- Used Teacher Forcing during training on a sequence sorting task.
- Implemented manual backpropagation and the Adam optimizer.

**Results:**
- **Status:** Success. The network converged to a low loss (0.4165), effectively learning to point to the correct sorted elements.

**Next Steps:**
- Evaluate the Pointer Network on combinatorial optimization problems like the Traveling Salesperson Problem (TSP).
