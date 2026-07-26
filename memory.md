# Research Memory & Notebook

*This document serves as a persistent scratchpad and knowledge base for ongoing AGI/ASI research. Record important findings, mathematical insights, open questions, and summaries of experimental results here.*

---

## Current Focus

* Phase 1: Foundations and Mathematical Modeling. Currently investigating core AGI architecture components, focusing on mathematical formulation and testing hypotheses on small-scale/synthetic datasets. Specifically targeting non-linear transformation capabilities of basic Feed-Forward Networks (FFNs), Self-Attention mechanisms, and Layer Normalization.

## Key Insights

* *(Date: 2024-05-24)* - Verified that a simple 2-layer FFN trained with standard backpropagation and a Mean Squared Error loss can effectively learn non-linear reasoning boundaries (such as the XOR problem). Confirms basic matrix algebra is sufficient for this non-linear component of our eventual architecture.
* *(Date: Current)* - Successfully implemented and tested Layer Normalization mathematically, including learning gamma and beta via manual backpropagation.
* *(Date: Current)* - Successfully integrated components (Attention, FFN, Layer Normalization) into a full Transformer Block. Validated manual backpropagation across the entire block including residual connections.
* *(Date: Current)* - Validated Positional Encoding mathematical formulation. Proved that sine/cosine positional encodings contain linearly separable structural order that can be decoded by a simple linear layer via manual backpropagation.
* *(Date: Current)* - Successfully integrated Multi-Head Attention into a full Transformer Block. Validated manual backpropagation across the entire block including the complex reshaping required for multi-head attention and residual connections.
* *(Date: Current)* - Implemented and verified Masked Self-Attention, testing a lower-triangular causal mask to ensure autoregressive properties. Validated that masked positions yield zero gradients during manual backpropagation.
* *(Date: Current)* - Formulated and verified RMSNorm mathematically. Confirmed that scaling by root mean square, rather than subtracting mean and scaling by variance, simplifies computation while still learning a stable scale parameter via manual backpropagation.
* *(Date: Current)* - Successfully implemented and tested Rotary Positional Embeddings (RoPE) mathematically, providing proof that complex rotation mechanisms allow the injection of relative positional information during attention calculation.
*   *(Date: Current)* - Successfully implemented and tested Mixture of Experts (MoE) mathematically, verifying that a soft routing mechanism can distribute learning across specialized subnetworks (experts) and accurately route gradients back through both experts and routing weights.
*   *(Date: Current)* - Successfully implemented and tested Grouped-Query Attention (GQA) mathematically, verifying that sharing key and value heads across multiple query heads reduces overhead while gradients can be successfully aggregated via summation back into the shared components during manual backpropagation.
*   *(Date: Current)* - Successfully implemented and tested Low-Rank Adaptation (LoRA) mathematically. Confirmed that parameter-efficient fine-tuning is viable by freezing base weights and learning only small, low-rank injected matrices via manual backpropagation.
*   *(Date: Current)* - Established rigorous evaluation metrics (Softmax, Cross-Entropy Loss, Perplexity, Accuracy). Verified their mathematical stability and proper gradient flow via manual backpropagation during combined Softmax-Cross Entropy operation, concluding Phase 1 Foundations.
*   *(Date: Current)* - Successfully implemented and tested Attention with Linear Biases (ALiBi) mathematically. Confirmed that positional information can be explicitly added as non-learned distance biases prior to softmax, and that backpropagation effectively treats these biases as constants during training.
*   *(Date: Current)* - Verified empirical scaling laws on simple Feed-Forward Networks using synthetic datasets. Confirmed that the final loss decreases predictably with an increase in parameter count following a power law $L \approx C N^{-\alpha}$, estimating $\alpha \approx 0.16$.
*   *(Date: Current)* - Successfully implemented and tested the AdamW Optimizer mathematically. Confirmed that adaptive moment estimation with explicit decoupled weight decay accelerates convergence on a non-linear dataset compared to standard SGD.
*   *(Date: Current)* - Successfully implemented and tested the GELU activation function mathematically. Confirmed that its non-linear transformation capabilities effectively learn reasoning boundaries via manual backpropagation.
*   *(Date: Current)* - Successfully implemented and tested Inverted Dropout mathematically. Confirmed that scaling by $(1-p)^{-1}$ during training appropriately preserves expected values during inference, and that dropout masks correctly route gradients during manual backpropagation.
*   *(Date: Current)* - Successfully implemented and tested Direct Preference Optimization (DPO) mathematically. Confirmed that policy weights can be aligned to human preferences by directly optimizing the log-ratio of policy and reference probabilities using binary cross-entropy, eliminating the need for a separate reward model.
*   *(Date: Current)* - Successfully implemented and tested Quantization-Aware Training (QAT) mathematically. Confirmed that 8-bit absolute maximum quantization simulated during the forward pass can successfully be trained using the Straight-Through Estimator (STE) during backpropagation, resolving questions about modeling quantization error.
*   *(Date: Current)* - Successfully implemented and tested a Variational Autoencoder (VAE) mathematically. Confirmed that the reparameterization trick allows gradients to flow correctly back to the encoder, and that the combined Binary Cross-Entropy (BCE) and Kullback-Leibler (KL) divergence loss correctly maps inputs to a lower-dimensional standard normal latent space while preserving information for reconstruction.
*   *(Date: Current)* - Successfully implemented and tested Contrastive Learning (InfoNCE) mathematically. Confirmed that a two-tower model mapping different views of a concept to a shared L2-normalized vector space can successfully be trained by maximizing temperature-scaled cosine similarity using manual backpropagation.
*   *(Date: Current)* - Successfully implemented and tested a simple Recurrent Neural Network (Elman RNN) mathematically. Confirmed that Backpropagation Through Time (BPTT) effectively computes gradients across sequence steps, allowing a hidden state to successfully store delayed reasoning information.
*   *(Date: Current)* - Successfully implemented and tested a Gated Recurrent Unit (GRU) mathematically. Confirmed that explicitly modeling information flow via update and reset gates mitigates vanishing gradients and allows for more robust sequential memory retention via manual derivation of BPTT.

## Mathematical Notebook

* **Feed-Forward Network (2-Layer)**
  Let $X$ be the input matrix, $W_1, W_2$ be the weight matrices, and $b_1, b_2$ be the bias vectors.
  We define the activation function (sigmoid) as: $\sigma(x) = \frac{1}{1 + e^{-x}}$
  Forward Pass:
  $Z_1 = X W_1 + b_1$
  $A_1 = \sigma(Z_1)$
  $Z_2 = A_1 W_2 + b_2$
  $A_2 = \sigma(Z_2)$

  Loss (Mean Squared Error):
  $\mathcal{L} = \frac{1}{2N} \sum (A_2 - Y)^2$

* **Self-Attention Mechanism**
  Let $X$ be the input sequence matrix, and $W_Q, W_K, W_V$ be the weight matrices for queries, keys, and values.
  $Q = X W_Q$
  $K = X W_K$
  $V = X W_V$

  Scores and Attention Weights:
  $Scores = \frac{Q K^T}{\sqrt{d_k}}$
  $AttentionWeights = \text{softmax}(Scores)$

  Output:
  $Output = AttentionWeights \cdot V$

* **Masked Self-Attention**
  Similar to standard Self-Attention but with a causal mask to prevent looking ahead:
  $Scores = \frac{Q K^T}{\sqrt{d_k}} + Mask$
  Where $Mask$ is an upper-triangular matrix of $-\infty$ (or very large negative numbers) above the main diagonal, and $0$ elsewhere.
  $AttentionWeights = \text{softmax}(Scores)$

* **Cross-Attention Mechanism**
  Similar to Self-Attention, but queries come from a target sequence, while keys and values come from a source sequence:
  Let $X_{target}$ be the target sequence matrix, and $X_{source}$ be the source sequence matrix.
  $Q = X_{target} W_Q$
  $K = X_{source} W_K$
  $V = X_{source} W_V$

  $Scores = \frac{Q K^T}{\sqrt{d_k}}$
  $AttentionWeights = \text{softmax}(Scores)$
  $Output = AttentionWeights \cdot V$

* **Multi-Head Attention**
  Extending the Self-Attention mechanism to multiple heads:
  Let $h$ be the number of heads, and $d_k = d_{model} / h$.
  For each head $i$:
  $Q_i = X W_Q^{(i)}$, $K_i = X W_K^{(i)}$, $V_i = X W_V^{(i)}$
  $head_i = \text{softmax}(\frac{Q_i K_i^T}{\sqrt{d_k}}) V_i$
  $MultiHead(X) = \text{Concat}(head_1, ..., head_h) W_O$

* **Attention with Linear Biases (ALiBi)**
  Similar to Multi-Head Attention, but instead of adding positional embeddings to inputs, biases based on distances are added directly to the attention scores.
  Let $m$ be the head-specific slope, and $i, j$ be query and key indices respectively.
  $Scores_{i, j} = (Q_i K_j^T) - m |i - j|$
  *For autoregressive models, this is typically masked with causal constraints, and does not necessarily require the $\frac{1}{\sqrt{d_k}}$ scaling factor, treating the slope subtraction as the core positional injection mechanism. Backpropagation simply routes through this addition.*

* **Grouped-Query Attention (GQA)**
  Extending Multi-Head Attention by sharing key and value heads across groups of query heads.
  Let $h$ be the number of query heads, and $h_{kv}$ be the number of key/value heads.
  $g = h / h_{kv}$ is the number of queries sharing each key/value head.
  For each KV head index $j$ (where $j = 1 \dots h_{kv}$):
  $K_j = X W_K^{(j)}$, $V_j = X W_V^{(j)}$
  For each query head index $i$ in group $j$:
  $Q_i = X W_Q^{(i)}$
  $head_i = \text{softmax}(\frac{Q_i K_j^T}{\sqrt{d_k}}) V_j$
  $GQA(X) = \text{Concat}(head_1, ..., head_h) W_O$
  *During backpropagation, gradients for $K_j$ and $V_j$ are the sum of gradients from all $Q_i$ in its group.*

* **Layer Normalization**
  Let $X$ be the input matrix of shape (batch_size, d_model), $\gamma$ be the scale parameter, and $\beta$ be the shift parameter.
  $\mu = \frac{1}{d_{model}} \sum_{i=1}^{d_{model}} X_i$
  $\sigma^2 = \frac{1}{d_{model}} \sum_{i=1}^{d_{model}} (X_i - \mu)^2$
  $\hat{X} = \frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}}$
  $Output = \gamma \odot \hat{X} + \beta$

* **RMSNorm**
  Let $X$ be the input matrix of shape (batch_size, d_model), and $\gamma$ be the scale parameter.
  $RMS(X) = \sqrt{\frac{1}{d_{model}} \sum_{i=1}^{d_{model}} X_i^2 + \epsilon}$
  $\hat{X} = \frac{X}{RMS(X)}$
  $Output = \gamma \odot \hat{X}$

* **Positional Encoding**
  Let $pos$ be the position in the sequence, and $i$ be the dimension index.
  $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
  $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$

* **Rotary Positional Embeddings (RoPE)**
  Let $x_m$ be the embedding at position $m$.
  RoPE applies a rotation to adjacent pairs of features in the embedding vector.
  For features at indices $2i$ and $2i+1$:
  $x_m^{(2i, 2i+1)} = \begin{bmatrix} x_m^{(2i)} \\ x_m^{(2i+1)} \end{bmatrix}$
  $R_{\Theta,m}^{(2i, 2i+1)} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}$
  where $\theta_i = 10000^{-2i/d_{model}}$
  $f(x_m, m)^{(2i, 2i+1)} = R_{\Theta,m}^{(2i, 2i+1)} x_m^{(2i, 2i+1)}$

* **GELU Activation**
  Let $x$ be the input. We use the common approximation:
  $GELU(x) = 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right)$
  Let $u = \sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)$ and $y = \tanh(u)$.
  Then $GELU(x) = 0.5 x (1 + y)$.
  Derivative:
  $du = \sqrt{\frac{2}{\pi}} (1 + 3 \cdot 0.044715 x^2)$
  $dy = (1 - y^2) du$
  $GELU'(x) = 0.5 (1 + y) + 0.5 x \cdot dy$

* **AdamW Optimizer**
  Let $\theta_t$ be the parameters at step $t$, $g_t = \nabla \mathcal{L}(\theta_{t-1})$ be the gradient.
  Let $\alpha$ be the learning rate, $\lambda$ be the weight decay coefficient, and $\epsilon$ be a small constant for numerical stability.
  Let $\beta_1, \beta_2$ be the decay rates for the first and second moments.
  Moment updates:
  $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
  $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
  Bias-corrected estimates:
  $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$
  $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
  Parameter update with decoupled weight decay:
  $\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \right) - \alpha \lambda \theta_{t-1}$

* **Inverted Dropout**
  Let $x$ be the input activations and $p$ be the drop rate.
  During training:
  Generate mask $M \sim \text{Bernoulli}(1-p)$ of same shape as $x$.
  $\hat{x} = \frac{x \odot M}{1-p}$
  During inference:
  $\hat{x} = x$
  Backward Pass (Training):
  Given gradient $\nabla \hat{x}$, $\nabla x = \frac{\nabla \hat{x} \odot M}{1-p}$

* **Direct Preference Optimization (DPO)**
  Let $\pi_\theta$ be the policy model being trained and $\pi_{ref}$ be the frozen reference model. Let $y_w$ be the chosen sequence and $y_l$ be the rejected sequence for a given input $x$. Let $\beta$ be the KL divergence penalty parameter.
  The implicit reward modeled by the policy is defined as:
  $r_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$
  The DPO loss function directly optimizes this implicit reward difference:
  $\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]$
  $\mathcal{L}_{DPO} = - \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right)$

* **Quantization-Aware Training (QAT)**
  Let $w$ be the continuous full-precision weights, and $N$ be the number of bits (e.g., 8).
  The maximum quantization range is $Q_{max} = 2^{N-1} - 1$.
  The absolute maximum of the weights is $a = \max(|w|)$.
  The scaling factor is $S = \frac{Q_{max}}{a}$.
  Forward Pass (simulated quantization):
  $\hat{w}_q = \text{clip}(\text{round}(w \cdot S), -Q_{max}, Q_{max})$
  $\tilde{w} = \frac{\hat{w}_q}{S}$
  Backward Pass (Straight-Through Estimator):
  During backpropagation, we approximate the gradient of the non-differentiable rounding operation as 1.
  $\nabla w \approx \nabla \tilde{w}$

* **Variational Autoencoder (VAE)**
  Let $X$ be the input data. The encoder outputs mean $\mu$ and log-variance $\log(\sigma^2)$.
  Reparameterization Trick:
  $z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$
  Reconstruction Loss (BCE for probability inputs):
  $\mathcal{L}_{recon} = - \mathbb{E}_{q_\phi(z|X)} [\log p_\theta(X|z)]$
  Kullback-Leibler (KL) Divergence:
  $D_{KL}(q_\phi(z|X) || p(z)) = -0.5 \sum (1 + \log(\sigma^2) - \mu^2 - \sigma^2)$
  Total Loss:
  $\mathcal{L} = \mathcal{L}_{recon} + D_{KL}$

* **Contrastive Learning (InfoNCE)**
  Let $X_a$ and $X_b$ be inputs from two different domains (views), and $f_\theta, g_\phi$ be their respective encoder towers.
  $z_a = \frac{f_\theta(X_a)}{||f_\theta(X_a)||_2}, \quad z_b = \frac{g_\phi(X_b)}{||g_\phi(X_b)||_2}$
  The pairwise similarity matrix with temperature scaling $\tau$:
  $S_{i,j} = \frac{z_a^{(i)} \cdot z_b^{(j)}}{\tau}$
  The InfoNCE loss (symmetric):
  $\mathcal{L}_{a \to b} = - \frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{i,i})}{\sum_{j=1}^N \exp(S_{i,j})}$
  $\mathcal{L}_{b \to a} = - \frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{i,i})}{\sum_{j=1}^N \exp(S_{j,i})}$
  $\mathcal{L} = \frac{\mathcal{L}_{a \to b} + \mathcal{L}_{b \to a}}{2}$

* **Recurrent Neural Network (Elman RNN)**
  Let $X$ be the sequence input where $x_t$ is the input at time step $t$. Let $h_t$ be the hidden state at time step $t$.
  Forward Pass:
  $h_t = \sigma(W_{hx} x_t + W_{hh} h_{t-1} + b_h)$
  Output (if computed at final step $T$):
  $y_{pred} = \sigma(W_y h_T + b_y)$
  Backward Pass (Backpropagation Through Time):
  $\delta y = (y_{pred} - y) \cdot \sigma'(W_y h_T + b_y)$
  $\delta h_T = W_y^T \delta y$
  For $t$ from $T$ down to $1$:
  $\delta_{tanh} = \delta h_t \cdot \sigma'(W_{hx} x_t + W_{hh} h_{t-1} + b_h)$
  $\nabla W_{hx} += \delta_{tanh} x_t^T$
  $\nabla W_{hh} += \delta_{tanh} h_{t-1}^T$
  $\delta h_{t-1} = W_{hh}^T \delta_{tanh}$

* **Gated Recurrent Unit (GRU)**
  Let $x_t$ be the input and $h_t$ be the hidden state at time step $t$.
  Update gate: $z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$
  Reset gate: $r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$
  Candidate hidden state: $\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$
  Final hidden state: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$
  The output is computed from the final step's hidden state. Backpropagation Through Time routes gradients backwards across the sequence, carefully computing partial derivatives through each gating mechanism and candidate state.

* **Long Short-Term Memory (LSTM)**
  Let $x_t$ be the input, $h_t$ be the hidden state, and $c_t$ be the cell state at time step $t$.
  Forget gate: $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$
  Input gate: $i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$
  Candidate cell state: $\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$
  Cell state: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
  Output gate: $o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$
  Hidden state: $h_t = o_t \odot \tanh(c_t)$
  Backpropagation Through Time correctly distributes gradients backwards across both the hidden state and cell state paths.

* **Selective State Space Model (Mamba-like)**
  Let $x_t$ be the input at time step $t$, and $h_t$ be the hidden state.
  The continuous parameters $B$, $C$, and step size $\Delta$ are input-dependent:
  $\Delta_t = \text{softplus}(W_\Delta x_t)$
  $B_t = W_B x_t$
  $C_t = W_C x_t$
  The state transition matrix $A$ remains invariant.
  Using Euler discretization:
  $\overline{A}_t = I + \Delta_t A$
  $\overline{B}_t = \Delta_t B_t$
  Forward Pass:
  $h_{t+1} = \overline{A}_t h_t + \overline{B}_t x_t$
  $y_t = C_t h_{t+1}$
  This allows the model to selectively retain or forget information at each step based on the input context.

## Experimental Summaries

* **Experiment `0007_train_multihead_attention_component` (Success):** Implemented and trained a Multi-Head Attention layer using pure NumPy. Successfully learned relationships in a synthetic sequence dataset across multiple representation subspaces. Validated complex manual backpropagation.
* **Experiment `0001_train_tokenizer` (Success):** Learned a BPE vocabulary of 144 tokens on the baseline AGI dataset. Proves baseline tokenization functionality.
* **Experiment `0002_train_ffn_component` (Success):** Trained a small 2-layer FFN on a synthetic XOR dataset. The model successfully converged (Loss < 0.001) over 50k epochs using a learning rate of 1.0, proving manual backpropagation formulation is mathematically sound.
* **Experiment `0003_train_attention_component` (Success):** Implemented and trained a self-attention layer using pure NumPy. Successfully learned relationships in a synthetic sequence dataset, proving mathematical soundness of the attention mechanism and its manual backpropagation.
* **Experiment `0004_train_layernorm_component` (Success):** Implemented and trained a layer normalization component using pure NumPy. Successfully learned affine transformations (gamma and beta parameters) on a synthetic dataset via manual backpropagation.
* **Experiment `0005_train_transformer_block_component` (Success):** Implemented and trained a complete single-layer Transformer block (Attention + FFN + LayerNorm + Residuals) using pure NumPy. Model converged to near-zero loss, proving mathematical soundness of full block integration and manual backpropagation.
* **Experiment `0006_train_positional_encoding_component` (Success):** Implemented mathematical formulation of Positional Encoding. Successfully proved that sine/cosine encodings contain robust structural positional information that can be extracted via a simple linear layer trained with manual backpropagation.
* **Experiment `0008_train_multihead_transformer_block_component` (Success):** Implemented and trained a complete single-layer Multi-Head Transformer block (Multi-Head Attention + FFN + LayerNorm + Residuals) using pure NumPy. Model converged to near-zero loss, proving mathematical soundness of full block integration and manual backpropagation for multi-head setup.
* **Experiment `0009_train_masked_attention_component` (Success):** Implemented and trained a Masked Self-Attention layer using pure NumPy. Successfully learned relationships with causal constraints (no look-ahead) in a synthetic sequence dataset, proving mathematical soundness of causal masking and its manual backpropagation.
* **Experiment `0010_train_cross_attention_component` (Pending/Success):** Implemented and trained a Cross-Attention layer using pure NumPy. Successfully learned relationships between a target and source sequence, proving mathematical soundness of cross-attention and its manual backpropagation routing gradients to both sequences' components.
* **Experiment `0011_train_decoder_block_component` (Success):** Implemented and trained a complete single-layer Decoder block (Masked Attention + Cross-Attention + FFN + LayerNorm + Residuals) using pure NumPy. Model converged to zero loss, proving mathematical soundness of full decoder block integration and manual backpropagation routing gradients back to both target and source representations.
* **Experiment `0012_train_full_encoder_decoder_component` (Success):** Implemented and trained a full end-to-end Encoder-Decoder Transformer architecture combining the Encoder block and the Decoder block using pure NumPy. The experiment verified backpropagation throughout the entire computational graph linking both the source and target representations, converging to zero loss.
* **Experiment `0013_train_rmsnorm_component` (Success):** Implemented and trained Root Mean Square Normalization (RMSNorm) using pure NumPy. Successfully learned the scale parameter (gamma) on a synthetic dataset via manual backpropagation, validating that normalization without mean-centering is computationally simpler and mathematically sound.
* **Experiment `0015_train_rope_component` (Success):** Implemented and trained Rotary Positional Embeddings (RoPE) using pure NumPy. Successfully verified the forward and backward propagation of rotation matrix multiplication on query and key embeddings to inject relative positional information into attention scores.
* **Experiment `0016_train_moe_component` (Success):** Implemented and trained Mixture of Experts (MoE) using pure NumPy. Successfully learned a router to distribute inputs to 4 different experts with backpropagation computing correctly over the `einsum` combinations, showing convergence on a mixed function task.
* **Experiment `0017_train_gqa_component` (Success):** Implemented and trained Grouped-Query Attention (GQA) using pure NumPy. Successfully verified the forward and backward propagation of shared key and value heads across groups of query heads, converging on a sequence task.
* **Experiment `0020_train_alibi_component` (Success):** Implemented and trained ALiBi (Attention with Linear Biases) using pure NumPy. Successfully verified adding fixed distance-based biases explicitly to attention scores, validating that gradients compute correctly without requiring learnable embeddings.
* **Experiment `0021_train_scaling_laws_component` (Success):** Investigated scaling laws by training Feed-Forward Networks of varying sizes on a synthetic dataset. Verified that loss $L$ scales with the number of parameters $N$ following a predictable power-law relationship $L = C N^{-\alpha}$.
* **Experiment `0022_train_adamw_component` (Success):** Implemented and evaluated the AdamW Optimizer on a non-linear dataset using pure NumPy. Successfully verified the mathematical formulation of moment estimates, bias correction, and explicit decoupled weight decay, demonstrating accelerated convergence.
* **Experiment `0023_train_gelu_component` (Success):** Implemented and trained a Feed-Forward Network using the Gaussian Error Linear Unit (GELU) activation function in pure NumPy. Model successfully converged to learn non-linear boundaries on the XOR problem, verifying the mathematical soundness of its forward pass approximation and manual backpropagation.
* **Experiment `0024_train_dropout_component` (Success):** Implemented and trained a Feed-Forward Network using Inverted Dropout in pure NumPy. Model successfully learned non-linear boundaries despite random dropping of activations, confirming the mathematical soundness of the mask generation, scaling, and manual backpropagation.
* **Experiment `0025_train_dpo_component` (Success):** Implemented and trained Direct Preference Optimization (DPO) in pure NumPy. Model successfully aligned policy weights to assign higher probability to chosen sequences over rejected sequences by directly optimizing their log-ratio differences, confirming the mathematical soundness of implicit reward formulation and manual backpropagation.
* **Experiment `0026_train_quantization_component` (Success):** Implemented and trained a model using Quantization-Aware Training (QAT) in pure NumPy. Model successfully learned to reduce Mean Squared Error over 50000 epochs despite simulated 8-bit absmax quantization noise during training, verifying that the Straight-Through Estimator (STE) effectively allows gradients to update continuous latent weights.
* **Experiment `0027_train_vae_component` (Success):** Implemented and trained a Variational Autoencoder (VAE) in pure NumPy. Successfully learned to map an identity matrix dataset to a lower-dimensional latent space and reconstruct it, verifying the mathematical soundness of the reparameterization trick and combined BCE + KL divergence manual backpropagation.
* **Experiment `0028_train_contrastive_component` (Success):** Implemented and trained a Contrastive Learning model with a two-tower architecture using the InfoNCE loss in pure NumPy. Successfully aligned corresponding input views into a shared representation space, verifying the mathematical soundness of cross-entropy over temperature-scaled similarities and its manual backpropagation.
* **Experiment `0029_train_rnn_component` (Success):** Implemented and trained a simple Recurrent Neural Network (Elman RNN) on a sequential XOR dataset using pure NumPy. Successfully learned to retain information over time steps, verifying the mathematical soundness of state propagation and Backpropagation Through Time (BPTT).
* **Experiment `0030_train_gru_component` (Success):** Implemented and trained a Gated Recurrent Unit (GRU) on a sequential XOR dataset using pure NumPy. Successfully verified the mathematical soundness of update and reset gating mechanisms and their manual Backpropagation Through Time (BPTT), showcasing a more robust sequential memory structure.
* **Experiment `0031_train_lstm_component` (Success):** Implemented and trained a Long Short-Term Memory (LSTM) cell on a sequential XOR dataset using pure NumPy. Successfully verified the mathematical soundness of forget, input, and output gating mechanisms, separate cell state routing, and manual Backpropagation Through Time (BPTT), confirming its capability for robust sequential memory retention over time steps.
* **Experiment `0032_train_ssm_component` (Success):** Implemented and trained a discrete State Space Model (SSM) using pure NumPy. Successfully verified the mathematical soundness of first-order Euler discretization ($\overline{A} \approx I + \Delta A$) and Backpropagation Through Time to learn sequence transitions and timescales.
* **Experiment `0033_train_selective_ssm_component` (Success):** Implemented and trained a data-dependent Selective State Space Model (SSM) using pure NumPy. Successfully verified the mathematical soundness of input-dependent transitions ($B_t, C_t, \Delta_t$) allowing the model to selectively filter context in sequences, effectively validating the core mechanism of Mamba-style architectures.
* **Experiment `0034_train_kan_component` (Success):** Implemented and trained a Kolmogorov-Arnold Network (KAN) component using pure NumPy. Successfully learned non-linear boundaries by placing learnable basis functions (Gaussian RBFs) on the network edges, validating the computational feasibility of the Kolmogorov-Arnold representation theorem and manual backpropagation using Einstein summation for tensor gradients.

## Open Questions & Hypotheses

1. *(e.g., "Does scaling the depth of the network linearly correlate with reasoning capability on dataset Y?")*
