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

## Open Questions & Hypotheses

1. *(e.g., "Does scaling the depth of the network linearly correlate with reasoning capability on dataset Y?")*
2. *(e.g., "Can we formulate a strict mathematical bound on the error introduced by quantization technique Z?")*
