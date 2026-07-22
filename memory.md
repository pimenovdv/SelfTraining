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

## Open Questions & Hypotheses

1. *(e.g., "Does scaling the depth of the network linearly correlate with reasoning capability on dataset Y?")*
2. *(e.g., "Can we formulate a strict mathematical bound on the error introduced by quantization technique Z?")*
