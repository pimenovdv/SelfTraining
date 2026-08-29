# Research Memory & Notebook

*This document serves as a persistent scratchpad and knowledge base for ongoing AGI/ASI research. Record important findings, mathematical insights, open questions, and summaries of experimental results here.*

---

## Current Focus

* Phase 1: Foundations and Mathematical Modeling. Currently investigating core AGI architecture components, focusing on mathematical formulation and testing hypotheses on small-scale/synthetic datasets. Specifically targeting non-linear transformation capabilities of basic Feed-Forward Networks (FFNs), Self-Attention mechanisms, and Layer Normalization.

## Key Insights
* *(Date: Current)* - Successfully implemented and tested a t-Distributed Stochastic Neighbor Embedding (t-SNE) component mathematically in pure NumPy. Confirmed that by converting high-dimensional Euclidean distances into conditional probabilities representing similarities, and minimizing the Kullback-Leibler divergence with a Student-t distribution in the lower-dimensional space, the model effectively learns structure-preserving dimensionality reduction using gradient descent with momentum and early exaggeration.
* *(Date: Current)* - Successfully implemented and tested an Intrinsic Curiosity Module (ICM) mathematically in pure NumPy. Confirmed that generating intrinsic reward based on forward model prediction error, while learning action-conditioned state representations via an inverse model, effectively encourages exploration without relying on extrinsic rewards via manual backpropagation.
* *(Date: Current)* - Successfully implemented and tested a Denoising Autoencoder (DAE) component mathematically in pure NumPy. Confirmed that by forcing the model to reconstruct clean data from noise-corrupted inputs, it learns robust, foundational representations rather than simply copying the input via manual backpropagation.
* *(Date: Current)* - Successfully implemented and tested an Evolution Strategies (ES) component mathematically in pure NumPy. Confirmed that black-box optimization via stochastic parameter perturbation and fitness-weighted updates effectively learns non-linear representations without calculating analytical gradients (backpropagation).
* *(Date: Current)* - Successfully implemented and tested a Gumbel-Softmax component mathematically in pure NumPy. Confirmed that using the reparameterization trick with Gumbel noise allows differentiable sampling from a categorical distribution, validating the temperature annealing and manual backpropagation.
* *(Date: Current)* - Successfully implemented and tested a Skip-Gram component with Negative Sampling mathematically in pure NumPy. Confirmed that maximizing the dot product of target-context embeddings while minimizing target-negative embeddings successfully clusters semantically similar representations via manual backpropagation.
* *(Date: Current)* - Successfully implemented and tested an End-To-End Memory Network (MemN2N) component mathematically in pure NumPy. Confirmed that computing soft attention over explicit memory representations using a query vector effectively routes necessary facts to generate accurate answers, successfully verifying its manual backpropagation through memory embeddings.
* *(Date: Current)* - Successfully implemented and tested a Graph Attention Network (GAT) component mathematically in pure NumPy. Confirmed that computing masked attention scores across neighbors based on concatenated and linearly transformed node features successfully updates representations, effectively modeling graph structure and validating its manual backpropagation.
* *(Date: Current)* - Successfully implemented and tested a Graph Convolutional Network (GCN) component mathematically in pure NumPy. Confirmed that normalizing the adjacency matrix and passing features through it effectively propagates information across nodes, validating its forward pass and manual backpropagation on a synthetic graph dataset.
* *(Date: Current)* - Successfully implemented and tested a Vision Transformer (ViT) component mathematically in pure NumPy. Confirmed that extracting non-overlapping patches, adding positional embeddings and a class token, and processing them via Multi-Head Self-Attention effectively classifies spatial patterns via manual backpropagation.
* *(Date: Current)* - Successfully implemented and tested an MLP-Mixer block mathematically in pure NumPy. Confirmed that a sequence of Token-mixing MLPs (operating across the sequence dimension on transposed features) and Channel-mixing MLPs (operating across the channel dimension) can effectively model sequences without attention mechanisms, routing gradients correctly through transpose operations via manual backpropagation.
* *(Date: Current)* - Successfully implemented and tested Batch Normalization mathematically. Confirmed that normalizing across the batch dimension and learning scale/shift parameters accelerates convergence and effectively routes gradients back through mean and variance calculations via manual backpropagation.
* *(Date: Current)* - Successfully implemented and tested Group Normalization mathematically. Confirmed that dividing channels into groups and normalizing within those groups allows stable normalization independent of batch size, correctly routing gradients through reshaped features via manual backpropagation.
* *(Date: Current)* - Successfully implemented and tested an Energy-Based Model (EBM) mathematically in pure NumPy. Confirmed that using Contrastive Divergence alongside Langevin Dynamics effectively learns an energy surface that models a continuous target distribution.
* *(Date: Current)* - Successfully implemented and tested Decoupled Neural Interfaces (DNI) mathematically in pure NumPy. Confirmed that using auxiliary networks to predict synthetic gradients allows layers to update asynchronously, bypassing standard backpropagation locks.

* *(Date: Current)* - Successfully implemented and tested a Generative Flow Network (GFlowNet) mathematically in pure NumPy. Confirmed that optimizing the Trajectory Balance loss successfully learns a policy that generates objects (paths) with probabilities proportional to a given reward, while correctly estimating the log partition function via manual backpropagation.
* *(Date: Current)* - Explored Grokking on a modular addition task mathematically in pure NumPy. Confirmed that standard cross-entropy and gradient descent initially memorize the algorithmic dataset by overfitting spurious patterns (reaching 100% train accuracy while test accuracy remains at random chance), forming the necessary pre-condition for the later structural representation generalization phase.
* *(Date: Current)* - Successfully implemented and tested Elastic Weight Consolidation (EWC) mathematically. Confirmed that computing the Fisher Information Matrix to approximate parameter importance and applying a targeted L2 penalty allows the model to learn a sequential task while significantly mitigating catastrophic forgetting of a previous task, verified via manual backpropagation.

* *(Date: Current)* - Successfully implemented and tested an Echo State Network (ESN) mathematically in pure NumPy. Confirmed that a fixed, random recurrent reservoir effectively projects temporal dynamics into a high-dimensional state space, allowing a simple linear readout trained via Ridge Regression to successfully predict a chaotic Mackey-Glass time series, validating Reservoir Computing principles.
* *(Date: Current)* - Successfully implemented and tested a Bayesian Neural Network (BNN) mathematically in pure NumPy. Confirmed that modeling weights as probability distributions via the reparameterization trick allows optimization of the Evidence Lower Bound (ELBO), effectively balancing predictive accuracy (NLL) with uncertainty estimation (KL divergence) using Bayes by Backprop.
*   *(Date: Current)* - Successfully implemented and tested a Continuous Normalizing Flow via Conditional Flow Matching (CFM) mathematically in pure NumPy. Confirmed that a neural network can learn to predict the constant vector field connecting a base Gaussian distribution to the data distribution, effectively modeling the probability flow ODE via manual backpropagation.
*   *(Date: Current)* - Successfully implemented and tested a Denoising Diffusion Probabilistic Model (DDPM) mathematically in pure NumPy. Confirmed that the reverse process can learn to predict and remove Gaussian noise incrementally, validating the mathematical formulation of the forward diffusion and the reverse step via manual backpropagation on a simple MLP.
*   *(Date: Current)* - Successfully implemented and tested Linear Attention mathematically. Confirmed that by applying a positive kernel feature map (like ELU + 1) to Queries and Keys, the attention calculation can be reformulated as `\phi(Q) (\phi(K)^T V)`, reducing the computational complexity from O(N^2) to O(N) while correctly computing gradients via manual backpropagation.
*   *(Date: Current)* - Successfully implemented and tested a Sparse Autoencoder (SAE) mathematically. Confirmed that an overcomplete hidden layer with an L1 penalty can learn to disentangle representations into sparse, interpretable features, validating its mechanism for mechanistic interpretability via manual backpropagation.

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
*   *(Date: Current)* - Successfully implemented and tested the REINFORCE policy gradient algorithm mathematically. Confirmed that optimizing expected returns via gradient ascent on the log probability of sampled actions (scaled by a baseline-adjusted return) allows a policy network to learn effective behaviors via manual backpropagation.
*   *(Date: Current)* - Successfully implemented and tested a Variational Autoencoder (VAE) mathematically. Confirmed that the reparameterization trick allows gradients to flow correctly back to the encoder, and that the combined Binary Cross-Entropy (BCE) and Kullback-Leibler (KL) divergence loss correctly maps inputs to a lower-dimensional standard normal latent space while preserving information for reconstruction.
*   *(Date: Current)* - Successfully implemented and tested a Vector Quantized Variational Autoencoder (VQ-VAE) mathematically. Confirmed that discrete representations can be learned via a codebook lookup, using the Straight-Through Estimator (STE) to successfully route gradients from the decoder back to the encoder, effectively ignoring the non-differentiable argmin step during backpropagation.
*   *(Date: Current)* - Successfully implemented and tested Contrastive Learning (InfoNCE) mathematically. Confirmed that a two-tower model mapping different views of a concept to a shared L2-normalized vector space can successfully be trained by maximizing temperature-scaled cosine similarity using manual backpropagation.
*   *(Date: Current)* - Successfully implemented and tested a simple Recurrent Neural Network (Elman RNN) mathematically. Confirmed that Backpropagation Through Time (BPTT) effectively computes gradients across sequence steps, allowing a hidden state to successfully store delayed reasoning information.
*   *(Date: Current)* - Successfully implemented and tested a Gated Recurrent Unit (GRU) mathematically. Confirmed that explicitly modeling information flow via update and reset gates mitigates vanishing gradients and allows for more robust sequential memory retention via manual derivation of BPTT.

## Mathematical Notebook

* **Skip-Gram (Negative Sampling)**
  Let $v_c$ be the target word embedding and $u_o$ be the context word embedding.
  The objective is to maximize $P(o|c) = \sigma(u_o^T v_c)$ for true contexts, and minimize $P(w|c) = \sigma(u_w^T v_c)$ for $k$ negative samples.
  Loss: $L = - \log \sigma(u_o^T v_c) - \sum_{i=1}^k \log \sigma(-u_{w_i}^T v_c)$.
  During backpropagation, gradients route backward to update the embedding matrices for both context and target vocabularies.

* **Graph Attention Network (GAT)**
  Let $X$ be the input feature matrix and $A$ be the adjacency matrix (with self-loops $A_{i,i}=1$).
  A shared linear transformation, parameterized by a weight matrix $W$, is applied to every node: $Z = X W$.
  Self-attention mechanism computes attention coefficients:
  $e_{ij} = \text{LeakyReLU}(a^T [Z_i || Z_j])$ where $||$ is concatenation and $a$ is the attention weight vector.
  Masked attention ensures we only compute $e_{ij}$ for nodes $j$ in the neighborhood of $i$ (where $A_{ij} > 0$):
  $\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(\text{LeakyReLU}(a^T [Z_i || Z_j]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(a^T [Z_i || Z_k]))}$
  The node features are updated as a linear combination of their neighbors' features:
  $H_i = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} Z_j\right)$
  During backpropagation, gradients route correctly backwards through the masked softmax attention and the concatenated feature vectors.

* **Graph Convolutional Network (GCN)**
  Let $X$ be the input feature matrix, $A$ be the adjacency matrix, and $I$ be the identity matrix.
  The normalized adjacency matrix is computed as $\hat{A} = A + I$.
  Let $\hat{D}$ be the diagonal node degree matrix of $\hat{A}$.
  The symmetric normalized adjacency matrix is $A_{norm} = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$.
  A single GCN layer is defined as:
  $H^{(l+1)} = \sigma(A_{norm} H^{(l)} W^{(l)})$
  where $W^{(l)}$ is the weight matrix and $\sigma$ is an activation function (e.g., ReLU).
  During backpropagation, gradients route backward through the feature transformations and the structural graph propagations defined by the normalized adjacency matrix.

* **Generative Adversarial Network (GAN)**
  Let $G$ be the generator mapping a prior noise distribution $p_z(z)$ to the data space, and $D$ be the discriminator estimating the probability that a sample came from the true data distribution $p_{data}(x)$ rather than $G$.
  The two models are trained simultaneously in a two-player minimax game with value function $V(G, D)$:
  $\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$
  In practice, to avoid vanishing gradients early in training for the generator, $G$ maximizes $\log(D(G(z)))$ instead of minimizing $\log(1 - D(G(z)))$.
  During backpropagation, gradients for $D$ flow from its output based on real/fake labels, and gradients for $G$ flow from $D$'s output back through $D$ (with $D$'s weights frozen) into $G$'s parameters.

* **Batch Normalization**
  Let $X$ be the input matrix of shape (batch_size, num_features).
  $\mu_B = \frac{1}{m} \sum_{i=1}^m x_i$ (batch mean)
  $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$ (batch variance)
  $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
  $Output_i = \gamma \hat{x}_i + \beta$
  During backpropagation, gradients route back through the scale ($\gamma$) and shift ($\beta$) parameters, and also through the normalization process which requires computing partial derivatives with respect to the batch variance and mean.

* **Group Normalization**
  Let $X$ be the input matrix of shape (batch_size, num_features) where num_features is the number of channels $C$.
  Channels are divided into $G$ groups, each of size $D = C/G$. The features are reshaped to (batch_size, $G$, $D$).
  $\mu_g = \frac{1}{D} \sum_{i=1}^D x_{g,i}$ (group mean)
  $\sigma_g^2 = \frac{1}{D} \sum_{i=1}^D (x_{g,i} - \mu_g)^2$ (group variance)
  $\hat{x}_{g,i} = \frac{x_{g,i} - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}$ (normalized value)
  The normalized features are reshaped back to (batch_size, $C$) and scaled by channel-wise parameters.
  $Output_c = \gamma_c \hat{x}_c + \beta_c$
  During backpropagation, gradients route back through $\gamma$ and $\beta$, and through the reshaped variance and mean calculations within each group.

* **Adaptive Layer Normalization (AdaLN)**
  Let $X$ be the input sequence and $c$ be a conditioning vector (e.g., timestep or class embedding).
  Instead of learning static parameters $\gamma$ and $\beta$, AdaLN generates them dynamically via linear projections:
  $\gamma = c W_\gamma + b_\gamma$
  $\beta = c W_\beta + b_\beta$
  Standard normalization is applied:
  $\mu = \frac{1}{d} \sum X_i, \quad \sigma^2 = \frac{1}{d} \sum (X_i - \mu)^2$
  $\hat{X} = \frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}}$
  $Output = \gamma \odot \hat{X} + \beta$
  During backpropagation, gradients route from the output through $\gamma$ and $\beta$ into the conditioning network weights $W_\gamma$ and $W_\beta$.

* **Reversible Residual Networks (RevNet)**
  A RevNet block operates on a partitioned state $(x_1, x_2)$ allowing constant-memory backpropagation.
  The forward process is given by:
  $y_1 = x_1 + F(x_2)$
  $y_2 = x_2 + G(y_1)$
  The exact inverse is computed during the backward pass:
  $x_2 = y_2 - G(y_1)$
  $x_1 = y_1 - F(x_2)$
  This permits exact gradient calculation for intermediate steps without caching activations (excluding the active layer).

* **Conditional Flow Matching (CFM)**
  Let $x_0 \sim \mathcal{N}(0, I)$ be the base distribution and $x_1 \sim p_{data}$ be the data distribution.
  The flow path is constructed as a straight line:
  $x_t = (1 - t) x_0 + t x_1$ for $t \in [0, 1]$.
  The target vector field is constant for a given sample pair:
  $u_t(x_t|x_1) = x_1 - x_0$
  The network $v_\theta(x_t, t)$ learns to match this vector field using the MSE objective:
  $\mathcal{L}_{CFM} = \mathbb{E}_{t \sim U(0,1), x_0, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$

* **Denoising Diffusion Probabilistic Model (DDPM)**
  Let $x_0$ be the original data. The forward process adds noise over $T$ steps according to a variance schedule $\beta_1, \dots, \beta_T$.
  $q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$
  Using the reparameterization trick, we can sample $x_t$ directly from $x_0$:
  $\alpha_t = 1 - \beta_t$
  $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
  $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$
  The reverse process learns a model $\epsilon_\theta(x_t, t)$ to predict the noise $\epsilon$ added to $x_0$.
  The training objective simplifies to:
  $\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]$

* **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
  Let $X$ be the high-dimensional data points. We compute pairwise affinities:
  $p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$
  The joint probabilities are symmetric:
  $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$
  Let $Y$ be the low-dimensional map points. We use a Student-t distribution with one degree of freedom to measure similarities in the embedded space, alleviating the crowding problem:
  $q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}$
  The objective is to minimize the Kullback-Leibler divergence:
  $C = KL(P || Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$
  The gradient is computed as:
  $\frac{\partial C}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + ||y_i - y_j||^2)^{-1}$
  During optimization, gradient descent with momentum and early exaggeration (multiplying $p_{ij}$ by a constant early in training) is used to find the optimal $Y$.

* **Linear Attention**
  Standard attention computes $O = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$, which has $O(N^2)$ complexity where $N$ is sequence length.
  Linear Attention bypasses this by using a kernel feature map $\phi(x)$ (e.g., $\text{ELU}(x) + 1$) to ensure non-negativity.
  $\phi_Q = \phi(Q)$
  $\phi_K = \phi(K)$
  Instead of computing the $N \times N$ attention matrix, we use associativity:
  $Output_i = \frac{\sum_{j} (\phi_{Q_i} \cdot \phi_{K_j}) V_j}{\sum_{j} (\phi_{Q_i} \cdot \phi_{K_j})}$
  Vectorized form:
  $Num = \phi_Q (\phi_K^T V)$  # Complexity $O(N \cdot d_k^2)$
  $Denom = \phi_Q \sum_{j} \phi_{K_j}^T$
  $Output = \frac{Num}{Denom}$
  This effectively reduces sequence dimension complexity from $O(N^2)$ to $O(N)$.

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

* **Vector Quantized Variational Autoencoder (VQ-VAE)**
  Let $x$ be the input, encoder $E(x) = z_e$, decoder $D(z_q)$, and an embedding space $e \in \mathbb{R}^{K \times D}$ where $K$ is the size of the discrete latent space.
  Forward pass (Vector Quantization):
  $z_q = e_k \quad \text{where} \quad k = \text{argmin}_j \|z_e - e_j\|_2$
  Backward pass (Straight-Through Estimator):
  $\nabla_{z_e} L \approx \nabla_{z_q} L$ (Gradients are copied directly from decoder input to encoder output).
  Loss function components:
  1. Reconstruction Loss: $\mathcal{L}_{recon} = -\log p(x|z_q)$
  2. Codebook Loss (updates embeddings): $\mathcal{L}_{codebook} = \| \text{sg}[z_e] - e \|_2^2$ (where $\text{sg}$ is stop-gradient)
  3. Commitment Loss (keeps encoder output close to embeddings): $\mathcal{L}_{commit} = \beta \| z_e - \text{sg}[e] \|_2^2$
  Total Loss: $\mathcal{L} = \mathcal{L}_{recon} + \mathcal{L}_{codebook} + \mathcal{L}_{commit}$

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

* **Sparse Autoencoder (SAE)**
  Let $x$ be the input data. The encoder maps it to an overcomplete hidden representation $z$:
  $z = \text{ReLU}(x W_e + b_{enc})$
  The decoder reconstructs the input:
  $\hat{x} = z W_d + b_{dec}$
  The loss combines Mean Squared Error for reconstruction and an L1 penalty for sparsity:
  $\mathcal{L} = \frac{1}{B \cdot D} \sum (x - \hat{x})^2 + \lambda \frac{1}{B} \sum |z|$
  Backpropagation correctly routes gradients for both the reconstruction error and the L1 penalty (using the sign of $z$) back through the network.

* **MLP-Mixer**
  Let $X \in \mathbb{R}^{S \times C}$ be the input sequence matrix where $S$ is sequence length and $C$ is channels.
  The MLP-Mixer block applies two operations with skip connections:
  1. Token Mixing (operates on columns):
  $U = X + \text{MLP}_{token}(\text{LayerNorm}(X)^T)^T$
  2. Channel Mixing (operates on rows):
  $Y = U + \text{MLP}_{channel}(\text{LayerNorm}(U))$
  During backpropagation, gradients route correctly through both transposed dimensions for the token mixing MLP and standard dimensions for the channel mixing MLP.

* **gMLP (Gated MLP)**
  Let $X \in \mathbb{R}^{N \times d}$ be the input matrix.
  The gMLP block utilizes a Spatial Gating Unit (SGU) to model spatial dependencies:
  $Z = X U$ (Linear projection where $U \in \mathbb{R}^{d \times 2d_{hidden}}$)
  $Z_{act} = \text{ReLU}(Z)$
  Split $Z_{act}$ into $Z_1$ and $Z_2$ along the hidden dimension.
  Spatial projection across sequence length $N$:
  $\tilde{Z}_2 = W Z_2 + b$ (where $W \in \mathbb{R}^{N \times N}$)
  Gating mechanism:
  $S = Z_1 \odot \tilde{Z}_2$
  $Y = S V$ (Output projection where $V \in \mathbb{R}^{d_{hidden} \times d}$)
  During backpropagation, gradients effectively flow through the element-wise gating operation and the sequence-wise spatial projection using tensor contractions.

* **Bayesian Neural Network (Bayes by Backprop)**
  Let $w$ be the weights of a neural network modeled as a probability distribution $q(w|\theta)$, where $\theta = (\mu, \rho)$.
  To ensure strictly positive variance, we parameterize standard deviation as $\sigma = \text{softplus}(\rho) = \log(1 + \exp(\rho))$.
  The Reparameterization Trick is used to sample weights differentiably:
  $w = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$
  The objective is to minimize the negative Evidence Lower Bound (ELBO):
  $\mathcal{L}(\theta) = \text{KL}[q(w|\theta) || P(w)] - \mathbb{E}_{q(w|\theta)}[\log P(\mathcal{D}|w)]$
  Assuming a standard normal prior $P(w) = \mathcal{N}(0, I)$, the analytical KL divergence is:
  $\text{KL} = 0.5 \sum (\mu^2 + \sigma^2 - 1 - 2\log(\sigma))$
  During backpropagation, gradients of the ELBO with respect to $\mu$ and $\rho$ are computed directly.

* **Restricted Boltzmann Machine (RBM)**
  An RBM is a bipartite generative model with visible units $v$ and hidden units $h$.
  The joint distribution is defined by the energy function:
  $E(v, h) = -v^T W h - b_v^T v - b_h^T h$
  The conditional probabilities for Gibbs sampling are given by:
  $P(h_j = 1 | v) = \sigma(W_{\cdot j}^T v + b_{h, j})$
  $P(v_i = 1 | h) = \sigma(W_{i \cdot} h + b_{v, i})$
  During training, weights are updated using Contrastive Divergence (CD-k), typically CD-1:
  $\Delta W \propto \langle v h^T \rangle_{data} - \langle v h^T \rangle_{recon}$

* **End-To-End Memory Network (MemN2N)**
  Let $x_i$ be facts in memory and $q$ be the query.
  Input memory representation: $m_i = A x_i$
  Output memory representation: $c_i = C x_i$
  Query embedding: $u = B q$
  Match query to memory to compute attention probabilities:
  $p_i = \text{softmax}(u^T m_i)$
  Compute output vector by summing over output memories weighted by probabilities:
  $o = \sum p_i c_i$
  Predict final answer based on combined output and query:
  $\hat{a} = \text{softmax}(W(o + u))$
  Backpropagation properly routes through $W$, the combination sum, the probability calculation, and into embeddings $A$, $C$, and $B$.

* **REINFORCE (Policy Gradient)**
  Let $\pi_\theta(a|s)$ be a parameterized stochastic policy.
  The goal is to maximize the expected return:
  $J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^T \gamma^t R_t \right]$
  The policy gradient theorem gives the gradient:
  $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) G_t \right]$
  where $G_t = \sum_{k=t}^T \gamma^{k-t} R_k$ is the return from step $t$.
  To reduce variance, a baseline $b(s_t)$ can be subtracted from the return:
  $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) (G_t^{(i)} - b)$
  During training, manual backpropagation computes this gradient and updates weights via gradient ascent to increase the likelihood of actions leading to higher returns.

* **Deep Q-Network (DQN)**
  Let $Q(s, a; \theta)$ be the Q-network with parameters $\theta$ and $Q(s, a; \theta^-)$ be the target network with parameters $\theta^-$.
  Experiences $(s_t, a_t, r_t, s_{t+1}, done)$ are stored in a replay buffer.
  A mini-batch is sampled uniformly from the buffer.
  The Temporal Difference (TD) target for step $i$ is calculated as:
  $y_i = \begin{cases} r_i & \text{if done} \\ r_i + \gamma \max_{a'} Q(s_{i+1}, a'; \theta^-) & \text{otherwise} \end{cases}$
  The loss function is the Mean Squared Error between the predicted Q-value and the target:
  $L(\theta) = \frac{1}{N} \sum_{i} (y_i - Q(s_i, a_i; \theta))^2$
  During training, the target network parameters $\theta^-$ are updated to match $\theta$ periodically. Backpropagation routes gradients back through the Q-network solely for the chosen actions.

* **Actor-Critic (RL)**
  Let $\pi_\theta(a|s)$ be the policy (Actor) and $V_w(s)$ be the value function (Critic).
  The Critic is trained to minimize the Temporal Difference (TD) error:
  $\delta_t = R_t + \gamma V_w(s_{t+1}) - V_w(s_t)$
  The Critic loss is $\mathcal{L}_V = \frac{1}{2} \delta_t^2$, leading to the gradient descent update:
  $w \leftarrow w + \alpha_w \delta_t \nabla_w V_w(s_t)$
  The Actor uses the TD error as a baseline-adjusted return for policy gradient ascent:
  $\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(a_t | s_t) \delta_t$
  During training with a shared hidden layer, gradients from both the actor's log-probability scaling and the critic's TD error minimization flow backwards through the shared representation.

* **Generative Flow Network (GFlowNet)**
  Let $s$ be a state and $a$ be an action. The forward policy is $P_F(s_{t+1}|s_t)$.
  Let $P_B(s_t|s_{t+1})$ be the backward policy (often uniform or fixed).
  A trajectory $\tau = (s_0, s_1, \dots, s_n)$ has forward probability $P_F(\tau) = \prod P_F(s_{t+1}|s_t)$ and backward probability $P_B(\tau) = \prod P_B(s_t|s_{t+1})$.
  The Trajectory Balance (TB) loss introduces a learnable parameter $Z$ (partition function):
  $\mathcal{L}_{TB}(\tau) = (\log Z + \sum \log P_F(s_{t+1}|s_t) - \log R(x) - \sum \log P_B(s_t|s_{t+1}))^2$
  where $R(x)$ is the reward of the terminal state $x = s_n$.
  During backpropagation, gradients route back through the TB loss to update both the policy weights predicting $P_F$ and the $\log Z$ parameter.

* **Generative Flow Network (GFlowNet)**
  Let $s$ be a state and $a$ be an action. The forward policy is $P_F(s_{t+1}|s_t)$.
  Let $P_B(s_t|s_{t+1})$ be the backward policy (often uniform or fixed).
  A trajectory $\tau = (s_0, s_1, \dots, s_n)$ has forward probability $P_F(\tau) = \prod P_F(s_{t+1}|s_t)$ and backward probability $P_B(\tau) = \prod P_B(s_t|s_{t+1})$.
  The Trajectory Balance (TB) loss introduces a learnable parameter $Z$ (partition function):
  $\mathcal{L}_{TB}(\tau) = (\log Z + \sum \log P_F(s_{t+1}|s_t) - \log R(x) - \sum \log P_B(s_t|s_{t+1}))^2$
  where $R(x)$ is the reward of the terminal state $x = s_n$.
  During backpropagation, gradients route back through the TB loss to update both the policy weights predicting $P_F$ and the $\log Z$ parameter.

## Experimental Summaries
* **Experiment `0106_train_gflownet_component` (Success):** Implemented and evaluated a Generative Flow Network (GFlowNet) using pure NumPy. Successfully learned to generate trajectories with probabilities proportional to terminal rewards by optimizing the Trajectory Balance loss via manual backpropagation, accurately learning the log partition function.
* **Experiment `0077_train_skipgram_component` (Success):** Implemented and evaluated a Skip-Gram component using pure NumPy. Successfully learned semantic word clusters by optimizing context and target embeddings using Negative Sampling and manual backpropagation.
* **Experiment `0078_train_cbow_component` (Success):** Implemented and evaluated a Continuous Bag of Words (CBOW) component using pure NumPy. Successfully learned semantic word clusters by predicting a target word from the average of its context word embeddings using manual backpropagation.
* **Experiment `0064_train_dqn_component` (Success):** Implemented and evaluated a Deep Q-Network (DQN) using pure NumPy. Successfully verified the mathematical formulation of Q-learning stabilized by experience replay and target networks, learning to navigate a simple grid environment via manual backpropagation.
* **Experiment `0063_train_ppo_component` (Success):** Implemented and verified Proximal Policy Optimization (PPO) using pure NumPy. Successfully learned an optimal policy via a clipped surrogate objective with multiple epochs per rollout, validating gradient-based optimization of policy ratios.
* **Experiment `0062_train_actor_critic_component` (Success):** Implemented and evaluated an Actor-Critic architecture using pure NumPy. Successfully verified the mathematical formulation of simultaneous policy gradient ascent and value function regression utilizing Temporal Difference (TD) errors, allowing stable online learning through manual backpropagation on a shared hidden layer.
* **Experiment `0061_train_reinforce_component` (Success):** Implemented and evaluated the REINFORCE policy gradient algorithm using pure NumPy. Successfully verified the mathematical formulation of maximizing expected returns via gradient ascent on the log probability of actions scaled by standardized returns, learning to navigate a simple grid environment via manual backpropagation.
* **Experiment `0060_train_memory_network_component` (Success):** Implemented and evaluated an End-To-End Memory Network (MemN2N) component using pure NumPy. Successfully learned to route reasoning paths by applying soft attention over stored facts to correctly answer queries, verifying the complex routing of gradients through multiple memory and query embedding matrices.
* **Experiment `0059_train_gat_component` (Success):** Implemented and trained a Graph Attention Network (GAT) using pure NumPy. Successfully verified the mathematical formulation of masked self-attention over graphs, achieving successful convergence on a node classification task while manually backpropagating through the attention and feature aggregation steps.
* **Experiment `0053_train_gcn_component` (Success):** Implemented and trained a Graph Convolutional Network (GCN) using pure NumPy. Successfully verified the mathematical formulation of graph convolutions by propagating information across nodes via a normalized adjacency matrix and manual backpropagation, achieving high accuracy on a synthetic graph dataset.
* **Experiment `0052_train_gan_component` (Success):** Implemented and trained a Generative Adversarial Network (GAN) using pure NumPy. Successfully verified the adversarial minimax mathematical formulation by co-training a Generator to match a 1D Gaussian distribution and a Discriminator to distinguish real from fake samples, utilizing manual backpropagation.
* **Experiment `0049_train_gmlp_component` (Success):** Implemented and trained a gMLP (Gated MLP) component using pure NumPy. Successfully verified its ability to model spatial/sequential dependencies without attention mechanisms by employing a Spatial Gating Unit (SGU) that combines element-wise multiplication with sequence-wise linear projection, validating its manual backpropagation across spatial and channel dimensions.
* **Experiment `0048_train_mlpmixer_component` (Success):** Implemented and trained an MLP-Mixer block using pure NumPy. Successfully verified the sequence-learning capabilities of alternating Token-mixing MLPs and Channel-mixing MLPs, validating its mathematical soundness as an alternative to self-attention via complex manual backpropagation through transposed sequences.
* **Experiment `0046_train_groupnorm_component` (Success):** Implemented and trained a Group Normalization component using pure NumPy. Successfully learned to scale and shift normalized inputs after dividing channels into groups, validating the mathematical soundness of reshaping features for grouped statistics and manually backpropagating gradients through the grouped groups.
* **Experiment `0045_train_batchnorm_component` (Success):** Implemented and trained a Batch Normalization component using pure NumPy. Successfully learned to scale and shift normalized inputs across the batch dimension, validating the mathematical soundness of normalization parameter updates and the complex manual backpropagation through batch statistics.
* **Experiment `0043_train_adaln_component` (Success):** Implemented and trained an Adaptive Layer Normalization (AdaLN) component using pure NumPy. Successfully learned to dynamically predict scale and shift parameters ($\gamma$, $\beta$) from a conditioning input via linear projections, confirming the mathematical soundness and gradient flow through the conditional formulation.
* **Experiment `0042_train_revnet_component` (Success):** Implemented and trained a Reversible Residual Network block using pure NumPy. Successfully proved the mathematical formulation allowing exact input reconstruction during the backward pass ($O(1)$ intermediate activation storage), verifying that manual backpropagation through the reconstructed states effectively reduces loss.
* **Experiment `0041_train_grokking_component` (Success):** Implemented and trained a 2-layer MLP on modular addition using pure NumPy to study Grokking. Successfully observed the rapid memorization phase (100% train accuracy, ~0% test accuracy), validating the initial training dynamics on algorithmic datasets prior to delayed generalization.
* **Experiment `0039_train_sae_component` (Success):** Implemented and trained a Sparse Autoencoder (SAE) using pure NumPy. Successfully learned to reconstruct input data while projecting it into a sparse, overcomplete latent representation via an L1 penalty, validating the mathematical formulation and manual backpropagation for mechanistic interpretability.
* **Experiment `0040_train_vqvae_component` (Success):** Implemented and trained a Vector Quantized Variational Autoencoder (VQ-VAE) using pure NumPy. Successfully learned to reconstruct input data using discrete latent representations via a codebook. Verified that the Straight-Through Estimator (STE) correctly routes gradients back to the encoder, enabling the learning of categorical latent variables.
* **Experiment `0038_train_flow_matching_component` (Success):** Implemented and trained a Continuous Normalizing Flow using Conditional Flow Matching (CFM) on a synthetic 2D dataset using pure NumPy. Successfully learned to predict the constant vector field mapping the base distribution to the data distribution, validating the mathematical formulation of the straight-line probability flow ODE and its manual backpropagation.
* **Experiment `0037_train_ddpm_component` (Success):** Implemented and trained a Denoising Diffusion Probabilistic Model (DDPM) on a synthetic 2D dataset using pure NumPy. Successfully learned to predict the added noise in the reverse process using a simple MLP and manual backpropagation, validating the mathematical formulation of the diffusion steps and simplified MSE objective.
* **Experiment `0014_train_swiglu_component` (Success):** Implemented and trained a Swish-Gated Linear Unit (SwiGLU) component using pure NumPy. Successfully verified the mathematical formulation of forward and backward passes for the SwiGLU activation, testing its ability to learn non-linear reasoning boundaries.
* **Experiment `0018_train_lora_component` (Success):** Implemented and trained a Low-Rank Adaptation (LoRA) component using pure NumPy. Successfully verified parameter-efficient fine-tuning principles by learning low-rank adapter matrices while keeping the base weight matrix frozen via manual backpropagation.
* **Experiment `0019_train_evaluation_metrics_component` (Success):** Formulated, implemented, and tested core evaluation metrics (Softmax, Cross-Entropy Loss, Perplexity, Accuracy) using pure NumPy. Successfully verified the mathematical stability and proper gradient flow via manual backpropagation during combined Softmax-Cross Entropy operation.
* **Experiment `0055_train_esn_component` (Success):** Implemented and verified an Echo State Network (ESN) mathematically using pure NumPy. Successfully verified Reservoir Computing principles by predicting a chaotic time series using a fixed random reservoir and a linear readout layer trained via Ridge Regression.
* **Experiment `0067_train_ewc_component` (Success):** Implemented and evaluated Elastic Weight Consolidation (EWC) using pure NumPy. Successfully computed the Fisher Information Matrix and applied an L2 penalty weighted by Fisher information, effectively reducing catastrophic forgetting on a sequential linear regression task.
* **Experiment `0036_train_retention_component` (Success):** Implemented and trained a Retention Mechanism (from RetNet) using pure NumPy. Successfully verified the mathematical soundness of both its parallel attention-like training formulation and its $O(1)$ recurrent inference formulation without KV-caching, validating its forward pass and manual backpropagation.

* **Experiment `0035_train_linear_attention_component` (Success):** Implemented and trained a Linear Attention component using pure NumPy. Successfully learned relationships in a synthetic sequence dataset while bypassing the $O(N^2)$ softmax attention matrix computation, validating the mathematical formulation of the $\phi(x) = \text{ELU}(x) + 1$ kernel trick and its manual backpropagation.

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
* **Experiment `0044_train_knowledge_distillation_component` (Success):** Implemented and evaluated Knowledge Distillation (KD) using pure NumPy. Successfully transferred dark knowledge from a larger teacher model to a smaller student model by minimizing both the KL Divergence of temperature-scaled soft targets and standard Cross-Entropy for hard labels, validating its manual backpropagation.
* **Experiment `0047_train_highway_component` (Success):** Implemented and trained a Highway Network component using pure NumPy. Successfully verified the gating mechanism (transform and carry gates) allowing representations to conditionally pass through unhindered, verifying its computational structure and manual backpropagation as a viable method to mitigate the vanishing gradient problem in deep architectures.
* **Experiment `0050_train_hypernetwork_component` (Success):** Implemented and trained a Hypernetwork component using pure NumPy. Successfully learned to dynamically generate weights and biases for a primary network conditioned on contextual inputs, verifying the structural tensor contractions (`einsum`) and manual backpropagation of gradients through the generated parameters back to the hypernetwork.
* **Experiment `0051_train_hopfield_component` (Partial Success):** Implemented and evaluated a Hopfield Network component using pure NumPy. Successfully learned a symmetric weight matrix via Hebbian learning and verified the mathematical soundness of energy minimization during asynchronous updates to retrieve stored patterns.
* **Experiment `0054_train_rbm_component` (Success):** Implemented and evaluated a Restricted Boltzmann Machine (RBM) component using pure NumPy. Successfully learned to model the distribution of a synthetic binary dataset and verified the mathematical soundness of the energy-based model and manual parameter updates using Contrastive Divergence (CD-1).
* **Experiment `0056_train_bnn_component` (Success):** Implemented and evaluated a Bayesian Neural Network (BNN) component using pure NumPy. Successfully verified the mathematical soundness of the Bayes by Backprop algorithm, learning parameter distributions via the reparameterization trick and optimizing the Evidence Lower Bound (ELBO) on a non-linear reasoning task.
* **Experiment `0057_train_neural_ode_component` (Success):** Implemented and evaluated a Neural ODE component using pure NumPy. Successfully verified continuous-depth modeling by evolving hidden states via Euler numerical integration and manual backpropagation on a non-linear dataset.
* **Experiment `0058_train_snn_component` (Success):** Implemented and evaluated a Spiking Neural Network (SNN) component with Leaky Integrate-and-Fire neurons using pure NumPy. Successfully verified surrogate gradient backpropagation to learn non-linear decision boundaries through discrete spiking time steps.
* **Experiment `0064_train_dqn_component` (Success):** Implemented and evaluated a Deep Q-Network (DQN) component using pure NumPy. Successfully verified the mathematical formulation of Q-values calculation, experience replay, target networks, and manual backpropagation for deep reinforcement learning.
* **Experiment `0065_train_maml_component` (Success):** Implemented and evaluated a First-Order Model-Agnostic Meta-Learning (MAML) component using pure NumPy. Successfully verified the mathematical formulation of inner loop task adaptation and outer loop meta-initialization updates via manual backpropagation.
* **Experiment `0066_train_tcn_component` (Success):** Implemented and evaluated a Temporal Convolutional Network (TCN) component using pure NumPy. Successfully verified causal dilated convolutions and residual connections for sequence modeling, calculating gradients via manual backpropagation.

* **Experiment `0068_train_ctrnn_component` (Success):** Implemented and evaluated a Continuous-Time Recurrent Neural Network (CTRNN) component using pure NumPy. Successfully learned a continuous moving average dynamic over sequential data, confirming the mathematical formulation of continuous state evolution using Euler integration and Backpropagation Through Time (BPTT).
* **Experiment `0069_train_feedback_alignment_component` (Success):** Implemented and evaluated a Random Feedback Alignment (FA) component using pure NumPy. Successfully learned non-linear boundaries by propagating errors backward through fixed random weight matrices instead of symmetric transposed forward weights, verifying its computational viability as a biologically plausible learning mechanism.
* **Experiment `0070_train_dfa_component` (Success):** Implemented and evaluated a Direct Feedback Alignment (DFA) component using pure NumPy. Successfully learned non-linear boundaries by propagating the output error directly to each hidden layer via fixed random matrices, validating the feasibility of parallel weight updates across layers.

* **Experiment `0071_train_elm_component` (Success):** Implemented and evaluated an Extreme Learning Machine (ELM) component using pure NumPy. Successfully learned non-linear boundaries through rapid one-shot analytical learning by solving for output weights via the Moore-Penrose pseudoinverse of random hidden features, bypassing iterative backpropagation entirely.

* **Experiment `0072_train_rbf_component` (Success):** Implemented and evaluated a Radial Basis Function (RBF) Network component using pure NumPy. Successfully learned non-linear boundaries using localized Gaussian basis functions, validating the optimization of centroids, widths, and output weights via manual backpropagation.

* **Experiment `0073_train_som_component` (Success):** Implemented and evaluated a Self-Organizing Map (SOM) component using pure NumPy. Successfully learned unsupervised topological representation of data onto a 2D grid using competitive learning.

* **Experiment `0074_train_nca_component` (Success):** Implemented and evaluated a Neural Cellular Automata (NCA) component using pure NumPy. Successfully learned to iteratively grow a target pattern from a single seed pixel, confirming self-organizing pattern generation.

* **Experiment `0075_train_ltc_component` (Success):** Implemented and evaluated a Liquid Time-Constant (LTC) Network component using pure NumPy. Successfully verified dynamically adapting continuous-time continuous-depth dynamics by varying the time constant based on input, optimizing via manual backpropagation.
* **Experiment `0076_train_ntm_component` (Success):** Implemented and evaluated a Neural Turing Machine (NTM) component using pure NumPy. Successfully verified the mathematical formulation of content-based memory addressing and differentiable read/write operations via manual backpropagation.

* **Experiment `0081_train_fnet_component` (Success):** Implemented and evaluated an FNet block component using pure NumPy. Successfully learned sequence relationships by replacing self-attention with a parameter-free 2D Fourier Transform for mixing over sequence and hidden dimensions.

## Open Questions & Hypotheses

1. *(e.g., "Does scaling the depth of the network linearly correlate with reasoning capability on dataset Y?")*

### Experiment 0080: Perceiver Bottleneck
- **Hypothesis:** We can reduce the $O(N^2)$ complexity of standard self-attention to $O(N \cdot M)$ (where $M$ is the number of trainable latents and $N$ is sequence length) by using cross-attention where the latents act as queries and the input sequence acts as keys and values.
- **Action:** Implemented a Perceiver Bottleneck component in `train_perceiver_component.py` with manual backpropagation.
- **Outcome:** The model successfully converged (Final Loss: ~0.000283), learning to summarize a sequence into a fixed-size latent representation.
- **Next Steps:** Consider applying this mechanism to multimodal inputs or very long sequences to exploit the reduced complexity.

### Experiment 0081: FNet Block
- **Hypothesis:** We can replace the computationally expensive self-attention mechanism with a parameter-free 2D Fast Fourier Transform (FFT) along the sequence and hidden dimensions, maintaining the ability to mix tokens effectively while significantly improving efficiency.
- **Action:** Implemented an FNet Block component in `train_fnet_component.py` using NumPy`s FFT and evaluated it on a sequence inversion task with manual backpropagation.
- **Outcome:** The model successfully converged (Final Loss: ~0.258), learning sequence relationships without attention parameters.
- **Next Steps:** Consider integrating the FNet block into larger encoder structures to compare performance against standard Transformer blocks on more complex tasks.

### Experiment 0082: Predictive Coding Network (PCN)
- **Hypothesis:** We can learn complex non-linear representations using a biologically plausible local learning rule instead of global backpropagation, by performing iterative inference to minimize local prediction errors and updating weights based on these settled states.
- **Action:** Implemented a PCN component in `train_pcn_component.py` using NumPy, tested on a Sine wave regression dataset.
- **Outcome:** The model successfully converged (Final MSE: ~0.0009), demonstrating that local Hebbian-like updates on prediction errors are sufficient for learning.
- **Next Steps:** Investigate scaling this local learning rule to deeper architectures or comparing its sample efficiency directly against equivalent models trained with standard backpropagation.

### Experiment 0083: Capsule Network (Dynamic Routing)
- **Hypothesis:** We can preserve hierarchical spatial relationships and part-whole representations by replacing scalar neurons with vector capsules and replacing max-pooling with dynamic routing by agreement.
- **Action:** Implemented a Capsule Network component in `train_capsule_network_component.py` demonstrating dynamic routing between primary and routing capsules using pure NumPy.
- **Outcome:** The dynamic routing algorithm successfully converged, routing inputs to the correct higher-level capsules based on agreement.
- **Next Steps:** Investigate integrating capsule layers with convolutional front-ends to evaluate on structured visual reasoning tasks.

### Experiment 0084: Deep Sets (Permutation Invariant Networks)
- **Hypothesis:** We can process unordered sets by applying an independent transformation to each element and aggregating them with a symmetric function (e.g., sum pooling).
- **Action:** Implemented a Deep Sets component in `train_deepsets_component.py` using pure NumPy to classify sets based on sum thresholding.
- **Outcome:** The model successfully learned permutation-invariant features and converged on the set classification task.
- **Next Steps:** Consider exploring point cloud processing or multi-agent environments using set-based representations.

### Experiment 0085: Spectral Normalization
- **Hypothesis:** We can enforce Lipschitz continuity in linear layers without computationally expensive regularizations by using power iterations to divide weight matrices by their largest singular value.
- **Action:** Implemented a Spectral Normalization component in `train_spectral_normalization_component.py` using NumPy.
- **Outcome:** The network converged successfully on a binary classification task.
- **Next Steps:** Evaluate its impact on stabilizing Generative Adversarial Networks (GANs).

### Experiment 0086: Weight Normalization
- **Hypothesis:** Decoupling the length of weight vectors from their direction accelerates convergence and is suitable for tasks where batch statistics are unavailable or noisy.
- **Action:** Implemented a Weight Normalization component in `train_weight_normalization_component.py` using NumPy.
- **Outcome:** The network converged successfully and rapidly on a binary classification task.
- **Next Steps:** Compare convergence speed directly against Batch Normalization and RMSNorm in deeper architectures.

### Experiment 0087: Energy-Based Model (EBM)
- **Hypothesis:** We can learn an implicit probability distribution by parameterizing an energy function mapping inputs to scalars, and using gradient-based Langevin dynamics to sample from it.
- **Action:** Implemented an Energy-Based Model in `train_ebm_component.py` using NumPy.
- **Outcome:** The network successfully converged, lowering the energy of data samples relative to noise samples drawn via Langevin dynamics.
- **Next Steps:** Explore applying EBMs to continuous control and reinforcement learning settings.

### Experiment 0088: Decoupled Neural Interfaces (DNI)
- **Hypothesis:** We can decouple layers during backpropagation by using auxiliary networks to predict synthetic gradients, enabling asynchronous training updates.
- **Action:** Implemented Decoupled Neural Interfaces in `train_dni_component.py` using NumPy.
- **Outcome:** The network converged successfully using synthetic gradients on local layer updates.
- **Next Steps:** Evaluate the scaling characteristics of synthetic gradients on deeper architectures or recurrent models where BPTT locking is severe.

### Experiment 0090: Mixture Density Network (MDN)
- **Hypothesis:** We can model multi-modal conditional probability distributions $p(y|x)$ by outputting the parameters (mixing coefficients, means, and variances) of a Gaussian Mixture Model from a neural network.
- **Action:** Implemented a Mixture Density Network in `train_mdn_component.py` using NumPy.
- **Outcome:** The network successfully minimized the Negative Log-Likelihood, learning the mapping for an inverse kinematics toy problem where one input corresponds to multiple possible outputs.
- **Next Steps:** Explore MDNs for sequence modeling or advanced reinforcement learning environments requiring multi-modal continuous action spaces.

### Experiment 0091: RealNVP Normalizing Flow
- **Hypothesis:** We can model complex, high-dimensional probability distributions exactly by applying a series of invertible, learnable transformations (a normalizing flow) to a simple base distribution like a Gaussian.
- **Action:** Implemented a RealNVP Normalizing Flow in `train_realnvp_component.py` using NumPy.
- **Outcome:** The network successfully minimized the Negative Log-Likelihood on a 2D dataset, transforming it to match an isotropic Gaussian.
- **Next Steps:** Explore applying invertible architectures and normalizing flows to high-dimensional generation tasks (e.g., images) or as expressive prior distributions for VAEs.

### Experiment 0092: Neural Autoregressive Distribution Estimator (NADE)
- **Hypothesis:** We can model the exact joint probability distribution of high-dimensional binary data by factoring it into a product of conditional distributions, sharing weights across these conditionals for efficiency.
- **Action:** Implemented a Neural Autoregressive Distribution Estimator (NADE) in `train_nade_component.py` using NumPy.
- **Outcome:** The network successfully minimized the Negative Log-Likelihood on a synthetic sequential binary dataset, learning the conditional probabilities.
- **Next Steps:** Explore applying autoregressive models to more complex sequential generation tasks or combining them with other generative frameworks.

### Experiment 0094: Sinusoidal Representation Network (SIREN)
- **Hypothesis:** Neural networks with periodic activation functions (sine) and a specific initialization scheme can effectively model complex natural signals and their derivatives, overcoming the spectral bias of standard MLPs.
- **Action:** Implemented a Sinusoidal Representation Network (SIREN) in `train_siren_component.py` using pure NumPy.
- **Outcome:** The network successfully fit a high-frequency 1D composite signal ($y = \sin(10x) + \cos(25x)$) with very low error using manual backpropagation.
- **Next Steps:** Explore applying implicit neural representations (like SIREN) to higher-dimensional signals such as images or audio, or incorporating them into larger generative architectures.

### Experiment 0095: Conditional Neural Process (CNP)
- **Hypothesis:** We can model a distribution over functions and perform few-shot regression by encoding context points into a global representation and decoding it along with target inputs to predict output distributions.
- **Action:** Implemented a Conditional Neural Process in `train_cnp_component.py` using NumPy.
- **Outcome:** The network successfully minimized Negative Log-Likelihood on a family of sine waves, learning to infer the underlying function from a few context points.
- **Next Steps:** Explore applying CNPs or their attention-based variants (NPs, ANPs) to time-series forecasting or complex meta-learning tasks.

### Experiment 0096: SimSiam (Simple Siamese Networks)
- **Hypothesis:** We can learn meaningful representations without contrastive learning (negative pairs) or moving average momentum encoders by using a Siamese architecture with a predictor network on one branch and a stop-gradient operation on the other to prevent collapse.
- **Action:** Implemented SimSiam in `train_simsiam_component.py` using pure NumPy, including the encoder, predictor, and cosine similarity loss.
- **Outcome:** The network successfully minimized the negative cosine similarity between differently augmented views of the same data, confirming that representations were learned without collapsing into trivial constant solutions.
- **Next Steps:** Explore applying self-supervised non-contrastive methods to larger-scale image or sequence data to pretrain robust, generalizable encoders.

### Experiment 0098: Flow Matching
- **Hypothesis:** We can model a complex continuous target distribution by regressing a vector field that optimally transports a simple base distribution (Gaussian) to the target via straight probability paths, avoiding the need for exact invertibility constraints required by standard normalizing flows.
- **Action:** Implemented Flow Matching (Continuous Normalizing Flow) in `train_flow_matching_component.py` using pure NumPy with an Adam Optimizer.
- **Outcome:** The network successfully minimized the vector field matching loss. Euler integration of the learned vector field accurately transported base Gaussian samples into a target distribution forming a 2D ring of 8 Gaussians.
- **Next Steps:** Explore optimal transport variants of Flow Matching or integrate the flow into generation tasks in higher dimensions.

### Experiment 0099: Masked Autoencoder (MAE)
- **Hypothesis:** We can learn robust representations of data by masking a significant portion of the input and training a network to reconstruct the missing parts using an asymmetric encoder-decoder architecture.
- **Action:** Implemented Masked Autoencoder in `train_mae_component.py` using pure NumPy.
- **Outcome:** The network successfully minimized the MSE loss, effectively learning to reconstruct the masked portions of the input sequences using pure mathematical operations.
- **Next Steps:** Explore applying self-supervised masked modeling to larger-scale image (ViT) or sequence data to pretrain robust, generalizable encoders.

### Experiment 0101: Barlow Twins
- **Hypothesis:** We can learn meaningful representations without contrastive learning (negative pairs) or asymmetric momentum encoders by applying a redundancy-reduction objective that drives the cross-correlation matrix between representations of distorted sample versions towards the identity matrix.
- **Action:** Implemented a Barlow Twins component in `train_barlow_twins_component.py` using pure NumPy, including the manual backpropagation of the cross-correlation loss.
- **Outcome:** The network successfully minimized the objective, reducing redundancy across feature dimensions and avoiding representation collapse.
- **Next Steps:** Evaluate the sample efficiency and representation robustness of Barlow Twins against other non-contrastive methods like SimSiam on more complex datasets.

### Experiment 0102: Hebbian Learning (Oja's Rule)
- **Hypothesis:** We can extract the principal component of a dataset without backpropagation by using Oja's rule, a stable, biologically plausible Hebbian learning mechanism that balances correlation-based synaptic growth with weight decay.
- **Action:** Implemented Hebbian learning with Oja's rule in `train_hebbian_component.py` using pure NumPy.
- **Outcome:** The network successfully updated its weights to match the theoretical first principal component of the synthetic dataset, verifying the mathematical equivalence between Hebbian plasticity and PCA.
- **Next Steps:** Explore applying Generalized Hebbian Algorithms to extract multiple principal components or integrate Hebbian rules within competitive learning architectures.

### Independent Component Analysis (ICA)
- **Concept:** Unsupervised blind source separation for non-Gaussian signals.
- **Action:** Implemented FastICA with fixed-point iteration and negentropy maximization.
- **Outcome:** Successfully recovered original mixed signals with high correlation, validating its representation learning capabilities.

### Experiment 0104: Wasserstein GAN (WGAN)
- **Hypothesis:** We can mitigate training instability and mode collapse in standard GANs by optimizing the Earth Mover's (Wasserstein-1) distance, replacing the discriminator with a critic (removing the sigmoid activation) and enforcing a Lipschitz constraint via weight clipping.
- **Action:** Implemented a WGAN component in `train_wgan_component.py` using pure NumPy, including RMSProp optimization and manual backpropagation.
- **Outcome:** The generator successfully approximated the target 1D Gaussian distribution, demonstrating smoother convergence of the W-distance compared to Jensen-Shannon divergence.
- **Next Steps:** Explore more advanced Lipschitz enforcement mechanisms, such as WGAN with Gradient Penalty (WGAN-GP).

### Experiment 0105: Bootstrap Your Own Latent (BYOL)
- **Hypothesis:** We can learn self-supervised representations without contrastive negative pairs and without avoiding collapse via explicit regularizations by using two interacting networks (online and target), where the target network is updated via an exponential moving average (EMA) of the online network.
- **Action:** Implemented a BYOL component in `train_byol_component.py` using pure NumPy, including the manual backpropagation of the normalized L2 prediction loss.
- **Outcome:** The network successfully minimized the objective between the online network's prediction and target network's projection on augmented views, indicating successful representation learning without collapse.
- **Next Steps:** Compare BYOL's representation quality to contrastive (InfoNCE) and other non-contrastive methods (Barlow Twins, SimSiam).

### Experiment 0109: Orthogonal RNN
- **Concept**: Preserving gradient norms over long sequences by constraining the hidden weight matrix to be orthogonal.
- **Action**: Implemented an Orthogonal RNN component (`train_orthogonal_rnn_component.py`) utilizing the Cayley transform $W = (I - A)(I + A)^{-1}$ with a skew-symmetric matrix $A = V - V^T$ parameterized by unconstrained matrix $V$.
- **Outcome**: The component trained successfully on a sequential task, demonstrating stable gradients compared to standard RNNs.
- **Next Steps**: Compare with unitary RNNs or implement more advanced parameterization for recurrent architectures.

### Experiment 0110: Joint Embedding Predictive Architecture (JEPA)
- **Hypothesis:** We can learn semantic representations by predicting the representation of a target signal from a context signal and a condition/action, utilizing a stop-gradient EMA target encoder to prevent collapse.
- **Action:** Implemented a JEPA component in `train_jepa_component.py` using pure NumPy, including an online encoder, an EMA target encoder, and a predictor network.
- **Outcome:** The model successfully minimized the prediction loss without representation collapse.
- **Next Steps:** Explore applying predictive representation learning to video sequences or hierarchical abstract planning models.

## LMU Component Integration
- Implemented and successfully trained a Legendre Memory Unit (LMU).
- The LMU utilizes fixed continuous-time matrices (A and B) derived from Legendre polynomials to create a state space model resilient to vanishing gradients over long sequences.

### Experiment 0112: Difference Target Propagation (DTP)
- **Hypothesis:** We can assign credit in deep architectures without backpropagation by using local inverse models to propagate target activations.
- **Action:** Implemented a DTP component in `train_target_propagation_component.py` using pure NumPy, training a 3-layer network with local forward updates and backward target propagation.
- **Outcome:** The model successfully fit a non-linear continuous mapping task without propagating gradients through hidden layers.
- **Next Steps:** Compare DTP with other biologically plausible credit assignment methods like Predictive Coding or Feedback Alignment.

### Experiment 0113: Variational Information Bottleneck (VIB)
- **Hypothesis:** We can learn robust representations by constraining the mutual information between the input and the latent space, forcing the network to ignore noise and focus on predictive features.
- **Action:** Implemented a Deep Variational Information Bottleneck component (`train_vib_component.py`) mathematically in pure NumPy, using the reparameterization trick and ELBO optimization balancing classification accuracy with KL divergence from a standard normal prior.
- **Outcome:** The model successfully filtered out noise features and maintained high predictive accuracy on the target classification task, learning a compact latent representation.
- **Next Steps:** Consider exploring conditional or disentangled representation learning frameworks.

### Deep Deterministic Policy Gradient (DDPG) Component (`train_ddpg_component.py`)
- **Mathematical Basis**: DDPG uses an actor-critic architecture for continuous action spaces. The critic learns the Q-function using the Bellman equation, and the actor updates its deterministic policy in the direction of the gradient of the Q-function with respect to the action: $\nabla_{\theta^\mu} J \approx \mathbb{E} [\nabla_a Q(s, a|\theta^Q)|_{a=\mu(s)} \nabla_{\theta^\mu} \mu(s|\theta^\mu)]$.
- **Verification**: The component successfully learned to navigate a continuous 1D environment using manual backpropagation and Ornstein-Uhlenbeck noise for exploration.

### Experiment 0115: Lottery Ticket Hypothesis (IMP)
- **Hypothesis:** Dense, randomly-initialized, feed-forward networks contain subnetworks ("winning tickets") that - when trained in isolation - reach test accuracy comparable to the original network in a similar number of iterations.
- **Action:** Implemented Iterative Magnitude Pruning (IMP) in `train_lottery_ticket_component.py`. The script trains a dense MLP, prunes the lowest magnitude weights, rewinds the remaining weights to their original initialization, and retrains the sparse network.
- **Outcome:** The sparse subnetwork (at ~67% sparsity) successfully retrained to near original accuracy from its initial weights, validating the existence of winning tickets.
- **Next Steps:** Consider integrating IMP into larger, more complex components to observe if sparse, trainable subnetworks emerge consistently across architectures.

### Experiment 0116: Sharpness-Aware Minimization (SAM)
- **Hypothesis:** We can improve generalization by explicitly penalizing the sharpness of the loss landscape. By computing gradients at weight values perturbed in the direction of the local loss gradient, we optimize for a flat minima.
- **Action:** Implemented Sharpness-Aware Minimization in `train_sam_component.py` mathematically in pure NumPy, using a two-step forward-backward process to compute the perturbation $\epsilon$ and then the final update gradient.
- **Outcome:** The model successfully converged on the non-linear classification task, confirming the mathematical implementation of the adversarial weight perturbation and sharpness-aware update.
- **Next Steps:** Evaluate the empirical generalization benefits of SAM compared to standard optimizers on more complex tasks.

### Experiment 0117: Forward-Forward Algorithm
- **Component:** Forward-Forward Layer
- **Purpose:** Investigate mathematically justified alternatives to backpropagation.
- **Insights:** Successfully implemented a gradient-free (across layers) learning method by maximizing 'goodness' (sum of squared activations) for positive samples and minimizing it for negative samples locally within each layer, proving its viability for simple classification tasks without deep credit assignment paths.

### Experiment 0118: Differentiable Architecture Search (DARTS)
- **Hypothesis:** We can identify an optimal neural network architecture efficiently by relaxing the discrete search space to be continuous and jointly optimizing both architecture parameters and weights using gradient descent.
- **Action:** Implemented DARTS in `train_darts_component.py` purely mathematically, specifying candidate operations (Linear, ReLU, Sigmoid, Zero) and using a bi-level optimization scheme.
- **Outcome:** The model successfully converged and correctly assigned the highest probability weight to the true underlying operation (ReLU) generating the data, validating the continuous relaxation approach for architecture search.
- **Next Steps:** Consider integrating DARTS into larger modular networks to automatically search for optimal sub-components.

### Experiment 0119: Prototypical Networks (ProtoNet)
- **Hypothesis:** A neural network can learn to perform few-shot classification by mapping examples into a metric space where instances of a given class cluster around a single prototype representation (the mean of the support set embeddings).
- **Action:** Implemented ProtoNet in `train_protonet_component.py` mathematically in pure NumPy, using Euclidean distance to class prototypes and manual backpropagation on the episodic training loss.
- **Outcome:** The ProtoNet successfully learned a metric embedding to classify new query points, achieving high accuracy on the synthetic few-shot episodes.
- **Next Steps:** Consider testing the component on more complex datasets or comparing it with other meta-learning approaches such as MAML.

### Experiment 0120: Fast Gradient Sign Method (FGSM)
- **Hypothesis:** Neural networks are highly vulnerable to small perturbations aligned with the loss gradient. Adversarial training on these dynamically generated examples can improve robustness.
- **Action:** Implemented the FGSM attack and adversarial training in `train_fgsm_component.py` mathematically in pure NumPy, testing on an MLP with alternating clean and adversarial updates.
- **Outcome:** The robust model showed a significant improvement in accuracy on adversarial examples compared to a standard model, verifying the capability of adversarial training to induce robustness.
- **Next Steps:** Explore more advanced attacks like PGD or certified robustness methods.

### Experiment 0121: Sparsemax Component
- **Hypothesis:** We can compute exactly sparse probability distributions by using Euclidean projection onto the probability simplex instead of softmax, providing a useful mechanism for sparse attention or discrete latent selections.
- **Action:** Implemented Sparsemax mathematically in pure NumPy in `train_sparsemax_component.py`, calculating the sorting-based threshold and passing gradients correctly through the non-zero support set.
- **Outcome:** The model converged successfully and generated exact zeros in the output probabilities, validating the thresholding and masking logic in both forward and backward passes.
- **Next Steps:** Consider replacing Softmax with Sparsemax in Attention layers to evaluate sparse attention mechanisms.

### Experiment 0122: Contrastive Predictive Coding (CPC)
- **Hypothesis:** By predicting future representations in latent space autoregressively, the model can learn useful data representations without needing to reconstruct high-dimensional inputs.
- **Action:** Implemented CPC in `train_cpc_component.py` mathematically in pure NumPy, using an RNN context network and InfoNCE loss across a batch of sequences.
- **Outcome:** The model successfully converged and minimized the InfoNCE loss, effectively distinguishing true future latent states from negative samples.
- **Next Steps:** Evaluate the learned representations by training a linear classifier on top of them for downstream tasks.

### Experiment 0123: Continuous Hopfield Network
- **Hypothesis:** By generalizing the classic binary Hopfield network to continuous states and using an exponential interaction function (log-sum-exp energy), we can drastically increase memory capacity and bridge associative memory with self-attention.
- **Action:** Implemented the Continuous (Modern) Hopfield Network mathematically in pure NumPy in `train_continuous_hopfield_component.py`, updating continuous state vectors to minimize the log-sum-exp energy function.
- **Outcome:** The network successfully retrieved target continuous patterns from noisy initializations, systematically decreasing the energy function to convergence.
- **Next Steps:** Evaluate the connection of this component to Transformer self-attention layers in a unified architecture.

### Experiment 0124: Neural Arithmetic Logic Unit (NALU)
- **Hypothesis:** By combining an additive accumulator and a multiplicative path controlled by a learned gate, a neural network can learn arithmetic operations that generalize systematically to numerical values outside the training distribution.
- **Action:** Implemented NALU in `train_nalu_component.py` mathematically in pure NumPy, using manual backpropagation to pass gradients through both the linear and log-space paths.
- **Outcome:** The model successfully converged on a multiplicative task, demonstrating that the gate learned to route the signal through the log-space multiplicative path effectively.
- **Next Steps:** Evaluate the model's ability to extrapolate on numbers significantly larger than those seen during training compared to standard MLPs.

### Experiment 0125: Adaptive Computation Time (ACT)
- **Hypothesis:** By introducing a ponder cost and a differentiable halting mechanism, the network can learn to use fewer computation steps when possible, while retaining the capacity to process inputs more deeply if required by the task, fully supported by exact manual gradients.
- **Action:** Implemented ACT in `train_act_component.py` mathematically in pure NumPy, using manual backpropagation to pass gradients through the dynamic computation graph, including the ponder probabilities and weights.
- **Outcome:** The model successfully converged, demonstrating the balance between task performance and ponder cost.
- **Next Steps:** Evaluate the model's ability to extrapolate on tasks requiring more computation steps for out-of-distribution inputs.

### Experiment 0126: Monte Carlo Tree Search (MCTS)
- **Hypothesis:** By combining Monte Carlo Tree Search (MCTS) with a neural network evaluating policy and value, the model can iteratively improve its policy through self-play and search.
- **Action:** Implemented MCTS in `train_mcts_component.py` mathematically in pure NumPy, using manual backpropagation to update the policy and value networks.
- **Outcome:** The model successfully converged and learned to navigate the gridworld environment to the goal state.
- **Next Steps:** Explore applying MCTS to more complex reinforcement learning and planning tasks.

### Experiment 0127: Relational Network (RN)
- **Hypothesis:** By applying a multi-layer perceptron to all pairs of objects in an input set and summing the results, a network can explicitly learn relational properties between the objects invariant to their order.
- **Action:** Implemented a Relational Network in `train_relational_network_component.py` mathematically in pure NumPy, using manual backpropagation to route gradients correctly through the permutation-invariant sum operation to the pairwise function.
- **Outcome:** The model successfully converged on a relational task.
- **Next Steps:** Evaluate the model on more complex visual reasoning tasks using extracted object embeddings.

### Experiment 0128: Pointer Network Component
- **Hypothesis:** By modifying the attention mechanism to output probabilities directly over the input sequence rather than blending encoder states, a neural network can successfully learn to point to input elements, enabling it to solve algorithmic tasks like sorting.
- **Action:** Implemented a Pointer Network in `train_pointer_network_component.py` mathematically in pure NumPy, using manual backpropagation to route gradients correctly through the attention mechanisms pointing to encoder states.
- **Outcome:** The model successfully converged on a sequence sorting task, demonstrating that the attention weights effectively learned to point to the correct input element.
- **Next Steps:** Evaluate the Pointer Network on combinatorial optimization problems like the Traveling Salesperson Problem (TSP).

### Experiment 0129: Twin Delayed DDPG (TD3)
- **Hypothesis:** By employing clipped double Q-learning, delayed policy updates, and target policy smoothing, an actor-critic model can mitigate overestimation bias in continuous action spaces.
- **Action:** Implemented TD3 in `train_td3_component.py` mathematically in pure NumPy, using manual backpropagation and twin critics.
- **Outcome:** The model successfully converged on a continuous control task.
- **Next Steps:** Evaluate the model on more complex robotic control environments.

### Experiment 0130: Neural Radiance Field (NeRF)
- **Hypothesis:** By mapping spatial coordinates with positional encoding to density and color using an MLP, a model can represent and render continuous 3D scenes via volume rendering equations.
- **Action:** Implemented NeRF in `train_nerf_component.py` mathematically in pure NumPy, using manual backpropagation through the volumetric rendering discrete approximation.
- **Outcome:** The model successfully overfit a single ray to render a target color.
- **Next Steps:** Evaluate the model on multi-view 3D scene reconstruction tasks.

### Experiment 0131: Soft Actor-Critic (SAC)
- **Hypothesis:** By maximizing an objective that includes both expected return and entropy, an off-policy actor-critic algorithm can achieve robust, sample-efficient learning in continuous action spaces while avoiding premature convergence.
- **Action:** Implemented Soft Actor-Critic in `train_sac_component.py` mathematically in pure NumPy, using manual backpropagation, twin critics, and the reparameterization trick for a stochastic Gaussian policy.
- **Outcome:** The model successfully converged on a continuous control task.
- **Next Steps:** Evaluate the model on continuous control tasks with sparse rewards and compare with deterministic policy gradient methods.

### Experiment 0132: Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
- **Hypothesis:** By dynamically adapting a multivariate normal distribution based on the fitness of drawn samples, an evolutionary strategy can optimize complex, non-differentiable objectives more efficiently than standard random search.
- **Action:** Implemented CMA-ES in `train_cmaes_component.py` mathematically in pure NumPy, updating the mean, covariance matrix, and step size rules to optimize a test function.
- **Outcome:** The model successfully converged on the objective function, adapting its search distribution effectively.
- **Next Steps:** Apply CMA-ES to optimize hyper-parameters or network architectures where backpropagation is not applicable.

### Experiment 0133: Decision Transformer
- **Hypothesis:** By framing offline reinforcement learning as a sequence modeling problem over state, action, and return-to-go tokens, a transformer architecture with causal self-attention can learn an expert policy directly from offline trajectories.
- **Action:** Implemented a Decision Transformer in `train_decision_transformer_component.py` mathematically in pure NumPy, using causal self-attention and manual backpropagation on the action prediction MSE loss. Gradient clipping was applied to stabilize training.
- **Outcome:** The model successfully converged on the offline dataset, accurately reproducing the expert policy.
- **Next Steps:** Evaluate the model on more complex offline RL benchmarks with discrete actions.

### Experiment 0136: Gaussian Mixture Models (GMM)
- **Hypothesis:** By employing Expectation-Maximization (EM), a model can iteratively learn the parameters (means, covariances, and weights) of multiple Gaussian distributions to model complex data distributions and perform soft clustering.
- **Action:** Implemented GMM in `train_gmm_component.py` mathematically in pure NumPy, using EM to alternate between calculating component responsibilities (E-step) and updating distribution parameters (M-step).
- **Outcome:** The implementation successfully clustered synthetic 2D data, maximizing the log-likelihood and recovering the parameters of the underlying Gaussian components.
- **Next Steps:** Explore more advanced clustering techniques such as Density-Based Spatial Clustering (DBSCAN) or spectral clustering for non-convex shapes.

### Experiment 0140: t-Distributed Stochastic Neighbor Embedding (t-SNE)
- **Hypothesis:** By converting high-dimensional Euclidean distances into conditional probabilities that represent similarities, and minimizing the Kullback-Leibler divergence with a Student-t distribution in the low-dimensional space, a model can perform structure-preserving dimensionality reduction, effectively handling non-linear manifolds and the crowding problem.
- **Action:** Implemented t-SNE in `train_tsne_component.py` mathematically in pure NumPy, using a binary search for perplexity, Student-t distribution for the embedded space, and gradient descent with momentum and early exaggeration.
- **Outcome:** The implementation successfully reduced the dimensionality of two separated 10D Gaussian clusters into 2D, preserving their separation and minimizing the KL divergence.
- **Next Steps:** Evaluate t-SNE on larger, more complex real-world datasets (like MNIST) to visualize higher-order structural relationships.

### Experiment 0137: Density-Based Spatial Clustering (DBSCAN)
- **Hypothesis:** Density-Based Spatial Clustering of Applications with Noise (DBSCAN) can effectively identify clusters of arbitrary, non-convex shapes by grouping together closely packed points, while explicitly handling outliers as noise.
- **Action:** Implemented DBSCAN in `train_dbscan_component.py` mathematically in pure NumPy, using density-reachability from core points based on $L_2$ norm.
- **Outcome:** The implementation successfully clustered synthetic non-convex data ("two moons"), correctly separating the structures and isolating noise, where K-Means would fail.
- **Next Steps:** Explore hierarchical density-based clustering (HDBSCAN) to alleviate the sensitivity to the distance hyperparameter.

### Experiment 0145: Bayesian Optimization
- **Hypothesis:** By employing a Gaussian Process as a surrogate model and maximizing an Expected Improvement acquisition function, a model can efficiently find the global maximum of complex, non-convex black-box functions.
- **Action:** Implemented Bayesian Optimization in `train_bayesian_optimization_component.py` mathematically in pure NumPy, using a Gaussian Process to model the objective and maximizing Expected Improvement to sample new points.
- **Outcome:** The implementation efficiently found a near-optimal maximum of the complex non-convex function $f(x) = x \sin(x)$ within the bounds.
- **Next Steps:** Integrate Bayesian Optimization for hyperparameter tuning of other components within the repository.

### Experiment 0150: Tabular Q-Learning Component
- **Hypothesis:** By iteratively updating an action-value function using the Bellman equation and an off-policy temporal difference learning rule, an agent can learn an optimal policy to navigate a Markov Decision Process without a model of the environment dynamics.
- **Action:** Implemented Tabular Q-Learning in `train_q_learning_component.py` mathematically in pure NumPy, using an epsilon-greedy strategy for exploration and exploiting the maximum Q-value for temporal difference updates.
- **Outcome:** The agent successfully converged, learning an optimal policy to navigate a 4x4 GridWorld environment to the goal state in the minimum number of steps.
- **Next Steps:** Evaluate Q-learning scaled with deep neural networks (DQN) for complex, high-dimensional state spaces.

### Experiment 0151: SARSA Component
- **Hypothesis:** By iteratively updating an action-value function using the Bellman equation and an on-policy temporal difference learning rule based on the actual action chosen by the behavior policy, an agent can learn a safer policy compared to Q-learning.
- **Action:** Implemented SARSA in `train_sarsa_component.py` mathematically in pure NumPy, applying the temporal difference update utilizing the specific next action selected by the epsilon-greedy policy.
- **Outcome:** The agent successfully learned to navigate the GridWorld environment to the goal state, correctly evaluating the policy it was currently executing.
- **Next Steps:** Compare SARSA and Q-learning in an environment with high penalty states adjacent to the optimal path (e.g., Cliff Walking) to observe risk-averse vs risk-seeking behavior.

### Experiment 0152: Support Vector Machine (SVM) Component
- **Hypothesis:** By optimizing a hinge loss function using subgradient descent, a linear Support Vector Machine can discover a robust decision boundary that maximizes the margin between different classes of a linearly separable dataset.
- **Action:** Implemented a linear Support Vector Machine in `train_svm_component.py` mathematically in pure NumPy, optimizing the weights and bias via subgradient descent on the regularized hinge loss objective.
- **Outcome:** The implementation correctly converged, achieving 100% accuracy and perfectly separating the distinct linear clusters.
- **Next Steps:** Extend the model to incorporate non-linear kernel functions (e.g., Radial Basis Function - RBF, Polynomial) to handle non-linearly separable datasets via the kernel trick.

### Experiment 0154: K-Nearest Neighbors (KNN)
- **Hypothesis:** By classifying a given sample based on the majority vote of its $k$-nearest neighbors in the feature space using a distance metric, a non-parametric model can effectively learn non-linear decision boundaries for classification tasks without making explicit assumptions about the underlying data distribution.
- **Action:** Implemented K-Nearest Neighbors in `train_knn_component.py` mathematically in pure NumPy, using Euclidean distance to compute similarities between training points and test samples, and applying a majority voting scheme for classification.
- **Outcome:** The implementation successfully classified a synthetic linearly separable dataset, achieving a high accuracy of 100.00% and correctly distinguishing between the two Gaussian clusters. The KNN algorithm effectively generalized the distance-based classification logic.
- **Next Steps:** Evaluate KNN on multi-class datasets and explore alternative distance metrics such as Manhattan or Minkowski distances, as well as distance-weighted voting strategies to account for the varying influences of closer vs. further neighbors.

### Experiment 0155: Gaussian Naive Bayes Component
- **Hypothesis:** By applying Bayes' theorem with the "naive" assumption of conditional independence between features given the class label, a probabilistic classifier can effectively and efficiently categorize continuous data using a Gaussian distribution to model the likelihood of each feature.
- **Action:** Implemented Gaussian Naive Bayes in `train_naive_bayes_component.py` mathematically in pure NumPy. The implementation computes the priors from the class frequencies and models the likelihood of each feature for each class as a Gaussian distribution by estimating the mean and variance from the training data.
- **Outcome:** The implementation successfully classified a synthetic dataset consisting of three distinct Gaussian clusters. The model achieved a high accuracy of 96.67% on the held-out test set, demonstrating its ability to accurately model the class-conditional distributions.
- **Next Steps:** Evaluate Naive Bayes on high-dimensional text classification tasks using a Multinomial or Bernoulli variant to handle discrete word counts or binary occurrence features.

### Experiment 0156: Logistic Regression Component
- **Hypothesis:** By optimizing a binary cross-entropy loss function using gradient descent, a Logistic Regression model can effectively learn a linear decision boundary to separate binary classes, outputting probabilities using the sigmoid function.
- **Action:** Implemented Logistic Regression mathematically in pure NumPy, computing predictions via the sigmoid function and updating weights using gradient descent on the log-loss.
- **Outcome:** The model successfully converged, learning the correct decision boundary and classifying the binary dataset with high accuracy.
- **Next Steps:** Extend the model to multi-class classification using Softmax regression (multinomial logistic regression) and test on multi-class datasets.

### Experiment 0157: Decision Tree Component
- **Hypothesis:** By recursively splitting the feature space based on threshold values that maximize Information Gain (derived from Gini Impurity), a Decision Tree can learn non-linear decision boundaries for classification tasks without requiring feature scaling or distributional assumptions.
- **Action:** Implemented a Decision Tree mathematically in pure NumPy, using recursive splitting and Information Gain based on Gini Impurity.
- **Outcome:** The implementation successfully classified a synthetic binary dataset, achieving a high accuracy, verifying the splitting logic and recursive tree building.
- **Next Steps:** Extend the model to regression tasks (Regression Trees) and implement pruning techniques to prevent overfitting on noisy datasets.

### Experiment 0158: Random Forest Component
- **Hypothesis:** By training an ensemble of Decision Trees on bootstrap samples of the dataset and selecting random feature subsets for each split, a Random Forest can significantly reduce the variance and overfitting typically associated with individual decision trees.
- **Action:** Implemented a Random Forest mathematically in pure NumPy, constructing an ensemble of decision trees using bagging and random feature selection, and using majority voting for final predictions.
- **Outcome:** The implementation successfully classified a synthetic binary dataset using an ensemble of trees, achieving a high accuracy and verifying the bootstrapping and majority voting logic.
- **Next Steps:** Evaluate the Random Forest on higher-dimensional datasets with correlated features and compare its robustness to noise against a single Decision Tree.

### Experiment 0159: AdaBoost Component
- **Hypothesis:** By iteratively training weak classifiers on a dataset where misclassified samples are assigned higher weights, an ensemble model (AdaBoost) can combine these weak models into a strong classifier capable of learning complex non-linear decision boundaries.
- **Action:** Implemented AdaBoost in `train_adaboost_component.py` mathematically in pure NumPy, using decision stumps as weak learners and iteratively updating sample weights based on classification errors.
- **Outcome:** The implementation successfully classified a synthetic binary dataset, achieving a high accuracy of 82.50%, demonstrating the effective boosting of weak learners.
- **Next Steps:** Extend AdaBoost to handle multi-class classification problems (e.g., using SAMME or SAMME.R algorithms) and compare its robustness against Random Forests.


### Exploration of Gradient Boosting Regression
- Investigated the concept of Gradient Boosting.
- Formulated a Gradient Boosting Regressor using Decision Stumps as weak learners to iteratively minimize Mean Squared Error.
- Tested the implementation mathematically on a noisy sine wave regression task, successfully minimizing error below the threshold.
- Authored script `train_gradient_boosting_component.py` and generated documentation `docs/0160_train_gradient_boosting_component.md`.

### Experiment 0167: Lasso Regression Component
- **Hypothesis:** By optimizing a Mean Squared Error objective with an L1 regularization penalty via subgradient descent, a Lasso Regression model can effectively perform feature selection by driving the weights of irrelevant features to exactly zero.
- **Action:** Implemented a Lasso Regression model mathematically in pure NumPy using subgradient descent.
- **Outcome:** The implementation successfully recovered the weights for a synthetic dataset and reduced the weight of an irrelevant feature to near zero.
- **Next Steps:** Extend the model to incorporate Elastic Net (combining L1 and L2 penalties).

### Experiment 0168: Elastic Net Regression Component
- **Hypothesis:** By combining L1 and L2 regularization penalties, an Elastic Net Regression model can effectively perform feature selection like Lasso while maintaining the regularization properties of Ridge regression, preventing overfitting in datasets with highly correlated features.
- **Action:** Implemented an Elastic Net Regression model mathematically in pure NumPy using subgradient descent, incorporating both `alpha * l1_ratio` for the L1 penalty and `alpha * (1 - l1_ratio)` for the L2 penalty.
- **Outcome:** The implementation successfully recovered the weights for a synthetic dataset, balancing between sparsity and magnitude penalization.
- **Next Steps:** Explore tree-based regression models for non-linear relationships.


### Experiment 0169: Decision Tree Regression Component
- **Hypothesis:** By recursively splitting the feature space based on threshold values that maximize variance reduction (minimizing Mean Squared Error within splits), a Decision Tree can effectively learn non-linear regression functions without requiring feature scaling or distributional assumptions.
- **Action:** Implemented a Decision Tree Regressor mathematically in pure NumPy. The model recursively finds splits that maximize the difference between the variance of the parent node and the weighted variance of the child nodes. Tested the implementation on a synthetic noisy sine wave dataset.
- **Outcome:** The model successfully fit the non-linear dataset. Achieved a low Mean Squared Error (MSE), indicating that the recursive splitting logic and leaf value calculation (mean of target values) correctly approximated the underlying function.
- **Next Steps:** Extend the model to an ensemble by implementing a Random Forest Regressor to improve generalization and reduce variance.

### Experiment 0170: Random Forest Regression Component
- **Hypothesis:** By training an ensemble of Decision Trees on bootstrap samples of the dataset and selecting random feature subsets for each split, a Random Forest Regressor can significantly reduce the variance and overfitting typically associated with individual decision trees, leading to lower Mean Squared Error on non-linear regression tasks on unseen data.
- **Action:** Implemented a Random Forest Regressor mathematically in pure NumPy, using an ensemble of decision trees with bootstrap aggregation and random feature selection.
- **Outcome:** The implementation successfully fit the non-linear dataset and achieved lower test Mean Squared Error compared to a single decision tree, verifying its variance reduction properties.
- **Next Steps:** Explore advanced boosting regression models such as XGBoost or LightGBM equivalents mathematically.

### Experiment 0171: XGBoost Regression Component
- **Hypothesis:** By employing a second-order Taylor expansion to approximate the loss function and using L2 regularization on leaf weights, an XGBoost Regressor can effectively and robustly minimize the objective function and prevent overfitting on non-linear data.
- **Action:** Implemented an XGBoost Regressor mathematically in pure NumPy, calculating gradients and hessians for Mean Squared Error and iteratively adding trees.
- **Outcome:** The model successfully fit a noisy non-linear dataset (sine wave) achieving a low MSE, verifying the gradient boosting mechanism.
- **Next Steps:** Consider adding LightGBM optimizations such as Gradient-based One-Side Sampling (GOSS).

### Experiment 0178: Scaling Law Projection Component
- **Hypothesis:** By fitting an empirical scaling law L = C * N^(-alpha) to the performance of smaller models, we can mathematically project the parameter count and computational resources (FLOPs) required to achieve a target AGI-level loss threshold.
- **Action:** Implemented a Scaling Law Projection component mathematically in pure NumPy, simulating model data, fitting a log-linear regression, and projecting resources for L=0.01.
- **Outcome:** The implementation successfully recovered the scaling law parameters (C and alpha) and projected the required scale, validating the mathematical mechanism for resource estimation.
- **Next Steps:** Refine projection by incorporating multi-modal data scaling laws and dataset size constraints (Chinchilla optimality).

### Experiment 0185: AGI/ASI Component
- **Hypothesis:** By implementing a meta-learning architecture, we can mathematically simulate the overarching training process for AGI/ASI systems.
- **Action:** Implemented the AGI/ASI component in pure NumPy (`train_agi_asi_component.py`) and executed it successfully.
- **Outcome:** The implementation successfully completed the overarching proxy training loop for meta learning, representing a symbolic transition towards training larger intelligent structures as described in Phase 5.

### Experiment 0189: Chinchilla Optimality Scaling Component
- **Hypothesis:** By employing empirical scaling laws L(N, D) = E + A/N^alpha + B/D^beta, we can mathematically determine the optimal allocation of a given compute budget between parameter count (N) and dataset size (D) to achieve the minimum possible loss.
- **Action:** Implemented a Chinchilla Optimality Scaling component mathematically in pure NumPy, projecting the optimal parameter count, dataset size, and resulting loss for various compute budgets.
- **Outcome:** The implementation successfully projected the optimal configurations for different compute budgets based on the Chinchilla scaling principle (D ~ 20N), validating the mathematical mechanism for compute-optimal model training.
- **Next Steps:** Apply findings from Chinchilla optimality scaling plots (`chinchilla_analysis.md`) to dynamically adjust the data collection rate relative to model size expansion in the Phase 5 overarching self-improvement loops.

### Experiment 0215: Canonical Correlation Analysis (CCA) Component
- **Hypothesis:** By finding linear projections that maximize the cross-covariance between two multimodal views (variables X and Y), a CCA component can effectively learn shared representations in a common subspace mathematically without supervision.
- **Action:** Implemented a Canonical Correlation Analysis (CCA) component mathematically in pure NumPy, using eigenvalue decomposition on the cross-covariance matrices. Tested the implementation on a synthetic multimodal dataset with an underlying shared latent variable.
- **Outcome:** The implementation successfully found projections yielding high canonical correlation (0.9967) between the views, verifying the mathematical mechanism for finding maximally correlated subspaces.
- **Next Steps:** Evaluate on real-world multimodal representation learning benchmarks and explore non-linear variants like Kernel CCA.

### Experiment 0217: Agglomerative Clustering Component
- **Hypothesis:** By iteratively merging the closest pairs of clusters based on a specified linkage criterion (e.g., single linkage), Agglomerative Clustering can effectively group non-convex or well-separated data without pre-specifying the number of clusters.
- **Action:** Implemented an Agglomerative Clustering component mathematically in pure NumPy, using single linkage based on pairwise Euclidean distances. Tested the implementation on a synthetic dataset with two well-separated clusters.
- **Outcome:** The implementation successfully clustered the dataset with 100.00% accuracy, verifying the bottom-up hierarchical clustering mathematical mechanism.
- **Next Steps:** Evaluate on datasets with varying densities and explore other linkage criteria like complete and average linkage.
- **Mean Shift Component**: Evaluates a Mean Shift Clustering component mathematically in pure NumPy, testing its ability to iteratively find density maxima using an RBF kernel.

### Experiment 0220: Label Propagation Component
- **Hypothesis:** By constructing an affinity graph and iteratively propagating labels from a small set of labeled data points to a larger set of unlabeled data points, a semi-supervised Label Propagation model can effectively classify the entire dataset using the underlying manifold structure.
- **Action:** Implemented a Label Propagation component mathematically in pure NumPy, using an RBF kernel affinity matrix and iterative transition matrix multiplication. Tested the implementation on a synthetic two-cluster dataset with most labels hidden.
- **Outcome:** The implementation successfully propagated the labels across the clusters, achieving 100.00% accuracy, verifying the semi-supervised learning mathematical mechanism.
- **Next Steps:** Evaluate on larger datasets and explore variations like Label Spreading which incorporate graph regularization.

### Experiment 0221: Isolation Forest Component
- **Hypothesis:** By building an ensemble of isolation trees using random feature selection and random split values, anomalies can be effectively isolated since they require fewer splits (shorter path lengths) to be separated from the rest of the data.
- **Action:** Implemented an Isolation Forest in `train_isolation_forest_component.py` mathematically in pure NumPy, using recursive binary splitting and evaluating mean path lengths for anomaly scoring.
- **Outcome:** The model successfully identified anomalies, which received significantly higher anomaly scores (shorter average path lengths) compared to the normal data points.
- **Next Steps:** Evaluate the model on higher-dimensional datasets with more complex anomaly structures.
### Experiment 0226: Partial Least Squares (PLS) Component
- **Hypothesis:** By finding latent variables that maximize the covariance between independent (X) and dependent (Y) variable sets, Partial Least Squares can effectively perform regression even when predictors are highly collinear or outnumber observations.
- **Action:** Implemented a Partial Least Squares Regression component mathematically in pure NumPy, using the NIPALS algorithm to iteratively extract orthogonal latent variables and their loadings.
- **Outcome:** The implementation successfully computed the latent components, verifying the mathematical mechanism for dimensionality reduction and regression on multivariate data.
- **Next Steps:** Evaluate on datasets with severe multicollinearity and compare predictive performance against Ridge regression and standard OLS.

### Experiment 0227: Quadratic Discriminant Analysis (QDA) Component
- **Hypothesis:** By calculating class-specific priors, means, and covariance matrices, Quadratic Discriminant Analysis can effectively classify instances by computing the likelihood of the instance belonging to each class, accounting for different variances across classes.
- **Action:** Implemented a Quadratic Discriminant Analysis component mathematically in pure NumPy, using maximum likelihood estimates for priors, means, and covariances. Tested the implementation on a synthetic dataset with two classes having different covariances.
- **Outcome:** The implementation successfully classified the dataset, verifying the probabilistic classification mathematical mechanism.
- **Next Steps:** Evaluate on datasets with more complex class distributions and compare predictive performance against Linear Discriminant Analysis and Naive Bayes.

### Experiment 0228: Kernel Density Estimation (KDE) Component
- **Hypothesis:** By placing a continuous kernel function (like a Gaussian) at each data point and averaging them, Kernel Density Estimation can effectively model the underlying continuous probability density function of an unknown data distribution.
- **Action:** Implemented a Kernel Density Estimation component mathematically in pure NumPy, using a Gaussian kernel with a specified bandwidth. Tested the implementation on synthetic bimodal data.
- **Outcome:** The implementation successfully estimated higher densities at the true modes and lower densities in the valleys, verifying the mathematical mechanism for continuous density estimation.
- **Next Steps:** Evaluate with different kernel functions (Epanechnikov, Tophat) and implement automatic bandwidth selection techniques like Silverman's rule of thumb.
- **Slow Feature Analysis (SFA)**: Successfully implemented and mathematically verified. SFA extracts slowly varying temporal features from rapidly varying input signals by solving a generalized eigenvalue problem on the signal and its temporal derivative covariance.

### Experiment 0230: Deep Belief Network (DBN) Component
- **Hypothesis:** By stacking multiple Restricted Boltzmann Machines (RBMs) and training them greedily layer-by-layer using Contrastive Divergence (CD-1), a Deep Belief Network can effectively learn deep hierarchical representations of the input data.
- **Action:** Implemented a Deep Belief Network (DBN) component mathematically in pure NumPy, consisting of stacked RBMs. Tested the implementation on a synthetic binary dataset to confirm greedy layer-wise training convergence.
- **Outcome:** The implementation successfully reduced the reconstruction error progressively across the layers, verifying the mathematical mechanism of unsupervised greedy layer-wise learning.
- **Next Steps:** Evaluate the fine-tuning of the entire network using backpropagation or the Wake-Sleep algorithm.
### Experiment 0231: Upper Confidence Bound (UCB) Component
- **Hypothesis:** By balancing exploration and exploitation using an upper confidence bound that increases for less frequently chosen actions, a UCB agent can effectively solve the multi-armed bandit problem and maximize cumulative reward.
- **Action:** Implemented a Upper Confidence Bound component mathematically in pure NumPy, testing it on a synthetic multi-armed bandit problem.
- **Outcome:** The implementation successfully favored the optimal action while sufficiently exploring sub-optimal ones, verifying the UCB mathematical mechanism.
- **Next Steps:** Evaluate in contextual bandit settings and compare against Thompson Sampling.

### Experiment 0232: Graph Isomorphism Network (GIN) Component
- **Hypothesis:** By using an injective aggregation function (summation with a learnable epsilon) and a multi-layer perceptron (MLP), a Graph Isomorphism Network can achieve maximum discriminative power among graph neural networks, capable of distinguishing different graph structures.
- **Action:** Implemented a Graph Isomorphism Network layer mathematically in pure NumPy, testing its forward pass on a synthetic graph.
- **Outcome:** The implementation successfully computed node representations, verifying the mathematical mechanism of injective neighbor aggregation.
- **Next Steps:** Evaluate on graph classification benchmarks to confirm its theoretical expressive power.

### Experiment 0233: Fuzzy C-Means (FCM) Component
- **Hypothesis:** By assigning soft probabilities (membership degrees) to each data point for belonging to multiple clusters, Fuzzy C-Means can more effectively cluster data with overlapping boundaries or ambiguities compared to hard clustering methods like K-Means.
- **Action:** Implemented a Fuzzy C-Means clustering component mathematically in pure NumPy, updating cluster centers based on weighted memberships and iterating until convergence. Tested on a synthetic dataset.
- **Outcome:** The implementation successfully partitioned the data and assigned membership values, verifying the mathematical mechanism of fuzzy clustering.
- **Next Steps:** Evaluate on datasets with highly overlapping clusters and tune the fuzzifier parameter (m) for optimal separation.

### Experiment 0234: Kernel Ridge Regression Component
- **Hypothesis:** By applying the kernel trick to ridge regression and solving it in the dual space, the model can effectively perform regularized non-linear regression, mapping the input features to an infinite-dimensional feature space.
- **Action:** Implemented a Kernel Ridge Regression component mathematically in pure NumPy, using an RBF kernel and solving the dual formulation linearly. Tested the implementation on a noisy sine wave dataset.
- **Outcome:** The implementation successfully learned the underlying non-linear pattern while ignoring most noise, verifying the mathematical mechanism of kernel methods and L2 regularization in dual form.
- **Next Steps:** Evaluate on complex non-linear regression benchmarks and explore other kernels like Polynomial and Sigmoid.

### Experiment 0235: Local Outlier Factor (LOF) Component
- **Hypothesis:** By comparing the local reachability density of a data point to the densities of its k-nearest neighbors, anomalies can be effectively identified as points that have significantly lower densities than their neighbors.
- **Action:** Implemented a Local Outlier Factor component mathematically in pure NumPy, computing k-distances, reachability distances, and local outlier factors. Tested on synthetic data with clustered normal points and distant outliers.
- **Outcome:** The implementation successfully assigned significantly higher LOF scores to the outliers compared to normal points, verifying the mathematical mechanism of local density-based anomaly detection.
- **Next Steps:** Evaluate on complex real-world datasets with varying local densities and explore optimization techniques for neighbor search.
- Contractive Autoencoder (CAE) mathematically verified with Frobenius norm regularization on Jacobian of hidden representations.

### Experiment 0237: Conditional Random Field (CRF) Component
- **Hypothesis:** By modeling the conditional probability of a label sequence given an input sequence using undirected graphical models, a CRF can effectively learn to tag sequences while considering dependencies between neighboring labels.
- **Action:** Implemented a Conditional Random Field component mathematically in pure NumPy, using the forward-backward algorithm to compute node and edge marginals for exact inference and gradient computation. Tested on a synthetic sequence dataset.
- **Outcome:** The implementation successfully learned the emission and transition potentials, correctly computing gradients using expected counts from the marginals, verifying the mathematical mechanism.
- **Next Steps:** Evaluate on real sequence tagging tasks like Named Entity Recognition or Part-of-Speech tagging and explore linear-chain approximations for longer sequences.
- [x] GraphSAGE implemented for neighbor sampling GNN approaches
- Implemented Particle Filter component (Sequential Monte Carlo) for non-linear state estimation.

### Experiment 0242: Value Iteration Component
- **Hypothesis:** By iteratively updating the value function using the Bellman optimality equation, the algorithm can converge to the optimal value function and extract the optimal policy for a given Markov Decision Process.
- **Action:** Implemented Value Iteration mathematically in pure NumPy and tested it on a simple gridworld environment.
- **Outcome:** The implementation successfully converged and found the optimal policy, verifying the mathematical mechanism of Value Iteration.
- **Next Steps:** Evaluate on larger state spaces and compare with reinforcement learning approaches like Q-Learning.

### Experiment 0243: Policy Iteration Component
- **Hypothesis:** By alternating between policy evaluation and policy improvement steps, Policy Iteration can converge to the optimal policy, potentially in fewer iterations than Value Iteration.
- **Action:** Implemented Policy Iteration mathematically in pure NumPy, consisting of iterative policy evaluation and greedy policy improvement, tested on a synthetic gridworld environment.
- **Outcome:** The implementation successfully converged to the optimal policy, verifying the mathematical mechanism of alternating evaluation and improvement steps.
- **Next Steps:** Evaluate in environments with stochastic transitions and compare computational efficiency with Value Iteration.
- **Sparse PCA**: Explored sparse principal component analysis using L1-regularized power iteration.
- **CP Decomposition**: Explored CANDECOMP/PARAFAC tensor decomposition via simulated ALS.
- **Tucker Decomposition**: Explored Tucker tensor decomposition (HOSVD) via simulated HOOI.
- **MCMC**: Explored Markov Chain Monte Carlo via the Metropolis-Hastings algorithm, confirming robust theoretical mean and variance estimation for a complex Gaussian mixture target probability density function.
- Implemented and verified Advantage Actor-Critic - A2C using shared features mathematically.
### Experiment 0249: Genetic Algorithm Component
- **Hypothesis:** By employing principles of natural selection including fitness-based selection, crossover (recombination), and mutation, a population-based optimization algorithm can effectively navigate complex, non-convex loss landscapes and find near-optimal global solutions without requiring gradients.
- **Action:** Implemented a Genetic Algorithm in `train_genetic_algorithm_component.py` mathematically in pure NumPy, using tournament selection, blend crossover, and Gaussian mutation to minimize the non-convex Rastrigin function.
- **Outcome:** The implementation successfully converged on the global minimum of the 2D Rastrigin function, demonstrating robust gradient-free optimization.
- **Next Steps:** Evaluate the integration of Genetic Algorithms for hyperparameter optimization and Neural Architecture Search (Neuroevolution).
### Experiment 0250: Differential Evolution Component
- **Hypothesis:** By perturbing a population of candidate solutions with scaled differences between other population members, Differential Evolution can efficiently optimize continuous, non-differentiable objectives without relying on analytical gradients.
- **Action:** Implemented Differential Evolution in `train_differential_evolution_component.py` in pure NumPy.
- **Outcome:** The algorithm successfully minimized the 5-dimensional Rastrigin function, effectively demonstrating its capability for global optimization over a multi-modal landscape.
- **Next Steps:** Evaluate its effectiveness in tuning neural network architectures or hyperparameter optimization compared to CMA-ES.
- Implemented and verified Gaussian Process Regression (GPR) for non-parametric Bayesian modeling.
### Experiment 0252: Ant Colony Optimization Component
- **Hypothesis:** By simulating the behavior of artificial ants updating pheromone trails on a graph, the algorithm can effectively navigate the search space to find near-optimal shortest paths for the Traveling Salesperson Problem (TSP).
- **Action:** Implemented Ant Colony Optimization in `train_aco_component.py` mathematically in pure NumPy, using heuristic distances and pheromone evaporation on a 15-city TSP instance.
- **Outcome:** The implementation successfully found a short tour length, demonstrating the capability of swarm intelligence for combinatorial optimization.
- **Next Steps:** Evaluate its applicability for routing problems or dynamic network optimization in comparison to other metaheuristics.
## Beta-VAE Exploration
Explored Disentangled Variational Autoencoders (Beta-VAE). Implemented a pure NumPy version to verify disentanglement capabilities where beta parameter penalizes Kullback-Leibler divergence.
- **GloVe**: Verified that GloVe can effectively learn word representations by factorizing a co-occurrence matrix with a weighted least squares objective, explicitly capturing global corpus statistics.
- Implemented Latent Semantic Analysis (LSA) component mathematically, applying SVD to a TF-IDF matrix to learn latent document and word representations.
- **Laplacian Eigenmaps**: Verified that Laplacian Eigenmaps can learn a low-dimensional manifold by preserving local connectivity using the graph Laplacian.
- **Squeeze-and-Excitation (SE) Block**: Explored channel attention mathematically in pure NumPy, learning to scale channels explicitly.
### Experiment 0258: Thompson Sampling Component
- **Hypothesis:** By maintaining a posterior distribution over the true reward probabilities of each arm and sampling from these distributions to select actions, the algorithm can effectively balance exploration and exploitation in a multi-armed bandit setting.
- **Action:** Implemented Thompson Sampling mathematically in pure NumPy, using Beta distributions as conjugate priors for Bernoulli rewards. Tested on a 3-arm bandit problem.
- **Outcome:** The implementation successfully converged to selecting the optimal arm most frequently, verifying the probabilistic exploration mechanism.
- **Next Steps:** Evaluate in contextual bandit settings and compare against UCB.
- Implemented DNC component.
- Implemented NEAT component.
- Implemented CMA-ES component.
