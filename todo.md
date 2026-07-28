# AGI/ASI Roadmap & To-Do

*This document outlines the strategic roadmap for achieving Artificial General Intelligence (AGI) and Artificial Superintelligence (ASI). The plan progresses from fundamental mathematical research to large-scale system integration.*

## Phase 1: Foundations and Mathematical Modeling (Current)

- [x] Define the core theoretical framework for intelligence modeling.
- [x] Investigate and mathematically formalize memory and reasoning structures. (Verified FFN for reasoning structures, implemented Self-Attention, Multi-Head Attention, Layer Normalization, Masked Attention, Cross-Attention, sequential RNN state memory, GRU gating mechanisms, LSTM cell states and gating mechanisms, and continuous State Space Model (SSM) discretizations).
- [x] Design and conduct initial small-scale experiments to test architectural hypotheses.
- [x] Establish rigorous evaluation metrics for component performance.

## Phase 2: Component Engineering and Optimization
- [x] Develop optimized implementations of successful theoretical models. (Implemented Batch Normalization for stabilizing and accelerating deep network training).
- [x] Develop grouped feature normalization. (Implemented Group Normalization as an alternative to Batch Normalization for small batch sizes).
- [x] Investigate generalization dynamics (Grokking) and structural representation learning mathematically. (Implemented Grokking MLP for modular arithmetic to observe memorization vs generalization phases).

- [x] Explore and implement generative models mathematically. (Implemented and verified Denoising Diffusion Probabilistic Model (DDPM) reverse process).
- [x] Explore and implement discrete generative representation models mathematically. (Implemented and verified VQ-VAE with Straight-Through Estimator).
- [x] Develop optimized implementations of successful theoretical models. (Implemented Linear Attention using ELU+1 feature map to reduce $O(N^2)$ complexity to $O(N)$).

- [x] Develop optimized implementations of successful theoretical models. (Implemented Retention mechanism bridging Transformer parallelization with RNN $O(1)$ recurrent inference).

- [x] Develop optimized implementations of successful theoretical models. (Implemented RMSNorm as an optimized alternative to LayerNorm, SwiGLU for enhanced capacity, RoPE for enhanced positional embeddings, Mixture of Experts for specialized routing, Grouped-Query Attention for optimized attention scaling, ALiBi for positional bias injection, AdamW optimizer for accelerated convergence, QAT for memory optimization, and Kolmogorov-Arnold Networks (KAN) for edge-based activation learning).
- [x] Investigate mechanistic interpretability of representations. (Implemented Sparse Autoencoder (SAE) to extract disentangled, sparse features from dense representations).
- [x] Develop memory-efficient continuous architectures. (Implemented Reversible Residual Networks (RevNet) to allow $O(1)$ memory backpropagation).
- [x] Develop dynamically conditioned normalization. (Implemented Adaptive Layer Normalization (AdaLN) to allow scaling and shifting based on conditioning inputs).
- [x] Study scaling laws for individual components (e.g., attention, memory retrieval).
- [x] Develop knowledge distillation methodologies mathematically. (Implemented and verified transferring knowledge from a Teacher MLP to a Student MLP using KL Divergence and Temperature scaling).
- [x] Develop methods to mitigate vanishing gradients in deep architectures. (Implemented and verified Highway Networks with transform and carry gating mechanisms).
- [x] Develop alternatives to self-attention for sequence modeling mathematically. (Implemented and verified MLP-Mixer combining Token-mixing and Channel-mixing MLPs).
- [x] Develop alternatives to self-attention for sequence modeling mathematically. (Implemented and verified gMLP utilizing a Spatial Gating Unit for modeling spatial dependencies).
- [ ] Create a modular, extensible codebase for integrating various AI subsystems.
- [x] Develop dynamic weight generation methodologies mathematically. (Implemented and verified Hypernetwork component for generating context-conditioned primary network weights).
- [ ] Automate the pipeline for training, evaluation, and documentation generation.

## Phase 3: Integration and Capability Emergence

- [x] Integrate optimized components into unified architectures. (Implemented and verified single Transformer Block, Decoder Block combining components, and a full end-to-end Encoder-Decoder architecture).
- [x] Develop mathematical models for preference alignment and safety. (Implemented and verified Direct Preference Optimization - DPO).
- [ ] Train medium-scale models to observe emergent capabilities.
- [ ] Analyze failure modes, alignment issues, and out-of-distribution generalization.
- [ ] Refine the architecture based on empirical results from integrated systems.

## Phase 4: Scaling to AGI

- [ ] Secure infrastructure for large-scale training.
- [ ] Apply empirical scaling laws to project resource requirements for AGI-level performance.
- [ ] Train a unified model on multi-modal, diverse data streams.
- [ ] Conduct rigorous safety and alignment testing on the general model.

## Phase 5: Transition to ASI

- [ ] Implement self-improvement and self-correction loops within the AGI architecture.
- [ ] Monitor capability takeoff and ensure alignment mechanisms hold under recursive self-improvement.
- [ ] Transition operational control to the aligned ASI system.
