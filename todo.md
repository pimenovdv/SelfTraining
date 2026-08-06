# AGI/ASI Roadmap & To-Do

*This document outlines the strategic roadmap for achieving Artificial General Intelligence (AGI) and Artificial Superintelligence (ASI). The plan progresses from fundamental mathematical research to large-scale system integration.*

## Phase 1: Foundations and Mathematical Modeling (Current)

- [x] Define the core theoretical framework for intelligence modeling.
- [x] Investigate and mathematically formalize memory and reasoning structures. (Verified FFN for reasoning structures, implemented Self-Attention, Multi-Head Attention, Layer Normalization, Masked Attention, Cross-Attention, sequential RNN state memory, GRU gating mechanisms, LSTM cell states and gating mechanisms, continuous State Space Model (SSM) discretizations, Hopfield Network associative memory, and End-To-End Memory Network).
- [x] Design and conduct initial small-scale experiments to test architectural hypotheses.
- [x] Establish rigorous evaluation metrics for component performance.

## Phase 2: Component Engineering and Optimization
- [x] Explore masked autoencoding for self-supervised representation learning mathematically. (Implemented and verified Masked Autoencoder - MAE with asymmetric encoder-decoder).
- [x] Explore intrinsic motivation and curiosity-driven exploration mathematically. (Implemented and verified Intrinsic Curiosity Module - ICM).
- [x] Explore robust representation learning mathematically. (Implemented and verified Denoising Autoencoder - DAE reconstructing data from noisy inputs).
- [x] Explore continuous implicit representations mathematically. (Implemented and verified Sinusoidal Representation Network - SIREN for modeling complex high-frequency signals).
- [x] Explore invertible generative modeling mathematically. (Implemented and verified RealNVP Normalizing Flow).
- [x] Explore permutation-invariant architectures mathematically. (Implemented and verified Deep Sets component).
- [x] Explore structural representation learning mathematically. (Implemented and verified Capsule Network with dynamic routing).
- [x] Explore biologically plausible alternatives to backpropagation mathematically. (Implemented and verified Predictive Coding Network - PCN using local iterative inference and Hebbian learning).
- [x] Explore differentiable discrete sampling mathematically. (Implemented and verified Gumbel-Softmax estimator).
- [x] Explore representation learning via word embeddings mathematically. (Implemented and verified Skip-Gram with Negative Sampling).
- [x] Explore non-contrastive self-supervised representation learning mathematically. (Implemented and verified SimSiam utilizing stop-gradient to prevent representation collapse).
- [x] Explore differentiable external memory structures mathematically. (Implemented and verified Neural Turing Machine - NTM with content-based addressing and differentiable read/write).
- [x] Explore topological representation learning mathematically. (Implemented and verified Self-Organizing Map - SOM).
- [x] Explore liquid time-constant dynamics mathematically. (Implemented and verified Liquid Time-Constant Network - LTC).
- [x] Explore continuous-time recurrent dynamics mathematically. (Implemented and verified Continuous-Time Recurrent Neural Network - CTRNN).
- [x] Explore biologically plausible learning rules mathematically (Feedback Alignment, Direct Feedback Alignment).
- [x] Explore sequence modeling alternatives with causal dilated convolutions. (Implemented and verified Temporal Convolutional Network - TCN).
- [x] Explore reinforcement learning foundational algorithms mathematically. (Implemented and verified REINFORCE policy gradient).
- [x] Explore Actor-Critic reinforcement learning mathematically. (Implemented and verified an Actor-Critic architecture with Temporal Difference learning).
- [x] Explore advanced reinforcement learning mathematically. (Implemented and verified Proximal Policy Optimization - PPO with clipped surrogate objective).
- [x] Explore and implement graph-based models mathematically. (Implemented and verified Graph Convolutional Network - GCN, and Graph Attention Network - GAT).
- [x] Develop optimized implementations of successful theoretical models. (Implemented Batch Normalization for stabilizing and accelerating deep network training).
- [x] Develop grouped feature normalization. (Implemented Group Normalization as an alternative to Batch Normalization for small batch sizes).
- [x] Develop optimized implementations of successful theoretical models. (Implemented Spectral Normalization using power iteration for Lipschitz continuity, and Weight Normalization for decoupling weight length from direction).
- [x] Investigate generalization dynamics (Grokking) and structural representation learning mathematically. (Implemented Grokking MLP for modular arithmetic to observe memorization vs generalization phases).
- [x] Explore decoupled neural updates mathematically. (Implemented and verified Decoupled Neural Interfaces (DNI) with Synthetic Gradients).

- [x] Explore and implement generative models mathematically. (Implemented and verified Denoising Diffusion Probabilistic Model (DDPM) reverse process).
- [x] Explore and implement discrete generative representation models mathematically. (Implemented and verified VQ-VAE with Straight-Through Estimator).
- [x] Explore and implement adversarial generative models mathematically. (Implemented and verified Generative Adversarial Network - GAN).
- [x] Explore and implement energy-based generative models mathematically. (Implemented and verified Restricted Boltzmann Machine - RBM using Contrastive Divergence).
- [x] Explore and implement continuous energy-based generative models mathematically. (Implemented and verified Energy-Based Model - EBM using Langevin Dynamics).
- [x] Explore and implement autoregressive generative models mathematically. (Implemented and verified Neural Autoregressive Distribution Estimator - NADE).
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
- [x] Explore parameter-free mixing alternatives to self-attention mathematically. (Implemented and verified FNet Block utilizing 2D Fourier Transform for token and channel mixing).
- [x] Explore alternatives to quadratic self-attention mathematically. (Implemented and verified Perceiver Bottleneck reducing $O(N^2)$ complexity to $O(N \cdot M)$ via cross-attention with trainable latents).
- [x] Explore Reservoir Computing and dynamic state projection mathematically. (Implemented and verified Echo State Network (ESN) with Ridge Regression readout on a chaotic time series).
- [x] Explore probabilistic uncertainty estimation in networks mathematically. (Implemented and verified Bayesian Neural Network (BNN) with Bayes by Backprop to optimize the ELBO).
- [x] Explore continuous-depth models mathematically. (Implemented and verified Neural ODE using Euler integration and manual backpropagation).
- [x] Explore biologically plausible spiking neural networks mathematically. (Implemented and verified Spiking Neural Network (SNN) component with Leaky Integrate-and-Fire neurons using Surrogate Gradients).
- [x] Explore Deep Q-Networks mathematically. (Implemented and verified Deep Q-Network - DQN with experience replay and target networks).
- [x] Explore localized basis functions mathematically. (Implemented and verified Radial Basis Function - RBF Network optimizing centroids and widths via backpropagation).
- [x] Explore gradient-free optimization mathematically. (Implemented and verified Evolution Strategies - ES for black-box optimization of neural network weights).
- [x] Explore multi-modal distribution modeling mathematically. (Implemented and verified Mixture Density Network - MDN predicting Gaussian Mixture parameters).
- [x] Explore meta-learning for functions mathematically. (Implemented and verified Conditional Neural Process - CNP).
- [ ] Create a modular, extensible codebase for integrating various AI subsystems.
- [x] Develop dynamic weight generation methodologies mathematically. (Implemented and verified Hypernetwork component for generating context-conditioned primary network weights).
- [x] Explore Meta-Learning mathematically. (Implemented and verified First-Order Model-Agnostic Meta-Learning - MAML).
- [ ] Automate the pipeline for training, evaluation, and documentation generation.

- [x] Explore self-organizing pattern generation mathematically. (Implemented and verified Neural Cellular Automata (NCA) growing a pattern from a seed).

- [x] Explore continuous normalizing flows via vector field regression mathematically. (Implemented and verified Flow Matching).

## Phase 3: Integration and Capability Emergence

- [x] Integrate optimized components into unified architectures. (Implemented and verified single Transformer Block, Decoder Block combining components, and a full end-to-end Encoder-Decoder architecture).
- [x] Develop mathematical models for preference alignment and safety. (Implemented and verified Direct Preference Optimization - DPO).
- [x] Explore continual learning and mitigate catastrophic forgetting mathematically. (Implemented and verified Elastic Weight Consolidation (EWC) using the Fisher Information Matrix to anchor parameters across sequential tasks).
- [x] Investigate non-iterative, analytical learning methods for rapid representation acquisition. (Implemented and verified Extreme Learning Machine (ELM) solving output weights via pseudoinverse).
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
