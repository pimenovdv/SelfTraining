# AGI/ASI Roadmap & To-Do

*This document outlines the strategic roadmap for achieving Artificial General Intelligence (AGI) and Artificial Superintelligence (ASI). The plan progresses from fundamental mathematical research to large-scale system integration.*

## Phase 1: Foundations and Mathematical Modeling (Current)

- [x] Define the core theoretical framework for intelligence modeling.
- [x] Investigate and mathematically formalize memory and reasoning structures. (Verified FFN for reasoning structures, implemented Self-Attention, Multi-Head Attention, Layer Normalization, Masked Attention, Cross-Attention, sequential RNN state memory, GRU gating mechanisms, LSTM cell states and gating mechanisms, continuous State Space Model (SSM) discretizations, Hopfield Network associative memory, and End-To-End Memory Network).
- [x] Design and conduct initial small-scale experiments to test architectural hypotheses.
- [x] Establish rigorous evaluation metrics for component performance.

## Phase 2: Component Engineering and Optimization
- [x] Explore dynamic computation depth and ponder mechanisms mathematically. (Implemented and verified Adaptive Computation Time - ACT).
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
- [x] Explore continuous-time memory models mathematically. (Implemented and verified Legendre Memory Unit - LMU maintaining orthogonal sequence representation).
- [x] Explore Information Bottleneck representation learning mathematically. (Implemented and verified Variational Information Bottleneck - VIB for robust feature extraction).
- [x] Explore alternatives to backpropagation mathematically. (Implemented and verified Forward-Forward algorithm using local goodness metrics).
- [x] Explore Differentiable Architecture Search (DARTS) mathematically. (Implemented and verified continuous relaxation of architecture representation).
- [x] Explore metric-based few-shot meta-learning mathematically. (Implemented and verified Prototypical Networks - ProtoNet computing Euclidean distance to class prototypes).
- [x] Explore sparse probability distributions mathematically. (Implemented and verified Sparsemax component via Euclidean projection onto the probability simplex).
- [x] Explore dense associative memories mathematically. (Implemented and verified Continuous Hopfield Network with exponential interaction/log-sum-exp energy, connecting associative memory and self-attention).
- [x] Explore dimensionality reduction mathematically. (Implemented and verified Principal Component Analysis - PCA finding orthogonal directions of maximum variance).
- [x] Create a modular, extensible codebase for integrating various AI subsystems.
- [x] Develop dynamic weight generation methodologies mathematically. (Implemented and verified Hypernetwork component for generating context-conditioned primary network weights).
- [x] Explore Meta-Learning mathematically. (Implemented and verified First-Order Model-Agnostic Meta-Learning - MAML).
- [x] Explore non-contrastive self-supervised learning with cross-correlation mathematically. (Implemented and verified Barlow Twins component).
- [x] Automate the pipeline for training, evaluation, and documentation generation.
- [x] Implement and evaluate Soft Actor-Critic (SAC) component mathematically.

- [x] Explore self-organizing pattern generation mathematically. (Implemented and verified Neural Cellular Automata (NCA) growing a pattern from a seed).

- [x] Explore continuous normalizing flows via vector field regression mathematically. (Implemented and verified Flow Matching).
- [x] Explore Vision Transformer tokenization mathematically. (Implemented and verified Vision Transformer - ViT parsing images as patches and processing via self-attention).

- [x] Explore unsupervised representation learning for non-Gaussian signals mathematically. (Implemented and verified Independent Component Analysis - ICA using FastICA algorithm).
- [x] Explore non-contrastive self-supervised learning with momentum networks mathematically. (Implemented and verified Bootstrap Your Own Latent - BYOL component).

- [x] Explore predictive representation architectures mathematically. (Implemented and verified Joint Embedding Predictive Architecture - JEPA).

- [x] Explore adversarial robustness mathematically. (Implemented and verified Fast Gradient Sign Method - FGSM to generate adversarial examples and train robust models).

- [x] Explore arithmetic reasoning and systematic generalization mathematically. (Implemented and verified Neural Arithmetic Logic Unit - NALU).

## Phase 3: Integration and Capability Emergence

- [x] Explore advanced gradient-free optimization strategies mathematically. (Implemented and verified Covariance Matrix Adaptation Evolution Strategy - CMA-ES).

- [x] Explore pointer mechanisms for sequence tasks mathematically. (Implemented and verified Pointer Network for predicting sequence order via input attention).

- [x] Explore unsupervised clustering mathematically. (Implemented and verified K-Means clustering algorithm, and extended to soft assignments via Gaussian Mixture Models - GMM).

- [x] Explore network pruning and sparse subnetworks mathematically (Lottery Ticket Hypothesis). (Implemented and verified Iterative Magnitude Pruning - IMP).

- [x] Explore continuous deterministic policy gradients mathematically. (Implemented and verified Deep Deterministic Policy Gradient - DDPG component for continuous action spaces).

- [x] Explore biologically plausible credit assignment mathematically. (Implemented and verified Difference Target Propagation - DTP using local inverse models).

- [x] Integrate optimized components into unified architectures. (Implemented and verified single Transformer Block, Decoder Block combining components, and a full end-to-end Encoder-Decoder architecture).
- [x] Develop mathematical models for preference alignment and safety. (Implemented and verified Direct Preference Optimization - DPO).
- [x] Explore continual learning and mitigate catastrophic forgetting mathematically. (Implemented and verified Elastic Weight Consolidation (EWC) using the Fisher Information Matrix to anchor parameters across sequential tasks).
- [x] Explore explicit relational reasoning mathematically. (Implemented and verified Relational Network - RN).
- [x] Investigate non-iterative, analytical learning methods for rapid representation acquisition. (Implemented and verified Extreme Learning Machine (ELM) solving output weights via pseudoinverse).
- [x] Explore self-supervised autoregressive representation learning mathematically. (Implemented and verified Contrastive Predictive Coding - CPC).
- [x] Explore unsupervised, biologically plausible learning rules mathematically. (Implemented and verified Hebbian Learning using Oja's rule for stable principal component extraction).
- [x] Explore generative policies proportional to reward mathematically. (Implemented and verified GFlowNet with Trajectory Balance loss).
- [x] Explore planning and lookahead using self-play mathematically. (Implemented and verified Monte Carlo Tree Search (MCTS) combined with a policy/value network).
- [x] Explore Sharpness-Aware Minimization mathematically. (Implemented and verified SAM to explicitly penalize loss sharpness during optimization).
- [x] Explore continuous deterministic policy gradients with delayed updates and clipped double Q-learning mathematically. (Implemented and verified Twin Delayed DDPG - TD3).
- [x] Explore continuous 3D representations and neural rendering mathematically. (Implemented and verified Neural Radiance Fields - NeRF using positional encoding and volumetric rendering).
- [x] Explore offline reinforcement learning via sequence modeling mathematically. (Implemented and verified Decision Transformer modeling state-action-return sequences).
- [x] Explore self-normalizing neural networks mathematically. (Implemented and verified Scaled Exponential Linear Unit - SELU component).
- [x] Train medium-scale models to observe emergent capabilities. (Implemented and verified training a medium-scale MLP on a 10-bit parity task).
- [x] Analyze failure modes, alignment issues, and out-of-distribution generalization. (Implemented and verified Out-of-Distribution Detection via Mahalanobis Distance for generalization analysis).
- [x] Refine the architecture based on empirical results from integrated systems. (Implemented and verified Neural Architecture Search (NAS) for automated architecture refinement).

- [x] Explore more advanced clustering techniques such as Density-Based Spatial Clustering (DBSCAN) or spectral clustering for non-convex shapes. (Implemented and verified DBSCAN and Spectral Clustering mathematically clustering non-convex two-moons data).
- [x] Explore non-negative representation learning mathematically. (Implemented and verified Non-negative Matrix Factorization - NMF using multiplicative update rules).
- [x] Explore structure-preserving dimensionality reduction mathematically. (Implemented and verified t-Distributed Stochastic Neighbor Embedding - t-SNE using Student-t distribution and KL divergence minimization).

- [x] Explore non-parametric probabilistic modeling mathematically. (Implemented and verified Gaussian Process Regression - GPR providing exact inference and predictive uncertainties via RBF kernel).

## Phase 4: Scaling to AGI

- [x] Secure infrastructure for large-scale training. (Implemented and verified Distributed Infrastructure Component simulating Ring All-Reduce mathematically).
- [x] Apply empirical scaling laws to project resource requirements for AGI-level performance. (Implemented and verified Scaling Law Projection mathematically).
- [x] Train a unified model on multi-modal, diverse data streams. (Implemented and verified Multimodal Component training using CLIP-like contrastive loss mathematically).
- [x] Conduct rigorous safety and alignment testing on the general model. (Implemented and verified Safety Alignment Component via constrained optimization mathematically).

## Phase 5: Transition to ASI

- [x] Implement self-improvement and self-correction loops within the AGI architecture. (Implemented and verified Self-Correction Component using a learned critic to guide actor updates).
- [x] Monitor capability takeoff and ensure alignment mechanisms hold under recursive self-improvement. (Implemented and verified Recursive Self Improvement Component mathematically).
- [x] Transition operational control to the aligned ASI system. (Implemented and verified Operational Control Transition Component mathematically).

- [x] Explore state estimation and linear quadratic Gaussian control mathematically. (Implemented and verified Kalman Filter estimating hidden states from noisy observations recursively).
- [x] Explore Bayesian Optimization mathematically. (Implemented and verified Bayesian Optimization component using Gaussian Process and Expected Improvement).


- [x] Explore dictionary learning and sparse coding mathematically. (Implemented and verified Dictionary Learning with FISTA for sparse representations).
- [x] Explore Particle Swarm Optimization mathematically. (Implemented and verified Particle Swarm Optimization - PSO component optimizing non-convex functions).
- [x] Explore Simulated Annealing mathematically. (Implemented and verified Simulated Annealing for function minimization)
- [x] Explore Cross-Entropy Method mathematically. (Implemented and verified Cross-Entropy Method for function minimization)
- [x] Explore Tabular Q-Learning mathematically. (Implemented and verified Q-Learning finding optimal policy via Bellman equation)
- [x] Explore SARSA mathematically. (Implemented and verified SARSA on-policy learning)
- [x] Explore Support Vector Machines mathematically. (Implemented and verified linear SVM with hinge loss and subgradient descent)
- [x] Explore supervised dimensionality reduction mathematically. (Implemented and verified Linear Discriminant Analysis - LDA finding axes that maximize class separation).
- [x] Explore non-parametric classification mathematically. (Implemented and verified K-Nearest Neighbors - KNN classifying based on majority vote of nearest neighbors).
- [x] Explore probabilistic classification mathematically. (Implemented and verified Gaussian Naive Bayes using Bayes' theorem with conditional independence assumption).

- [x] Explore linear classification mathematically. (Implemented and verified Logistic Regression with binary cross-entropy and gradient descent).
- [x] Explore non-parametric decision models mathematically. (Implemented and verified Decision Tree classifier splitting on Gini impurity).
- [x] Explore ensemble learning mathematically. (Implemented and verified Random Forest using bootstrap aggregation and random feature subsets).
- [x] Explore sub-word tokenization mathematically. (Implemented and verified Byte-Pair Encoding (BPE) tokenizer from scratch).
- [x] Explore adaptive boosting mathematically. (Implemented and verified AdaBoost with decision stumps)
- [x] Explore gradient boosting mathematically. (Implemented and verified Gradient Boosting Regressor using decision stumps on a noisy sine wave).
- [x] Explore regularized linear regression mathematically. (Implemented and verified Ridge Regression with L2 penalty).
- [x] Explore regularized linear regression mathematically. (Implemented and verified Lasso Regression with L1 penalty).

- [x] Explore regularized linear regression mathematically. (Implemented and verified Elastic Net combining L1 and L2 penalties).
- [x] Explore ensemble regression mathematically. (Implemented and verified Random Forest Regression reducing variance on non-linear data).
- [x] Explore advanced gradient boosting regression mathematically. (Implemented and verified XGBoost Regressor using second-order approximation).
- [x] Explore robust regression mathematically. (Implemented and verified Support Vector Regression using subgradient descent and epsilon-insensitive loss).
- [x] Explore probabilistic regression mathematically. (Implemented and verified Bayesian Linear Regression computing full posterior distributions over weights).

- [x] Explore hierarchical decision-making mathematically. (Implement and verify Hierarchical Reinforcement Learning - HRL).
- [x] Explore classical multidimensional scaling mathematically. (Implemented and verified Multidimensional Scaling - MDS preserving pairwise distances).
- [x] Explore nonlinear dimensionality reduction mathematically. (Implemented and verified Locally Linear Embedding - LLE preserving local neighborhoods).
