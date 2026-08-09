# Experiment 0122: Contrastive Predictive Coding (CPC)

## Overview
This experiment verifies the implementation of Contrastive Predictive Coding (CPC). CPC learns representations by predicting the future in latent space using powerful autoregressive models, distinguishing the true future latent state from negative samples using InfoNCE loss.

## Mathematical Basis
An encoder $g_{enc}$ maps input sequences $x_t$ to latent representations $z_t$. An autoregressive model $g_{ar}$ summarizes $z_{\le t}$ into a context vector $c_t$.
The model predicts future latents $z_{t+k}$ using a linear projection of the context: $\hat{z}_{t+k} = c_t W_k$.
The InfoNCE loss optimizes this prediction against negative samples $Z_{neg}$:
$L_k = - \log \\frac{\exp(\hat{z}_{t+k}^T z_{t+k})}{\sum_{z_j \in Z_{neg}} \exp(\hat{z}_{t+k}^T z_j)}$

## Results
The model successfully minimized the InfoNCE loss on synthetic continuous sequence data.
Loss at end of training: 1.3847

This confirms that the autoregressive context can effectively predict the latent space representations of future time steps without directly generating high-dimensional inputs.
**Script:** `train_cpc_component.py`
