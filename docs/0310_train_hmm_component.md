# Experiment 0310: Train Hidden Markov Model (HMM) Component

## Status
Success

## Description
Evaluates a Hidden Markov Model (HMM) component mathematically in pure NumPy, testing its ability to estimate hidden state transition and observation emission probabilities given only a sequence of observations using the Baum-Welch (Expectation-Maximization) algorithm.

## Results
- The model successfully learned parameters representing the underlying states, validating the implementation of the forward-backward algorithm and parameter updates.
- Match 0 distance: 1.7928
- Match 1 distance: 0.0831

**Script:** `train_hmm_component.py`
