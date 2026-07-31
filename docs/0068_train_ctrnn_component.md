# 0068_train_ctrnn_component

## Status
Success

## Component
Continuous-Time Recurrent Neural Network (CTRNN)

## Description
Implemented a Continuous-Time Recurrent Neural Network (CTRNN) using pure NumPy. The model incorporates trainable time constants (tau) and uses Euler integration to discretize and simulate the continuous-time differential equations governing the hidden states. Backpropagation Through Time (BPTT) was manually derived and verified to correctly update weights, biases, and time constants.

## Results
- Final Test MSE: 0.054927

The CTRNN successfully learned a continuous moving average dynamic over sequential data, confirming the mathematical formulation of continuous state evolution and gradient updates.

**Script:** `train_ctrnn_component.py`
