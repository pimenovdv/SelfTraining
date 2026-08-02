# 0075_train_ltc_component

## Status
Success

## Component
Liquid Time-Constant (LTC) Network

## Description
Implemented and evaluated a Liquid Time-Constant (LTC) Network component using pure NumPy. This component tests the capacity of dynamically adapting continuous-time ODE dynamics by varying the time constant based on input. Successfully optimized the dynamics and read-out layer via manual backpropagation.

## Results
- Final Test MSE: 0.248228

The model successfully learned the sequential task, verifying gradient flow through the adaptive ODE step.

**Script:** `train_ltc_component.py`
