import numpy as np
import json
import os

def pmu(x, a=1.0):
    return x * np.tanh(np.log1p(np.exp(a * x)))

def pmu_derivative(x, a=1.0):
    softplus_ax = np.log1p(np.exp(a * x))
    tanh_softplus_ax = np.tanh(softplus_ax)
    sech_softplus_ax_sq = 1 - tanh_softplus_ax**2
    sigmoid_ax = 1 / (1 + np.exp(-a * x))
    return tanh_softplus_ax + x * sech_softplus_ax_sq * a * sigmoid_ax

def main():
    print("Testing PMU Activation Component...")
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = pmu(x)
    dy = pmu_derivative(x)
    print("x:", x)
    print("PMU(x):", y)
    print("PMU'(x):", dy)

    assert y.shape == x.shape
    assert dy.shape == x.shape
    print("PMU Component test passed.")

if __name__ == "__main__":
    main()
