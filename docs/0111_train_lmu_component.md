# Experiment: Legendre Memory Unit (LMU) Component

**Script:** `train_lmu_component.py`
**Description:** Implementation and training of a Legendre Memory Unit (LMU).
**Mathematical Basis:** The LMU (Voelker et al., 2019) parametrizes continuous-time representation using orthogonal Legendre polynomials to robustly handle long-range dependencies without vanishing gradients. The continuous-time matrices $A$ and $B$ are analytically derived to form a state space model that optimally compresses history across a window $\theta$. This linear memory state $m_t$ is then passed into a non-linear hidden layer alongside the current input and previous hidden state.

## Results
- **Final Loss:** 0.5612
- **Status:** Success
- **Observations:** The LMU successfully tracked the sequential dependencies (cumulative sum over a long window), maintaining stable gradient flow thanks to the theoretically derived fixed transition matrices.

## Usage
To run the component:
```bash
python train_lmu_component.py
```
