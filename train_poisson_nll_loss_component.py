import numpy as np

def poisson_nll_loss(input_tensor, target_tensor, log_input=True, eps=1e-8):
    if log_input:
        loss = np.exp(input_tensor) - target_tensor * input_tensor
    else:
        loss = input_tensor - target_tensor * np.log(input_tensor + eps)
    return np.mean(loss)

def test_component():
    np.random.seed(42)
    # Target counts (e.g. 0, 1, 2, ...)
    targets = np.random.poisson(lam=2.0, size=(100,))
    # Predictions (log rates)
    inputs = np.log(targets + 1.0) + np.random.normal(0, 0.1, size=(100,))

    loss = poisson_nll_loss(inputs, targets, log_input=True)
    print(f"Poisson NLL Loss: {loss:.4f}")
    assert loss > 0, "Loss should be positive"

if __name__ == "__main__":
    test_component()
