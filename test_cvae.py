import numpy as np
from train_cvae_component import CVAE

def test_cvae():
    np.random.seed(42)
    X = np.random.randn(10, 4)
    Y = np.random.randn(10, 4)
    cvae = CVAE(input_dim=4, cond_dim=4, hidden_dim=16, latent_dim=2)
    Out = cvae.forward(X, Y)

    assert Out.shape == (10, 4)
    assert not np.isnan(Out).any()

    grads = cvae.backward(X, Y)
    assert 'W_e1' in grads
    assert not np.isnan(grads['W_e1']).any()
    print("test_cvae passed")

if __name__ == "__main__":
    test_cvae()
