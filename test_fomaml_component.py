import numpy as np
from train_fomaml_component import train_fomaml

def test_fomaml():
    np.random.seed(42)
    final_theta, final_loss = train_fomaml(meta_epochs=100)
    assert final_loss < 2.5
    print("FOMAML test passed.")

if __name__ == "__main__":
    test_fomaml()
