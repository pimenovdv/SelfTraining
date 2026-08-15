from core.module import Module
from core.registry import Registry
from core.pipeline import Pipeline
import numpy as np

@Registry.register("linear")
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.params['W'] = np.random.randn(in_features, out_features) * 0.1
        self.params['b'] = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.params['W']) + self.params['b']

    def backward(self, grad_output):
        self.grads['W'] = np.dot(self.x.T, grad_output)
        self.grads['b'] = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.params['W'].T)

def test_registry():
    assert "linear" in Registry.list_modules()
    layer = Registry.get("linear", in_features=2, out_features=3)
    assert isinstance(layer, Linear)

def test_pipeline():
    layer1 = Registry.get("linear", in_features=2, out_features=3)
    layer2 = Registry.get("linear", in_features=3, out_features=1)
    pipeline = Pipeline([layer1, layer2])

    x = np.array([[1.0, 2.0]])
    out = pipeline.forward(x)
    assert out.shape == (1, 1)

    grad = pipeline.backward(np.array([[1.0]]))
    assert grad.shape == (1, 2)
    print("Core framework tests passed.")

if __name__ == "__main__":
    test_registry()
    test_pipeline()
