class Pipeline:
    """A simple sequential pipeline of modules."""
    def __init__(self, modules):
        self.modules = modules

    def forward(self, x):
        out = x
        for module in self.modules:
            out = module.forward(out)
        return out

    def backward(self, grad_output):
        grad = grad_output
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def get_params(self):
        params = []
        for module in self.modules:
            params.append(module.get_params())
        return params
