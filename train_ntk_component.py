import torch
import torch.nn as nn
import torch.optim as optim

class NeuralTangentKernel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralTangentKernel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def compute_ntk(model, x1, x2):
    # Compute the Empirical Neural Tangent Kernel between two points x1 and x2
    model.zero_grad()

    # Compute outputs for x1
    out1 = model(x1)
    grads1 = []
    for o in out1.view(-1):
        model.zero_grad()
        o.backward(retain_graph=True)
        grads_o = [p.grad.view(-1).clone() for p in model.parameters() if p.grad is not None]
        if grads_o:
            grads1.append(torch.cat(grads_o))

    # Compute outputs for x2
    out2 = model(x2)
    grads2 = []
    for o in out2.view(-1):
        model.zero_grad()
        o.backward(retain_graph=True)
        grads_o = [p.grad.view(-1).clone() for p in model.parameters() if p.grad is not None]
        if grads_o:
            grads2.append(torch.cat(grads_o))

    # Compute NTK matrix
    ntk_matrix = torch.zeros((len(grads1), len(grads2)))
    for i, g1 in enumerate(grads1):
        for j, g2 in enumerate(grads2):
            ntk_matrix[i, j] = torch.dot(g1, g2)

    return ntk_matrix

def test_ntk():
    torch.manual_seed(42)
    input_dim = 10
    hidden_dim = 20
    model = NeuralTangentKernel(input_dim, hidden_dim)

    # Random input points
    x1 = torch.randn(2, input_dim)
    x2 = torch.randn(2, input_dim)

    ntk_val = compute_ntk(model, x1, x2)
    print("NTK Matrix computed:")
    print(ntk_val)
    assert ntk_val.shape == (2, 2)
    print("Test passed. Output shape is correct.")

if __name__ == '__main__':
    test_ntk()
