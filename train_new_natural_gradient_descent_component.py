import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

def natural_gradient_step(model, inputs, lr=0.01):
    # For a linear model f(x) = w^T x, the Fisher Information Matrix (FIM)
    # with respect to a Gaussian likelihood is proportional to E[x x^T]

    # Compute the empirical FIM
    batch_size = inputs.size(0)
    # inputs: [batch_size, 2]
    # E[x x^T]
    fim = torch.matmul(inputs.t(), inputs) / batch_size

    # Add a small value to the diagonal for stability
    fim = fim + 1e-5 * torch.eye(2)

    fim_inv = torch.inverse(fim)

    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'weight' in name:
                # FIM is [2, 2], grad is [1, 2]
                grad_nat = torch.matmul(param.grad, fim_inv)
                param.data -= lr * grad_nat
            elif 'bias' in name:
                # Simplified natural gradient step for bias
                param.data -= lr * param.grad

def main():
    print("Starting New Natural Gradient Descent training...")
    # Dummy data
    inputs = torch.randn(100, 2)
    labels = torch.randn(100, 1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100)

    model = SimpleModel()

    epochs = 100
    for epoch in range(epochs):
        for inputs_batch, labels_batch in dataloader:
            outputs = model(inputs_batch)
            loss = nn.MSELoss()(outputs, labels_batch)
            model.zero_grad()
            loss.backward()
            natural_gradient_step(model, inputs_batch)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("New Natural Gradient Descent training complete.")

if __name__ == "__main__":
    main()
