import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.float32)))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_masks=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        self.net = nn.Sequential(
            MaskedLinear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            MaskedLinear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            MaskedLinear(hidden_sizes[1], input_size)
        )
        self.m = {}
        self.create_mask()

    def create_mask(self):
        L = len(self.hidden_sizes)

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.input_size)
        for l in range(L):
            self.m[l] = np.random.randint(self.m[l-1].min(), self.input_size - 1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l-1][None, :] <= self.m[l][:, None] for l in range(L)]
        masks.append(self.m[L-1][None, :] < self.m[-1][:, None])

        # set the masks in all MaskedLinear layers
        layers = [m for m in self.net.modules() if isinstance(m, MaskedLinear)]
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)

    def forward(self, x):
        return self.net(x)

def train_made():
    print("Initializing MADE model...")
    # Generate some simple data: parity of binary vectors
    N = 1000
    D = 5
    data = np.random.randint(0, 2, size=(N, D)).astype(np.float32)
    data = torch.from_numpy(data)

    model = MADE(input_size=D, hidden_sizes=[32, 32])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 200
    losses = []

    print("Training MADE...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)

        # Binary cross entropy
        loss = F.binary_cross_entropy_with_logits(out, data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    print("Training complete.")

    plt.figure()
    plt.plot(losses)
    plt.title('MADE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.savefig('made_loss.png')
    print("Saved loss plot to made_loss.png")

if __name__ == "__main__":
    train_made()
