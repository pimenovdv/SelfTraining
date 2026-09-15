import torch
import torch.nn as nn
import torch.optim as optim

class QLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Simulated quantized base weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # LoRA parameters
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.dropout = nn.Dropout(p=lora_dropout)

        # init lora_A with kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x):
        # Y = XW + X * A * B * scaling
        base_out = nn.functional.linear(x, self.weight)

        lora_out = self.dropout(x)
        lora_out = nn.functional.linear(lora_out, self.lora_A)
        lora_out = nn.functional.linear(lora_out, self.lora_B)
        lora_out = lora_out * self.scaling

        return base_out + lora_out

def test_component():
    print("Testing QLoRA component...")
    try:
        # Create dummy input
        batch_size = 4
        in_features = 32
        out_features = 16

        x = torch.randn(batch_size, in_features)

        qlora_layer = QLoRALinear(in_features, out_features)

        # Test forward pass
        out = qlora_layer(x)

        # Simple optimization test (only LoRA parameters should be updated)
        target = torch.randn_like(out)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(qlora_layer.parameters(), lr=0.01)

        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify base weights have no grad
        assert qlora_layer.weight.grad is None, "Base weights should not have gradients"
        # Verify LoRA weights have grad
        assert qlora_layer.lora_A.grad is not None, "LoRA A weights should have gradients"

        print("Loss calculation and gradient check successful:", loss.item())
        print("QLoRA component successfully evaluated.")

    except Exception as e:
        print(f"Error evaluating component: {e}")

if __name__ == "__main__":
    test_component()
