import torch
import torch.nn as nn
import json
import os

class AdaptiveLogSoftmaxComponent(nn.Module):
    """
    Adaptive Log Softmax Component.
    This component calculates the adaptive log softmax, which is an efficient
    way to compute softmax over a very large vocabulary.
    """
    def __init__(self, in_features, n_classes, cutoffs):
        super().__init__()
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=in_features,
            n_classes=n_classes,
            cutoffs=cutoffs
        )

    def forward(self, hidden, target):
        return self.adaptive_log_softmax(hidden, target)

def test_component():
    in_features = 128
    n_classes = 10000
    cutoffs = [100, 1000, 5000]

    component = AdaptiveLogSoftmaxComponent(in_features, n_classes, cutoffs)

    batch_size = 32
    hidden = torch.randn(batch_size, in_features)
    target = torch.randint(0, n_classes, (batch_size,))

    output = component(hidden, target)

    print(f"Loss: {output.loss.item()}")
    print(f"Log probs shape: {component.adaptive_log_softmax.log_prob(hidden).shape}")

    os.makedirs('results', exist_ok=True)
    with open('results/adaptive_log_softmax_results.json', 'w') as f:
        json.dump({"loss": output.loss.item()}, f)

if __name__ == '__main__':
    test_component()
