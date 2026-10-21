import numpy as np
import json
import os

class AdaIN:
    """
    Adaptive Instance Normalization (AdaIN) component.

    AdaIN aligns the mean and variance of the content features with those of the style features.
    It is widely used in style transfer models.

    Formula:
    AdaIN(x, y) = sigma(y) * ((x - mu(x)) / sigma(x)) + mu(y)
    """
    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, content, style):
        # Expected shape: (batch_size, channels, spatial_dim_1, spatial_dim_2, ...)
        # We will compute mean and std across spatial dimensions.
        # For a 1D sequence or flattened image: (batch_size, channels, sequence_length)

        # Calculate mean and variance for content
        content_mean = np.mean(content, axis=-1, keepdims=True)
        content_std = np.std(content, axis=-1, keepdims=True) + self.eps

        # Calculate mean and variance for style
        style_mean = np.mean(style, axis=-1, keepdims=True)
        style_std = np.std(style, axis=-1, keepdims=True) + self.eps

        # Normalize content and scale/shift with style
        normalized_content = (content - content_mean) / content_std
        adain_output = style_std * normalized_content + style_mean

        return adain_output

def run_adain_test():
    # Test parameters
    batch_size = 2
    channels = 3
    sequence_length = 10

    # Create random content and style tensors
    np.random.seed(42)
    content = np.random.randn(batch_size, channels, sequence_length)
    style = np.random.randn(batch_size, channels, sequence_length) * 2.0 + 5.0

    # Initialize AdaIN
    adain = AdaIN()

    # Apply AdaIN
    output = adain(content, style)

    # Verify that the output has the same mean and std as the style tensor
    output_mean = np.mean(output, axis=-1, keepdims=True)
    output_std = np.std(output, axis=-1, keepdims=True)

    style_mean = np.mean(style, axis=-1, keepdims=True)
    style_std = np.std(style, axis=-1, keepdims=True)

    mean_diff = float(np.max(np.abs(output_mean - style_mean)))
    std_diff = float(np.max(np.abs(output_std - style_std)))

    results = {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "success": bool(mean_diff < 1e-4 and std_diff < 1e-4)
    }

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/adain_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"AdaIN test complete. Success: {results['success']}, Mean Diff: {results['mean_diff']:.6f}, Std Diff: {results['std_diff']:.6f}")

if __name__ == "__main__":
    run_adain_test()
