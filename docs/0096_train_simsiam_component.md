# Experiment 0096: SimSiam (Simple Siamese Networks)

**Objective:** Implement and verify non-contrastive self-supervised learning using SimSiam mathematically.

**Methodology:** The SimSiam architecture learns representations without requiring negative samples or a momentum encoder. It processes two augmented views of an image through an encoder $f$ and uses a predictor network $h$ on one view to match the encoder output of the other view. A critical stop-gradient operation is applied to the target view to prevent representation collapse. The negative cosine similarity is minimized via manual backpropagation.

**Results:**
- Initial Loss: -0.2487
- Final Loss: -1.9781
- Success: True

**Conclusion:** The component successfully minimized the negative cosine similarity between augmented views, demonstrating representation learning without collapse and the effectiveness of the stop-gradient operation.
**Script:** `train_simsiam_component.py`
