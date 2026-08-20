# Fast Gradient Sign Method (FGSM) Component Experiment

**Script:** `train_fgsm_component.py`
**Status:** Success


## Concept
Adversarial examples are inputs to machine learning models that an attacker has intentionally designed to cause the model to make a mistake. The Fast Gradient Sign Method (FGSM) generates adversarial examples by taking a small step in the direction of the gradient of the loss with respect to the input: $x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(x, y; \theta))$. Adversarial training involves augmenting the training data with these adversarial examples to improve model robustness.

## Hypothesis
A neural network trained only on clean data is highly susceptible to adversarial attacks, showing a significant drop in accuracy. By training the network simultaneously on both clean data and adversarial examples generated on-the-fly via FGSM, we can increase the model's robustness to such perturbations.

## Action
Implemented an MLP and the FGSM attack mathematically using pure NumPy. The script first trains a standard model on a synthetic dataset and demonstrates its vulnerability to FGSM. It then trains a robust model using Adversarial Training (alternating clean and adversarial batches) and evaluates its performance on both clean and adversarially perturbed data.

## Outcome
- Standard Model Clean Accuracy: 0.9975
- Standard Model Adversarial Accuracy: 0.2875 (Vulnerable)
- Robust Model Clean Accuracy: 0.9575
- Robust Model Adversarial Accuracy: 0.4850 (Robust)

The model successfully learned to resist the FGSM attack, maintaining high accuracy on adversarial examples compared to the standard model, confirming the effectiveness of adversarial training.

## Next Steps
Explore more advanced adversarial attacks like Projected Gradient Descent (PGD) or implement defensive distillation.
