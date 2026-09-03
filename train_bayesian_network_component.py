import numpy as np

def train_bayesian_network():
    print("Training Bayesian Network (Naive Bayes structure) component...")

    # Simple Naive Bayes for binary classification
    np.random.seed(42)
    # Generate synthetic data
    # Features: [f1, f2]
    # Class 0: f1 usually 0, f2 usually 0
    # Class 1: f1 usually 1, f2 usually 1

    X = []
    y = []

    for _ in range(500):
        c = np.random.binomial(1, 0.5)
        if c == 0:
            f1 = np.random.binomial(1, 0.2)
            f2 = np.random.binomial(1, 0.3)
        else:
            f1 = np.random.binomial(1, 0.8)
            f2 = np.random.binomial(1, 0.7)
        X.append([f1, f2])
        y.append(c)

    X = np.array(X)
    y = np.array(y)

    # Calculate priors
    p_y1 = np.mean(y == 1)
    p_y0 = 1 - p_y1

    # Calculate likelihoods (Conditional probabilities)
    # P(f_i=1 | y)
    p_f1_y0 = np.mean(X[y == 0, 0] == 1)
    p_f2_y0 = np.mean(X[y == 0, 1] == 1)

    p_f1_y1 = np.mean(X[y == 1, 0] == 1)
    p_f2_y1 = np.mean(X[y == 1, 1] == 1)

    print(f"Priors: P(Y=0)={p_y0:.3f}, P(Y=1)={p_y1:.3f}")
    print(f"Likelihoods Y=0: P(F1=1|Y=0)={p_f1_y0:.3f}, P(F2=1|Y=0)={p_f2_y0:.3f}")
    print(f"Likelihoods Y=1: P(F1=1|Y=1)={p_f1_y1:.3f}, P(F2=1|Y=1)={p_f2_y1:.3f}")

    # Inference on a new sample
    sample = np.array([1, 1])
    print(f"Inferring class for sample {sample}...")

    # P(Y=0 | F1=1, F2=1) \propto P(Y=0) * P(F1=1|Y=0) * P(F2=1|Y=0)
    prob_y0 = p_y0 * p_f1_y0 * p_f2_y0

    # P(Y=1 | F1=1, F2=1) \propto P(Y=1) * P(F1=1|Y=1) * P(F2=1|Y=1)
    prob_y1 = p_y1 * p_f1_y1 * p_f2_y1

    norm = prob_y0 + prob_y1
    p_y0_given_x = prob_y0 / norm
    p_y1_given_x = prob_y1 / norm

    print(f"P(Y=0|X) = {p_y0_given_x:.4f}")
    print(f"P(Y=1|X) = {p_y1_given_x:.4f}")

    # Evaluate accuracy on training set
    preds = []
    for x in X:
        p0 = p_y0 * (p_f1_y0 if x[0]==1 else (1-p_f1_y0)) * (p_f2_y0 if x[1]==1 else (1-p_f2_y0))
        p1 = p_y1 * (p_f1_y1 if x[0]==1 else (1-p_f1_y1)) * (p_f2_y1 if x[1]==1 else (1-p_f2_y1))
        preds.append(1 if p1 > p0 else 0)

    accuracy = np.mean(preds == y)
    print(f"Training Accuracy: {accuracy:.4f}")
    assert accuracy > 0.7, "Accuracy should be reasonably high."
    print("Bayesian Network (Naive Bayes) component trained and evaluated successfully.")

if __name__ == "__main__":
    train_bayesian_network()
