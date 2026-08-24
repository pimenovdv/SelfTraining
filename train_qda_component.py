import numpy as np

def qda_fit(X, y):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    priors = np.zeros(n_classes)
    means = np.zeros((n_classes, n_features))
    covariances = []

    for idx, c in enumerate(classes):
        X_c = X[y == c]
        priors[idx] = X_c.shape[0] / n_samples
        means[idx, :] = np.mean(X_c, axis=0)
        covariances.append(np.cov(X_c, rowvar=False) + np.eye(n_features) * 1e-4) # Regularization

    return priors, means, covariances, classes

def qda_predict(X, priors, means, covariances, classes):
    n_samples = X.shape[0]
    n_classes = len(classes)
    log_probs = np.zeros((n_samples, n_classes))

    for idx, c in enumerate(classes):
        prior = priors[idx]
        mean = means[idx]
        cov = covariances[idx]

        inv_cov = np.linalg.inv(cov)
        sign, log_det = np.linalg.slogdet(cov)
        det_term = sign * log_det

        diff = X - mean
        for i in range(n_samples):
            mahalanobis = diff[i].T @ inv_cov @ diff[i]
            log_probs[i, idx] = -0.5 * det_term - 0.5 * mahalanobis + np.log(prior)

    return classes[np.argmax(log_probs, axis=1)]

def main():
    print("Testing Quadratic Discriminant Analysis (QDA) Component...")
    np.random.seed(42)
    X1 = np.random.randn(50, 2) * 0.5 + np.array([2, 2])
    X2 = np.random.randn(50, 2) * 1.5 + np.array([-2, -2])

    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(50), np.ones(50)])

    priors, means, covariances, classes = qda_fit(X, y)
    preds = qda_predict(X, priors, means, covariances, classes)

    accuracy = np.mean(preds == y)
    print(f"QDA Accuracy: {accuracy * 100:.2f}%")

    if accuracy > 0.9:
        print("Success: QDA component verified successfully.")
    else:
        print("Failure: QDA component accuracy is too low.")

if __name__ == "__main__":
    main()
