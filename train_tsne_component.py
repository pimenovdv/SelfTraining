import numpy as np

def compute_pairwise_distances(X):
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return np.maximum(D, 0)

def Hbeta(D, beta=1.0):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X, tol=1e-5, perplexity=30.0):
    (n, d) = X.shape
    D = compute_pairwise_distances(X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        H, thisP = Hbeta(Di, beta[i])

        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    return P

def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000):
    (n, d) = X.shape

    np.random.seed(42)
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    P = x2p(X, 1e-5, perplexity)
    P = P + P.T
    P = P / np.sum(P)
    P = P * 4.
    P = np.maximum(P, 1e-12)

    for iter in range(max_iter):
        sum_Y = np.sum(np.square(Y), 1)
        num = 1.0 / (1.0 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        if iter < 20:
            momentum = 0.5
        else:
            momentum = 0.8

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < 0.01] = 0.01

        iY = momentum * iY - 200.0 * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        if iter == 100:
            P = P / 4.

        if (iter + 1) % 100 == 0:
            C = np.sum(P * np.log(P / Q))
            print(f"Iteration {iter + 1}: error is {C}")

    return Y

if __name__ == "__main__":
    print("Running t-SNE component test...")
    np.random.seed(42)
    X1 = np.random.randn(50, 10) + np.array([5]*10)
    X2 = np.random.randn(50, 10) + np.array([-5]*10)
    X = np.vstack([X1, X2])

    print(f"Input shape: {X.shape}")
    Y = tsne(X, max_iter=500)
    print(f"Output shape: {Y.shape}")

    y1_mean = np.mean(Y[:50], axis=0)
    y2_mean = np.mean(Y[50:], axis=0)
    print(f"Dist between clusters: {np.linalg.norm(y1_mean - y2_mean)}")
    print(f"Variance within cluster 1: {np.var(Y[:50])}")
    print(f"Variance within cluster 2: {np.var(Y[50:])}")
