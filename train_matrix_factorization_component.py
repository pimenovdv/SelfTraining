import numpy as np

class MatrixFactorization:
    def __init__(self, n_factors=5, learning_rate=0.01, reg=0.02, n_epochs=50):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs

    def fit(self, R):
        n_users, n_items = R.shape
        self.P = np.random.normal(scale=1./self.n_factors, size=(n_users, self.n_factors))
        self.Q = np.random.normal(scale=1./self.n_factors, size=(n_items, self.n_factors))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.b = np.mean(R[np.where(R != 0)])

        samples = [
            (i, j, R[i, j])
            for i in range(n_users)
            for j in range(n_items)
            if R[i, j] > 0
        ]

        for epoch in range(self.n_epochs):
            np.random.shuffle(samples)
            self._sgd(samples)
            if (epoch + 1) % 10 == 0:
                mse = self._mse(samples)
                print(f"Epoch: {epoch + 1} ; error = {mse:.4f}")

    def _sgd(self, samples):
        for i, j, r in samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            self.b_u[i] += self.learning_rate * (e - self.reg * self.b_u[i])
            self.b_i[j] += self.learning_rate * (e - self.reg * self.b_i[j])

            P_i = self.P[i, :][:]

            self.P[i, :] += self.learning_rate * (e * self.Q[j, :] - self.reg * self.P[i,:])
            self.Q[j, :] += self.learning_rate * (e * P_i - self.reg * self.Q[j,:])

    def get_rating(self, i, j):
        return self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)

    def _mse(self, samples):
        error = 0
        for i, j, r in samples:
            prediction = self.get_rating(i, j)
            error += (r - prediction) ** 2
        return error / len(samples)

    def full_matrix(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :] + self.P.dot(self.Q.T)

if __name__ == "__main__":
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])
    mf = MatrixFactorization(n_factors=2, n_epochs=200)
    mf.fit(R)
    print("Original matrix:")
    print(R)
    print("Reconstructed matrix:")
    print(np.round(mf.full_matrix(), 2))
