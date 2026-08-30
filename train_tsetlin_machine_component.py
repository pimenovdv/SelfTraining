import numpy as np

class TsetlinMachine:
    def __init__(self, n_features, n_clauses, T, s):
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.T = T
        self.s = s
        # Number of states per TA. Positive states include, negative states exclude.
        # Initialize randomly near the middle
        self.ta_state = np.random.choice([-1, 1], size=(n_clauses, 2 * n_features))
        self.clause_sign = np.ones(n_clauses)
        self.clause_sign[1::2] = -1  # Alternate signs for binary classification

    def transform(self, X):
        X_neg = 1 - X
        X_literals = np.concatenate([X, X_neg], axis=1)

        clause_outputs = np.zeros((X.shape[0], self.n_clauses))
        for j in range(self.n_clauses):
            # Include literals where state >= 0
            include_literals = self.ta_state[j] >= 0

            for i in range(X.shape[0]):
                if np.sum(include_literals) == 0:
                    # By default if no literals included, clause outputs 1 (some variants say 0, let's use 1 to allow learning)
                    # For XOR it's better to evaluate to 1
                    clause_outputs[i, j] = 1
                else:
                    # If all included literals are 1 in X, then clause is 1
                    clause_outputs[i, j] = np.all(X_literals[i, include_literals] == 1).astype(int)

        return clause_outputs, X_literals

    def predict(self, X):
        clause_outputs, _ = self.transform(X)
        class_sum = np.dot(clause_outputs, self.clause_sign)
        return (class_sum >= 0).astype(int)

    def update(self, X, y):
        clause_outputs, X_literals = self.transform(X)
        class_sum = np.dot(clause_outputs, self.clause_sign)
        class_sum = np.clip(class_sum, -self.T, self.T)

        # Calculate feedback probabilities
        prob_type_i = (self.T - class_sum) / (2 * self.T)
        prob_type_ii = (self.T + class_sum) / (2 * self.T)

        for i in range(X.shape[0]):
            for j in range(self.n_clauses):
                if y[i] == 1 and self.clause_sign[j] > 0:
                    if np.random.rand() < prob_type_i[i]:
                        self._type_i_feedback(j, X_literals[i], clause_outputs[i, j])
                elif y[i] == 1 and self.clause_sign[j] < 0:
                    if np.random.rand() < prob_type_ii[i]:
                        self._type_ii_feedback(j, X_literals[i], clause_outputs[i, j])
                elif y[i] == 0 and self.clause_sign[j] > 0:
                    if np.random.rand() < prob_type_ii[i]:
                        self._type_ii_feedback(j, X_literals[i], clause_outputs[i, j])
                elif y[i] == 0 and self.clause_sign[j] < 0:
                    if np.random.rand() < prob_type_i[i]:
                        self._type_i_feedback(j, X_literals[i], clause_outputs[i, j])

    def _type_i_feedback(self, j, X_literal, clause_output):
        if clause_output == 1:
            for k in range(2 * self.n_features):
                if X_literal[k] == 1:
                    # Reward include
                    if np.random.rand() < (self.s - 1.0) / self.s:
                        self.ta_state[j, k] += 1
                else:
                    # Penalize exclude
                    if np.random.rand() < 1.0 / self.s:
                        self.ta_state[j, k] -= 1
        else:
            for k in range(2 * self.n_features):
                # Penalize include (push to exclude)
                if np.random.rand() < 1.0 / self.s:
                    self.ta_state[j, k] -= 1

    def _type_ii_feedback(self, j, X_literal, clause_output):
        if clause_output == 1:
            for k in range(2 * self.n_features):
                if X_literal[k] == 0:
                    # Push towards include so clause will be 0 next time
                    self.ta_state[j, k] += 1

def main():
    np.random.seed(42)

    # Simple XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Train Tsetlin Machine
    tm = TsetlinMachine(n_features=2, n_clauses=20, T=10, s=2.0)

    epochs = 500
    for epoch in range(epochs):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        tm.update(X_shuffled, y_shuffled)

        if epoch % 50 == 0:
            preds = tm.predict(X)
            acc = np.mean(preds == y)
            print(f"Epoch {epoch}, Accuracy: {acc}")

    preds = tm.predict(X)
    acc = np.mean(preds == y)
    print(f"Final Accuracy: {acc}")

    if acc == 1.0:
        print("Success: Tsetlin Machine learned XOR function perfectly.")
    else:
        print("Failure: Tsetlin Machine could not learn XOR.")

if __name__ == "__main__":
    main()
