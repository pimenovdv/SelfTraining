import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, data):
        """Train using Hebbian learning rule."""
        num_patterns = len(data)
        for pattern in data:
            # Outer product of the pattern
            self.weights += np.outer(pattern, pattern)

        # Diagonal must be zero (no self-connections)
        np.fill_diagonal(self.weights, 0)

        # Normalize weights
        self.weights /= num_patterns

    def predict(self, pattern, steps=10, synchronous=False):
        """Recall a pattern."""
        state = np.copy(pattern)

        for _ in range(steps):
            if synchronous:
                # Update all neurons simultaneously
                state = np.sign(np.dot(self.weights, state))
                # Map exactly 0 to 1
                state[state == 0] = 1
            else:
                # Update neurons asynchronously
                order = np.random.permutation(self.size)
                for i in order:
                    activation = np.dot(self.weights[i], state)
                    state[i] = 1 if activation >= 0 else -1

        return state

def main():
    print("--- Training Hopfield Network Component ---")

    # 5x5 images flattened to 25 vector

    # Pattern 1: 'X'
    pattern_x = np.array([
        1, -1, -1, -1, 1,
        -1, 1, -1, 1, -1,
        -1, -1, 1, -1, -1,
        -1, 1, -1, 1, -1,
        1, -1, -1, -1, 1
    ])

    # Pattern 2: 'O'
    pattern_o = np.array([
        -1, 1, 1, 1, -1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        -1, 1, 1, 1, -1
    ])

    # Train Hopfield Network
    net = HopfieldNetwork(size=25)
    net.train([pattern_x, pattern_o])

    # Corrupt pattern 'X' (flip a few bits)
    corrupted_x = np.copy(pattern_x)
    corrupted_x[0:5] = [-1, -1, 1, 1, -1] # Corrupt top row

    print("Corrupted 'X':")
    print(corrupted_x.reshape(5,5))

    # Recall
    recalled_x = net.predict(corrupted_x, steps=5)

    print("\nRecalled 'X':")
    print(recalled_x.reshape(5,5))

    # Verify exact match
    success = np.array_equal(recalled_x, pattern_x)

    if success:
        print("\nHopfield Network successfully recalled the pattern!")
    else:
        print("\nHopfield Network failed to recall the pattern.")

if __name__ == "__main__":
    main()
