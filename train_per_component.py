import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        max_prio = np.max(self.priorities) if self.buffer else 1.0
        prio = max_prio if error is None else (np.abs(error) + 1e-5) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []

        prios = self.priorities[:len(self.buffer)]
        probs = prios / prios.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_errors):
        for idx, error in zip(batch_indices, batch_errors):
            self.priorities[idx] = (np.abs(error) + 1e-5) ** self.alpha

def test_per():
    print("Initializing Prioritized Experience Replay Buffer mathematically...")
    buffer = PrioritizedReplayBuffer(capacity=10, alpha=0.6)

    print("Adding transitions with distinct TD-errors...")
    for i in range(5):
        # Adding (state, action, reward, next_state), passing error=i+1
        buffer.add(f"transition_{i}", error=i+1)

    print("Current Priorities (unnormalized):", buffer.priorities[:5])

    samples, indices, weights = buffer.sample(batch_size=3, beta=0.4)
    print("Sampled Indices:", indices)
    print("Importance Sampling Weights:", weights)

    print("Updating priorities for sampled indices...")
    new_errors = np.random.rand(3) * 10
    buffer.update_priorities(indices, new_errors)
    print("Updated Priorities (unnormalized):", buffer.priorities[:5])

    print("PER mathematical component verified successfully.")

if __name__ == '__main__':
    test_per()
