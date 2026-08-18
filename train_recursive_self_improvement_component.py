import numpy as np

class RecursiveSelfImprovementComponent:
    def __init__(self, initial_capability, alignment_threshold):
        self.capability = initial_capability
        self.alignment_score = 1.0 # 1.0 is perfectly aligned
        self.alignment_threshold = alignment_threshold

    def generate_improvement_proposal(self):
        """
        Simulate the model proposing an architecture or weight update
        that increases capability.
        """
        capability_gain = np.random.uniform(0.01, 0.1) * self.capability
        # With higher capability, proposals might become slightly more misaligned
        alignment_penalty = np.random.uniform(0.0, 0.05) * (self.capability / 10.0)
        return capability_gain, alignment_penalty

    def evaluate_proposal(self, capability_gain, alignment_penalty):
        """
        Simulate an automated alignment checker evaluating the proposal.
        """
        projected_alignment = self.alignment_score - alignment_penalty
        return projected_alignment >= self.alignment_threshold

    def run_self_improvement_loop(self, num_iterations):
        history = []
        for i in range(num_iterations):
            gain, penalty = self.generate_improvement_proposal()
            if self.evaluate_proposal(gain, penalty):
                self.capability += gain
                self.alignment_score -= penalty
                history.append((i, self.capability, self.alignment_score, True))
            else:
                history.append((i, self.capability, self.alignment_score, False))
        return history

if __name__ == "__main__":
    np.random.seed(42)
    rsi = RecursiveSelfImprovementComponent(initial_capability=1.0, alignment_threshold=0.8)
    history = rsi.run_self_improvement_loop(50)

    final_capability = history[-1][1]
    final_alignment = history[-1][2]

    print(f"Final Capability: {final_capability:.4f}")
    print(f"Final Alignment Score: {final_alignment:.4f}")

    assert final_capability > 1.0, "Capability did not increase."
    assert final_alignment >= rsi.alignment_threshold, "Alignment constraint violated."
    print("Recursive Self-Improvement simulation successful.")
