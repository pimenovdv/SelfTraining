import numpy as np
import json
import os

def compute_reward(achieved_goal, desired_goal):
    distance = np.linalg.norm(achieved_goal - desired_goal)
    return 0.0 if distance < 0.1 else -1.0

def run_her_demonstration():
    """
    Demonstrates Hindsight Experience Replay (HER) mathematically.
    By augmenting transitions with achieved goals from the same trajectory,
    we can learn from failures even when the original goal was not reached.
    """
    np.random.seed(42)

    # A simple trajectory where state is the achieved goal.
    trajectory = [
        (np.array([0.0, 0.0]), 1, np.array([0.0, 1.0])),
        (np.array([0.0, 1.0]), 2, np.array([1.0, 1.0])),
        (np.array([1.0, 1.0]), 1, np.array([1.0, 2.0])),
        (np.array([1.0, 2.0]), 2, np.array([2.0, 2.0])),
    ]
    original_goal = np.array([5.0, 5.0])

    original_transitions = []
    for s, a, ns in trajectory:
        r = compute_reward(ns, original_goal)
        original_transitions.append({
            "state": s.tolist(),
            "action": a,
            "next_state": ns.tolist(),
            "goal": original_goal.tolist(),
            "reward": r
        })

    her_transitions = []
    # Use "future" strategy: sample a state later in the episode as the goal
    # For simplicity, we just use the final achieved state of the episode
    final_achieved_goal = trajectory[-1][2]

    for s, a, ns in trajectory:
        r = compute_reward(ns, final_achieved_goal)
        her_transitions.append({
            "state": s.tolist(),
            "action": a,
            "next_state": ns.tolist(),
            "goal": final_achieved_goal.tolist(),
            "reward": r
        })

    print("Original Transitions (always -1 reward):")
    for t in original_transitions:
        print(f"Goal: {t['goal']}, Achieved: {t['next_state']}, Reward: {t['reward']}")

    print("\nHER Transitions (final transition gets 0 reward):")
    for t in her_transitions:
        print(f"Goal: {t['goal']}, Achieved: {t['next_state']}, Reward: {t['reward']}")

    results = {
        "original": original_transitions,
        "her": her_transitions
    }

    os.makedirs("results", exist_ok=True)
    with open("results/her_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to results/her_results.json")

if __name__ == "__main__":
    run_her_demonstration()
