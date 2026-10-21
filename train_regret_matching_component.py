import numpy as np

def train_regret_matching_component():
    print("Testing Regret Matching Component...")

    # Actions: 0 = Rock, 1 = Paper, 2 = Scissors
    NUM_ACTIONS = 3
    utilities = np.array([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ])

    regret_sum_p1 = np.zeros(NUM_ACTIONS)
    regret_sum_p2 = np.zeros(NUM_ACTIONS)
    strategy_sum_p1 = np.zeros(NUM_ACTIONS)
    strategy_sum_p2 = np.zeros(NUM_ACTIONS)

    def get_strategy(regrets):
        strategy = np.maximum(regrets, 0)
        norm_sum = np.sum(strategy)
        if norm_sum > 0:
            strategy /= norm_sum
        else:
            strategy = np.ones(NUM_ACTIONS) / NUM_ACTIONS
        return strategy

    epochs = 100000
    for _ in range(epochs):
        strategy_p1 = get_strategy(regret_sum_p1)
        strategy_p2 = get_strategy(regret_sum_p2)

        strategy_sum_p1 += strategy_p1
        strategy_sum_p2 += strategy_p2

        a1 = np.random.choice(NUM_ACTIONS, p=strategy_p1)
        a2 = np.random.choice(NUM_ACTIONS, p=strategy_p2)

        action_utilities_p1 = utilities[:, a2]
        action_utilities_p2 = -utilities[a1, :]

        regret_sum_p1 += action_utilities_p1 - utilities[a1, a2]
        regret_sum_p2 += action_utilities_p2 - (-utilities[a1, a2])

    avg_strategy_p1 = strategy_sum_p1 / np.sum(strategy_sum_p1)
    avg_strategy_p2 = strategy_sum_p2 / np.sum(strategy_sum_p2)

    print(f"Final Average Strategy Player 1: {avg_strategy_p1}")
    print(f"Final Average Strategy Player 2: {avg_strategy_p2}")

    nash = np.array([1/3, 1/3, 1/3])
    if np.allclose(avg_strategy_p1, nash, atol=0.05) and np.allclose(avg_strategy_p2, nash, atol=0.05):
        print("Success! Regret Matching successfully learned the Nash Equilibrium.")
    else:
        print("Failed to converge to Nash Equilibrium.")

if __name__ == "__main__":
    train_regret_matching_component()
