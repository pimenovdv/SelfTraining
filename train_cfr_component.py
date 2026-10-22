import numpy as np

class Node:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)

    def get_strategy(self, realization_weight):
        strategy = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(strategy)
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            strategy = np.ones(self.num_actions) / self.num_actions

        self.strategy_sum += realization_weight * strategy
        return strategy

    def get_average_strategy(self):
        avg_strategy = np.copy(self.strategy_sum)
        normalizing_sum = np.sum(avg_strategy)
        if normalizing_sum > 0:
            avg_strategy /= normalizing_sum
        else:
            avg_strategy = np.ones(self.num_actions) / self.num_actions
        return avg_strategy

class KuhnTrainer:
    def __init__(self):
        self.node_map = {}

    def get_node(self, info_set):
        if info_set not in self.node_map:
            self.node_map[info_set] = Node(2) # PASS or BET
        return self.node_map[info_set]

    def cfr(self, cards, history, p0, p1):
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        # Return payoffs for terminal states
        if plays > 1:
            terminal_pass = history[plays - 1] == 'p'
            double_bet = history[plays - 2:plays] == 'bb'
            is_player_card_higher = cards[player] > cards[opponent]
            if terminal_pass:
                if history == 'pp':
                    return 1 if is_player_card_higher else -1
                else:
                    return 1
            elif double_bet:
                return 2 if is_player_card_higher else -2

        info_set = str(cards[player]) + history
        node = self.get_node(info_set)

        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = np.zeros(2)
        node_util = 0

        for a in range(2):
            next_history = history + ('p' if a == 0 else 'b')
            util[a] = -self.cfr(cards, next_history, p0 * strategy[a] if player == 0 else p0,
                                p1 * strategy[a] if player == 1 else p1)
            node_util += strategy[a] * util[a]

        for a in range(2):
            regret = util[a] - node_util
            node.regret_sum[a] += (p1 if player == 0 else p0) * regret

        return node_util

    def train(self, iterations):
        util = 0
        cards = [1, 2, 3] # J, Q, K
        for _ in range(iterations):
            np.random.shuffle(cards)
            util += self.cfr(cards[:2], "", 1, 1)
        print(f"Average game value: {util / iterations}")
        for info_set, node in sorted(self.node_map.items()):
            print(f"{info_set}: {node.get_average_strategy()}")

if __name__ == "__main__":
    trainer = KuhnTrainer()
    trainer.train(10000)
