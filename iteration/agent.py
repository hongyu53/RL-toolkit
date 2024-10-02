import numpy as np
import pandas as pd


class Agent:
    def __init__(self, width, length, terminal_states, gamma, theta):
        self.width = width
        self.length = length
        self.terminate_states = terminal_states
        self.gamma = gamma
        self.theta = theta
        self.value_table = pd.DataFrame(np.zeros((width, length)))
        self.policy_table = pd.DataFrame(
            [
                [["↑", "→", "↓", "←"] for _ in range(self.length)]
                for _ in range(self.width)
            ]
        )
        for i, j in terminal_states:
            self.policy_table.loc[i, j] = []

    def calculate_new_value(self, i, j, gamma):
        if [i, j] in self.terminate_states:
            return 0
        next_values = []
        for action in self.policy_table.loc[i, j]:
            if action == "↑":
                next_state = [max(i - 1, 0), j]
            elif action == "→":
                next_state = [i, min(j + 1, self.length - 1)]
            elif action == "↓":
                next_state = [min(i + 1, self.width - 1), j]
            else:
                next_state = [i, max(j - 1, 0)]
            next_values.append(self.value_table.loc[next_state[0], next_state[1]])
        reward = -1.0
        return reward + gamma * max(next_values)

    def value_iteration(self):
        count = 0
        while True:
            delta = 0
            update_value_table = self.value_table.copy()
            for i in range(self.width):
                for j in range(self.length):
                    old_value = self.value_table.loc[i, j]
                    new_value = self.calculate_new_value(i, j, self.gamma)
                    update_value_table.loc[i, j] = new_value
                    delta = max(delta, np.abs(old_value - new_value))
            self.value_table = update_value_table
            count += 1
            if delta < self.theta:
                break
        self.generate_policy()
        print(f"Value Iteration took {count} iterations")

    def generate_policy(self):
        for i in range(self.width):
            for j in range(self.length):
                if [i, j] in self.terminate_states:
                    self.policy_table.loc[i, j] = []
                else:
                    state_action_map = {
                        "↑": self.value_table.loc[max(i - 1, 0), j],
                        "→": self.value_table.loc[i, min(j + 1, self.length - 1)],
                        "↓": self.value_table.loc[min(i + 1, self.width - 1), j],
                        "←": self.value_table.loc[i, max(j - 1, 0)],
                    }
                    best_value = max(state_action_map.values())
                    best_actions = [
                        action
                        for action, value in state_action_map.items()
                        if value == best_value
                    ]
                    self.policy_table.loc[i, j] = best_actions

    def policy_iteration(self):
        count = 0
        while True:
            delta = 0
            update_value_table = self.value_table.copy()
            for i in range(self.width):
                for j in range(self.length):
                    old_value = self.value_table.loc[i, j]
                    new_value = self.calculate_new_value(i, j, self.gamma)
                    update_value_table.loc[i, j] = new_value
                    delta = max(delta, np.abs(old_value - new_value))
            self.value_table = update_value_table
            self.generate_policy()
            count += 1
            if delta < self.theta:
                break
        print(f"Policy Iteration took {count} iterations")


if __name__ == "__main__":
    agent = Agent(6, 6, [[0, 1], [5, 5]], 0.99, 0.001)
    agent.policy_iteration()
