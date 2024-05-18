import numpy as np
import pandas as pd


class Agent:
    def __init__(self, action_space, observation_space, lr, gamma, epsilon):
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(
            np.zeros((len(observation_space), len(action_space))),
            index=[str(obs) for obs in observation_space],
            columns=[str(act) for act in action_space],
        )

    def epsilon_greedy(self, state):
        actions = self.q_table.loc[str(state), :]
        if np.random.rand() < self.epsilon or (actions == 0).all():
            return np.random.choice(self.action_space)
        else:
            action_idx = self.q_table.columns.get_loc(actions.idxmax())
            return self.action_space[action_idx]

    def greedy(self, state):
        action_idx = self.q_table.columns.get_loc(self.q_table.loc[str(state)].idxmax())
        return self.action_space[action_idx]

    def save(self):
        self.q_table.to_csv("./q_learning/model/q_table.csv")

    def load(self):
        self.q_table = pd.read_csv("./q_learning/model/q_table.csv", index_col=0)
