import torch
import torch.nn as nn
from ddqn.agent import Agent as DDQN_Agent


class Agent(DDQN_Agent):
    class Network_(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(Agent.Network_, self).__init__()
            self.fc1 = nn.Linear(state_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 1)
            self.fc4 = nn.Linear(128, action_dim)

        def forward(self, state):
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            value = self.fc3(x)
            advantage = self.fc4(x)
            q_value = value + advantage - advantage.mean()
            return q_value

    def __init__(
        self,
        action_dim,
        state_dim,
        lr,
        gamma,
        epsilon,
        epsilon_decay,
        epsilon_min,
        update_interval,
    ):
        super(Agent, self).__init__(
            action_dim,
            state_dim,
            lr,
            gamma,
            epsilon,
            epsilon_decay,
            epsilon_min,
            update_interval,
        )
        self.eval_net = Agent.Network_(state_dim, action_dim)
        self.target_net = Agent.Network_(state_dim, action_dim)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr)

    def save(self):
        torch.save(self.eval_net.state_dict(), "./d3qn/model/d3qn.pt")

    def load(self):
        self.eval_net.load_state_dict(torch.load("./d3qn/model/d3qn.pt"))
        self.eval_net.eval()
