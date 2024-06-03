import torch
import torch.nn as nn


class Agent:
    class Network(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(Agent.Network, self).__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.state_value = nn.Linear(128, 1)
            self.advantage = nn.Linear(128, action_dim)

        def forward(self, state):
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            state_value = self.state_value(x)
            advantage = self.advantage(x)
            return state_value, advantage

    def __init__(
        self,
        state_dim,
        action_dim,
        lr,
        gamma,
        epsilon,
        epsilon_decay,
        epsilon_min,
        update_interval,
    ):
        self.eval_net = self.Network(state_dim, action_dim)
        self.target_net = self.Network(state_dim, action_dim)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_step_counter = 0
        self.update_interval = update_interval

    def select_action(self, state, test=False):
        if test:
            with torch.no_grad():
                _, advantage = self.eval_net(state)
                return torch.argmax(advantage).item()
        # epsilon-greedy
        if torch.rand(1) > self.epsilon:
            with torch.no_grad():
                _, advantage = self.eval_net(state)
                return torch.argmax(advantage).item()
        else:
            return torch.randint(0, self.action_dim, (1,)).item()

    def learn(self, training_set, num_epoch):
        # update target network
        if self.learning_step_counter % self.update_interval == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_step_counter += 1
        # load training set
        s, a, s_, r, d = training_set
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        s_ = torch.tensor(s_, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        for _ in range(num_epoch):
            state_value, advantage = self.eval_net(s)
            q_eval = (
                (state_value + advantage - advantage.mean())
                .gather(1, a.unsqueeze(1))
                .squeeze()
            )
