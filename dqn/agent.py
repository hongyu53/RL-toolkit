import torch
import torch.nn as nn


class Agent:
    class Network(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(Agent.Network, self).__init__()
            self.fc1 = nn.Linear(state_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, action_dim)

        def forward(self, state):
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            actions_value = self.fc3(x)
            return actions_value

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
        # networks
        self.dqn = Agent.Network(state_dim, action_dim)
        self.dqn_target = Agent.Network(state_dim, action_dim)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_optimizer = torch.optim.Adam(self.dqn.parameters(), lr)
        # hyperparameters
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
                return torch.argmax(self.dqn(state)).item()
        # epsilon-greedy policy
        if torch.rand(1) > self.epsilon:
            with torch.no_grad():
                action = torch.argmax(self.dqn(state)).item()
        else:
            action = torch.randint(0, self.action_dim, (1,)).item()
        return action

    def train(self, training_set, num_epoch):
        # update target
        if self.learning_step_counter % self.update_interval == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.learning_step_counter += 1
        # load training set
        for _ in range(num_epoch):
            s, a, s_, r, done = training_set
            s = torch.tensor(s, dtype=torch.float32)
            a = torch.tensor(a, dtype=torch.int64)
            s_ = torch.tensor(s_, dtype=torch.float32)
            r = torch.tensor(r, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)
            # q-value evaluation
            q_eval = self.dqn(s).gather(1, a.unsqueeze(1)).squeeze()
            q_next = self.dqn_target(s_).max(1)[0].squeeze()
            q_target = r + self.gamma * q_next * (1 - done)
            loss = nn.functional.mse_loss(q_eval, q_target)
            # update
            self.dqn_optimizer.zero_grad()
            loss.backward()
            self.dqn_optimizer.step()

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        torch.save(self.dqn.state_dict(), "./dqn/model/dqn.pt")

    def load(self):
        self.dqn.load_state_dict(torch.load("./dqn/model/dqn.pt"))
        self.dqn.eval()
