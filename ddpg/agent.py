import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent:
    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, action_bound):
            super(Agent.Actor, self).__init__()
            self.fc1 = nn.Linear(state_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, action_dim)
            self.action_bound = action_bound

        def forward(self, state):
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            return x * self.action_bound

    class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(Agent.Critic, self).__init__()
            self.fc1 = nn.Linear(state_dim + action_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 1)

        def forward(self, state, action):
            x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    def __init__(self, state_dim, action_dim, action_bound, lr, gamma, tau, noise_std):
        # Define actor and critic networks
        self.actor = Agent.Actor(state_dim, action_dim, action_bound)
        self.actor_target = Agent.Actor(state_dim, action_dim, action_bound)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic = Agent.Critic(state_dim, action_dim)
        self.critic_target = Agent.Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)

        self.action_bound = action_bound
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau  # soft update parameter
        self.noise_std = noise_std

    def select_action(self, state, test=False):
        if test:
            with torch.no_grad():
                return np.array([self.actor(state).item()])
        action = self.actor(state).item()
        noise = np.random.normal(0, self.noise_std, self.action_dim)
        return np.array([(action + noise).clip(-self.action_bound, self.action_bound)])

    def soft_update(self, target, source):
        for param_target, param_source in zip(target.parameters(), source.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param_source.data * self.tau
            )

    def train(self, training_set, epochs):
        for _ in range(epochs):
            # load training set
            s, a, s_, r, done = training_set
            s = torch.tensor(s, dtype=torch.float32)
            a = torch.tensor(a, dtype=torch.float32)
            s_ = torch.tensor(s_, dtype=torch.float32)
            r = torch.tensor(r, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)
            # compute critic loss
            target_Q = (
                r
                + self.gamma
                * (1 - done)
                * self.critic_target(s_, self.actor_target(s_)).squeeze().detach()
            )
            current_Q = self.critic(s, a).squeeze()
            critic_loss = F.mse_loss(current_Q, target_Q)
            # update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            # compute actor loss & update actor
            actor_loss = -self.critic(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # soft update target networks
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)

    def save(self):
        torch.save(self.actor.state_dict(), "./ddpg/model/actor.pth")
        torch.save(self.critic.state_dict(), "./ddpg/model/critic.pth")

    def load(self):
        self.actor.load_state_dict(torch.load("./ddpg/model/actor.pth"))
        self.critic.load_state_dict(torch.load("./ddpg/model/critic.pth"))
