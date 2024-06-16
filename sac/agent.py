import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent:
    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, action_bound):
            super(Agent.Actor, self).__init__()
            self.fc1 = nn.Linear(state_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, action_dim)
            self.fc4 = nn.Linear(256, action_dim)

            self.action_bound = action_bound
            self.log_std_min = -20
            self.log_std_max = 2

        def forward(self, state):
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            mean = self.fc3(x)
            log_std = self.fc4(x)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mean, std)

            z = normal.rsample()
            action = torch.tanh(z)
            action = action * self.action_bound
            log_prob = normal.log_prob(z) - torch.log(
                self.action_bound * (1 - (action / self.action_bound).pow(2) + 1e-6)
            )
            log_prob = log_prob.sum(1, keepdim=True)
            return action, log_prob

    class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(Agent.Critic, self).__init__()
            # Q1 architecture
            self.fc1 = nn.Linear(state_dim + action_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 1)
            # Q2 architecture
            self.fc4 = nn.Linear(state_dim + action_dim, 256)
            self.fc5 = nn.Linear(256, 256)
            self.fc6 = nn.Linear(256, 1)

        def forward(self, state, action):
            sa = torch.cat([state, action], 1)

            q1 = torch.relu(self.fc1(sa))
            q1 = torch.relu(self.fc2(q1))
            q1 = self.fc3(q1)

            q2 = torch.relu(self.fc4(sa))
            q2 = torch.relu(self.fc5(q2))
            q2 = self.fc6(q2)
            return q1, q2

    def __init__(self, state_dim, action_dim, action_bound, gamma, tau, alpha, lr):
        self.actor = Agent.Actor(state_dim, action_dim, action_bound)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = Agent.Critic(state_dim, action_dim)
        self.critic_target = Agent.Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.H_bar = torch.tensor([-action_dim], dtype=torch.float32)
        self.log_alpha = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state):
        action, _ = self.actor(state)
        return action.detach().numpy()

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def learn(self, training_set, num_epoch):
        for _ in range(num_epoch):
            s, a, s_, r, done = training_set
            s = torch.tensor(s, dtype=torch.float32)
            a = torch.tensor(a, dtype=torch.float32)
            s_ = torch.tensor(s_, dtype=torch.float32)
            r = torch.tensor(r, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)

            # Update critic
            with torch.no_grad():
                a_, log_prob = self.actor(s_)
                q1_target, q2_target = self.critic_target(s_, a_)
                q_target = torch.min(q1_target, q2_target)
                q_target = r + self.gamma * (1 - done) * (
                    q_target - self.alpha * log_prob
                )

            q1, q2 = self.critic(s, a)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update actor
            a, log_prob = self.actor(s)
            q1, q2 = self.critic(s, a)
            q = torch.min(q1, q2)
            actor_loss = (self.alpha * log_prob - q).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update alpha
            alpha_loss = -self.log_alpha * (log_prob + self.H_bar).detach().mean()

            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

            # Update target networks
            self.soft_update(self.critic_target, self.critic)

    def save(self):
        torch.save(self.actor.state_dict(), "./d3qn/model/actor.pth")
        torch.save(self.critic.state_dict(), "./d3qn/model/critic.pth")

    def load(self):
        self.actor.load_state_dict(torch.load("./d3qn/model/actor.pth"))
        self.critic.load_state_dict(torch.load("./d3qn/model/critic.pth"))
        self.actor.eval()
        self.critic.eval()
