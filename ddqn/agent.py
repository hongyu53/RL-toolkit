import torch
import torch.nn as nn
from dqn.agent import Agent as dqn_agent


class Agent(dqn_agent):
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
        super(Agent, self).__init__(
            state_dim,
            action_dim,
            lr,
            gamma,
            epsilon,
            epsilon_decay,
            epsilon_min,
            update_interval,
        )

    def train(self, training_set, num_epoch):
        # update target
        if self.learning_step_counter % self.update_interval == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.learning_step_counter += 1

        for _ in range(num_epoch):
            # load training set
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
        torch.save(self.dqn.state_dict(), "./ddqn/model/ddqn.pt")

    def load(self):
        self.dqn.load_state_dict(torch.load("./ddqn/model/ddqn.pt"))
        self.dqn.eval()
