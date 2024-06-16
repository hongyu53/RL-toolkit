import torch
import torch.nn as nn
from dqn.agent import Agent as DQN_Agent


class Agent(DQN_Agent):
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

    def learn(self, training_set, num_epoch):
        # update target
        if self.learning_step_counter % self.update_interval == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_step_counter += 1

        for _ in range(num_epoch):
            # load training set
            s, a, s_, r, d = training_set
            s = torch.tensor(s, dtype=torch.float32)
            a = torch.tensor(a, dtype=torch.int64)
            s_ = torch.tensor(s_, dtype=torch.float32)
            r = torch.tensor(r, dtype=torch.float32)
            d = torch.tensor(d, dtype=torch.float32)
            # q-value evaluation
            q_eval = self.eval_net(s).gather(1, a.unsqueeze(1)).squeeze()
            a_max = self.eval_net(s_).max(1)[1].squeeze()
            q_next = self.target_net(s_).gather(1, a_max.unsqueeze(1)).squeeze()
            q_target = r + self.gamma * q_next * (1 - d)
            loss = nn.functional.mse_loss(q_eval, q_target)
            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        torch.save(self.eval_net.state_dict(), "./ddqn/model/ddqn.pt")

    def load(self):
        self.eval_net.load_state_dict(torch.load("./ddqn/model/ddqn.pt"))
        self.eval_net.eval()
