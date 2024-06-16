import numpy as np


class ReplayBuffer:
    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.buffer = []
        self.pointer = 0

    def __len__(self):
        return len(self.buffer)

    def is_full(self):
        return len(self.buffer) >= self.memory_capacity

    def store(self, transition):
        s, a, s_, r, d = transition
        if len(self.buffer) < self.memory_capacity:
            self.buffer.append((s, a, s_, r, d))
        else:
            self.buffer[self.pointer] = (s, a, s_, r, d)
        self.pointer = (self.pointer + 1) % self.memory_capacity

    def sample(self):
        indices = np.random.choice(len(self.buffer), self.batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []
        for i in indices:
            s, a, s_, r, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            next_states.append(s_)
            rewards.append(r)
            dones.append(d)
        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards),
            np.array(dones),
        )
