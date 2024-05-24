import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ddqn.agent import Agent
from ddqn.assets import env, params
from ddqn.utils.replay_buffer import ReplayBuffer


class Trainer:
    def __init__(self):
        self.env = env
        self.agent = Agent(
            state_dim=params["state_dim"],
            action_dim=params["action_dim"],
            lr=params["lr"],
            gamma=params["gamma"],
            epsilon=params["epsilon"],
            epsilon_decay=params["epsilon_decay"],
            epsilon_min=params["epsilon_min"],
            update_interval=params["update_interval"],
        )
        self.buffer = ReplayBuffer(
            memory_capacity=params["memory_capacity"], batch_size=params["batch_size"]
        )
        self.num_episode = params["num_episode"]
        self.num_epoch = params["num_epoch"]

    def train(self):
        bar = tqdm(range(self.num_episode))
        episode_rewards = []

        for _ in bar:
            state, _ = self.env.reset()
            episode_reward = 0

            while True:
                action = self.agent.select_action(torch.Tensor(state))
                next_state, reward, done, _, _ = self.env.step(action)
                self.buffer.store((state, action, next_state, reward, done))
                if self.buffer.is_full():
                    training_set = self.buffer.sample()
                    self.agent.train(training_set, self.num_epoch)
                if done:
                    break
                state = next_state
                episode_reward += reward
            episode_rewards.append(episode_reward)
            bar.set_description(f"Episode Reward: {episode_reward:.2f}")
        # save
        self.agent.save()

        plt.figure()
        avg_rewards = []
        for i in range(len(episode_rewards)):
            avg_rewards.append(np.mean(episode_rewards[max(0, i - 100) : i + 1]))
        plt.plot(episode_rewards)
        plt.plot(avg_rewards, "-.")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend(["Reward", "Avg Reward"])
        plt.tight_layout()
        plt.savefig(f"./ddqn/result/training_rewards.png")
