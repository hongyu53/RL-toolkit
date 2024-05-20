import matplotlib.pyplot as plt
import numpy as np
import torch
from ddpg.agent import Agent
from ddpg.assets import env, params
from ddpg.utils.replay_buffer import ReplayBuffer
from tqdm import tqdm


class Trainer:
    def __init__(self):
        self.env = env
        self.agent = Agent(
            state_dim=params["state_dim"],
            action_dim=params["action_dim"],
            action_bound=params["action_bound"],
            lr=params["lr"],
            gamma=params["gamma"],
            tau=params["tau"],
            noise_std=params["noise_std"],
        )
        self.buffer = ReplayBuffer(
            memory_capacity=params["memory_capacity"], batch_size=params["batch_size"]
        )

    def train(self):
        num_episode = params["num_episode"]
        bar = tqdm(range(num_episode))
        episode_rewards = []

        for _ in bar:
            state, _ = self.env.reset()
            episode_reward = 0

            while True:
                action = self.agent.select_action(torch.Tensor(state))
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.buffer.store((state, action, next_state, reward, done))
                if self.buffer.is_full():
                    training_set = self.buffer.sample()
                    self.agent.train(training_set, params["num_epoch"])
                if done or truncated:
                    break
                state = next_state
                episode_reward += reward
            bar.set_description(f"Episode Reward: {episode_reward:.2f}")
            episode_rewards.append(episode_reward)

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
        plt.savefig(f"./ddpg/result/reward.png")
