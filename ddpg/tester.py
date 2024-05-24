import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ddpg.agent import Agent
from ddpg.assets import env, params


class Tester:
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

    def test(self):
        self.agent.load()
        episodes = 20
        episode_rewards = []
        bar = tqdm(range(episodes))

        for _ in bar:
            state, _ = self.env.reset()
            episode_reward = 0

            while True:
                action = self.agent.select_action(torch.Tensor(state), test=True)
                next_state, reward, done, truncated, _ = self.env.step(action)
                if done or truncated:
                    break
                state = next_state
                episode_reward += reward
            episode_rewards.append(episode_reward)
            bar.set_description(f"Episode Reward: {episode_reward:.2f}")
        # save
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
        plt.savefig(f"./ddpg/result/test_rewards.png")
