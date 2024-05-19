import matplotlib.pyplot as plt
import numpy as np
import torch
from dqn.assets import env, params
from dqn.model import Agent
from tqdm import tqdm


class Tester:
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

    def test(self):
        self.agent.load()
        episodes = 20
        episode_rewards = []
        bar = tqdm(range(episodes))

        for _ in bar:
            state, _ = self.env.reset()
            episode_reward = 0

            while True:
                env.render()
                action = self.agent.select_action(torch.Tensor(state), test=True)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                if done:
                    break
                state = next_state
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
        plt.savefig(f"./dqn/result/test_rewards.png")
