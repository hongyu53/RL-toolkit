from q_learning.agent import Agent
from q_learning.assets import env, params
from tqdm import tqdm


class Trainer:
    def __init__(self):
        self.env = env
        self.agent = Agent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            lr=params["lr"],
            gamma=params["gamma"],
            epsilon=params["epsilon"],
        )

    def train(self):
        episodes = params["num_episodes"]
        bar = tqdm(range(episodes))
        for _ in bar:
            episode_reward = 0
            state = self.env.reset()
            action = self.agent.epsilon_greedy(state)
            done = False
            while not done:
                next_state, reward, done = self.env.step(action)
                self.agent.q_table.loc[str(state), str(action)] += self.agent.lr * (
                    reward
                    + self.agent.gamma
                    * self.agent.q_table.loc[
                        str(next_state), str(self.agent.epsilon_greedy(next_state))
                    ]
                    - self.agent.q_table.loc[str(state), str(action)]
                )
                state = next_state
                action = self.agent.epsilon_greedy(state)
                episode_reward += reward
            bar.set_description(f"Episode Reward: {episode_reward}")
        # save
        self.agent.save()
