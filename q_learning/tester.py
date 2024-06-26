from q_learning.agent import Agent
from q_learning.assets import env, params


class Tester:
    def __init__(self):
        self.env = env
        self.agent = Agent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            lr=params["lr"],
            gamma=params["gamma"],
            epsilon=params["epsilon"],
        )

    def test(self):
        self.agent.load()
        state = self.env.reset()
        done = False
        episode_reward = 0
        while not done:
            self.env.render()
            action = self.agent.greedy(state)
            state, reward, done = self.env.step(action)
            episode_reward += reward
        print(f"Episode Reward: {episode_reward}")
