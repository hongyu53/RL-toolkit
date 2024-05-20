import gym

env = gym.make("MountainCar-v0")

params = {
    "lr": 0.0001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.99999,
    "epsilon_min": 0.00001,
    "batch_size": 64,
    "memory_capacity": 2000,
    "update_interval": 50,
    "episodes": 8000,
    "state_dim": env.observation_space.shape[0],
    "action_dim": env.action_space.n,
}
