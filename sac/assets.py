import gym

env = gym.make(
    "Pendulum-v1",
    max_episode_steps=200,
    g=9.81,
)

params = {
    # hyperparameters
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,
    "state_dim": env.observation_space.shape[0],
    "action_dim": env.action_space.shape[0],
    "action_bound": float(env.action_space.high[0]),
    # replay buffer
    "batch_size": 64,
    "memory_capacity": 1000,
    # training
    "num_episode": 1000,
    "num_epoch": 2,
}
