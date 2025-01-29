import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

env = gym.make("MountainCarContinuous-v0", render_mode="human")
# env = gym.make("MountainCarContinuous-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]


class CEM(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 128), nn.ReLU(), nn.Linear(128, self.action_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()  # Для непрерывного действия используем MSE

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state, epsilon):
        state = torch.FloatTensor(state)
        mean_action = self.forward(state).detach().numpy()
        action = mean_action + epsilon * np.random.randn(
            self.action_dim
        )  # Добавление шума
        return np.clip(
            action, -action_bound, action_bound
        )  # Ограничение действия границами

    def fit(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                elite_states.append(state)
                elite_actions.append(action)
        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.FloatTensor(np.array(elite_actions))
        pred_actions = self.forward(elite_states)

        loss = self.loss_fn(pred_actions, elite_actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def get_trajectory(env, agent, epsilon, max_len=1000):
    trajectory = {"states": [], "actions": [], "rewards": []}
    state, _ = env.reset()

    for _ in range(max_len):
        trajectory["states"].append(state)
        action = agent.get_action(state, epsilon)
        trajectory["actions"].append(action)
        state, reward, done, truncated, _ = env.step(action)
        trajectory["rewards"].append(reward)

        if done or truncated:
            break

    return trajectory


def train_agent(
    learning_rate,
    q_param,
    epsilon_start=0.000001,
    epsilon_decay=0.9,
    iteration_n=100,
    trajectory_n=1000,
):
    agent = CEM(state_dim, action_dim, learning_rate)
    rewards_history = []
    epsilon = epsilon_start

    for iteration in range(iteration_n):
        trajectories = [
            get_trajectory(env, agent, epsilon) for _ in range(trajectory_n)
        ]
        total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
        mean_reward = np.mean(total_rewards)
        rewards_history.append(mean_reward)
        print(
            f"Learning Rate: {learning_rate}, Q-Param: {q_param}, Iteration: {iteration}, Mean Total Reward: {mean_reward}"
        )

        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = [
            trajectory
            for trajectory in trajectories
            if np.sum(trajectory["rewards"]) > quantile
        ]

        if len(elite_trajectories) > 0:
            agent.fit(elite_trajectories)

        epsilon *= epsilon_decay

    return rewards_history


learning_rates = [0.01]
q_params = [0.7]
results = {}

for lr in learning_rates:
    for q in q_params:
        print(f"Training with Learning Rate: {lr} and Q-Param: {q}")
        rewards = train_agent(lr, q)
        results[(lr, q)] = rewards

plt.figure(figsize=(12, 8))
for (lr, q), rewards in results.items():
    plt.plot(rewards, label=f"LR: {lr}, Q: {q}")

plt.xlabel("Iteration")
plt.ylabel("Mean Total Reward")
plt.title("Hyperparameter Tuning Results")
plt.legend()
plt.grid()
plt.show()

env.close()
