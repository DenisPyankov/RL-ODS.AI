import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

env = gym.make("LunarLander-v2")
# env = gym.make("LunarLander-v2", render_mode="human")

state_dim = 8
action_n = 4


class CEM(nn.Module):
    def __init__(self, state_dim, action_n, learning_rate):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 128), nn.ReLU(), nn.Linear(128, self.action_n)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        probs = self.softmax(logits).detach().numpy()
        action = np.random.choice(self.action_n, p=probs)
        return action

    def fit(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                elite_states.append(state)
                elite_actions.append(action)
        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.LongTensor(elite_actions)
        pred_actions = self.forward(elite_states)

        loss = self.loss_fn(pred_actions, elite_actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def get_trajectory(env, agent, max_len=1000):
    trajectory = {"states": [], "actions": [], "rewards": []}
    state, _ = env.reset()

    for _ in range(max_len):
        trajectory["states"].append(state)
        action = agent.get_action(state)
        trajectory["actions"].append(action)
        state, reward, done, truncated, _ = env.step(action)
        trajectory["rewards"].append(reward)

        if done or truncated:
            break

    return trajectory


def train_agent(learning_rate, q_param, iteration_n=100, trajectory_n=100):
    agent = CEM(state_dim, action_n, learning_rate)
    rewards_history = []

    for iteration in range(iteration_n):
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
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

    return rewards_history


# learning_rates = [0.1, 0.01, 0.001, 0.0001]
learning_rates = [0.01]
# q_params = [0.6, 0.7, 0.8]
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
