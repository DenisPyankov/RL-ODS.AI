import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


# Deep CEM
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


# DQN
class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, output_dim)
        self.activation = nn.ReLU()

    def forward(self, input):
        hidden = self.linear_1(input)
        hidden = self.activation(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.activation(hidden)
        output = self.linear_3(hidden)
        return output


class DQN:
    def __init__(
        self,
        state_dim,
        action_n,
        epsilon_decrease,
        gamma=0.99,
        batch_size=128,
        lr=1e-3,
        epsilon_min=1e-2,
    ):
        self.state_dim = state_dim
        self.action_n = action_n
        self.q_model = Network(self.state_dim, self.action_n)
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.epsilon = 1
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = []
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=lr)

    def get_action(self, state):
        q_values = self.q_model(torch.FloatTensor(state)).data.numpy()
        max_action = np.argmax(q_values)
        probs = np.ones(self.action_n) * self.epsilon / self.action_n
        probs[max_action] += 1 - self.epsilon
        return np.random.choice(np.arange(self.action_n), p=probs)

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(
                torch.tensor, zip(*batch)
            )

            targets = (
                rewards
                + (1 - dones)
                * self.gamma
                * torch.max(self.q_model(next_states), dim=1).values
            )
            q_values = self.q_model(states)[torch.arange(self.batch_size), actions]
            loss = torch.mean((q_values - targets) ** 2)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.epsilon = max(self.epsilon - self.epsilon_decrease, self.epsilon_min)


# Training Functions
def train_deep_cem(
    state_dim, action_n, learning_rate, q_param, max_trajectories, trajectory_step=50
):
    agent = CEM(state_dim, action_n, learning_rate)
    rewards_history = []

    total_trajectories = 0
    while total_trajectories < max_trajectories:
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_step)]
        total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
        mean_reward = np.mean(total_rewards)
        rewards_history.append(mean_reward)

        print(
            f"Deep CEM - Total Trajectories: {total_trajectories + trajectory_step}, "
            f"Mean Reward (Last {trajectory_step}): {mean_reward:.2f}"
        )

        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = [
            trajectory
            for trajectory in trajectories
            if np.sum(trajectory["rewards"]) > quantile
        ]

        if len(elite_trajectories) > 0:
            agent.fit(elite_trajectories)

        total_trajectories += trajectory_step

    return rewards_history


def train_dqn(
    state_dim,
    action_n,
    gamma,
    learning_rate,
    epsilon_decrease,
    t_max,
    max_trajectories,
    trajectory_step=50,
):
    agent = DQN(state_dim, action_n, epsilon_decrease, gamma=gamma, lr=learning_rate)
    rewards_history = []

    total_trajectories = 0
    while total_trajectories < max_trajectories:
        total_reward_batch = []
        for _ in range(trajectory_step):
            total_reward = 0
            state, _ = env.reset()
            for _ in range(t_max):
                action = agent.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward

                agent.fit(state, action, reward, done, next_state)
                state = next_state

                if done:
                    break

            total_reward_batch.append(total_reward)

        mean_reward = np.mean(total_reward_batch)
        rewards_history.append(mean_reward)
        total_trajectories += trajectory_step

        print(
            f"DQN - Total Trajectories: {total_trajectories}, "
            f"Mean Reward (Last {trajectory_step}): {mean_reward:.2f}"
        )

    return rewards_history


# Experiment Settings
state_dim = 8
action_n = 4
max_trajectories = 300
trajectory_step = 50

# Hyperparameters
cem_params = {"learning_rate": 0.01, "q_param": 0.7}
dqn_params_list = [
    {"gamma": 0.99, "learning_rate": 1e-3, "epsilon_decrease": 1e-4, "t_max": 500},
    {"gamma": 0.99, "learning_rate": 1e-3, "epsilon_decrease": 1e-2, "t_max": 500},
    {"gamma": 0.99, "learning_rate": 1e-4, "epsilon_decrease": 1e-4, "t_max": 500},
    {"gamma": 0.99, "learning_rate": 1e-3, "epsilon_decrease": 1e-4, "t_max": 1000},
    {"gamma": 0.9999, "learning_rate": 1e-3, "epsilon_decrease": 1e-4, "t_max": 500},
    {"gamma": 0.9999, "learning_rate": 1e-2, "epsilon_decrease": 1e-4, "t_max": 500},
    {"gamma": 0.9999, "learning_rate": 1e-3, "epsilon_decrease": 5e-4, "t_max": 500},
    {"gamma": 0.9999, "learning_rate": 1e-3, "epsilon_decrease": 1e-4, "t_max": 1000},
]

# Collect Results
env = gym.make("LunarLander-v2")
results = {}

# Deep CEM
results["CEM"] = train_deep_cem(
    state_dim,
    action_n,
    cem_params["learning_rate"],
    cem_params["q_param"],
    max_trajectories,
    trajectory_step,
)

# DQN
for i, dqn_params in enumerate(dqn_params_list):
    results[f"DQN_{i}"] = train_dqn(
        state_dim,
        action_n,
        dqn_params["gamma"],
        dqn_params["learning_rate"],
        dqn_params["epsilon_decrease"],
        dqn_params["t_max"],
        max_trajectories,
        trajectory_step,
    )

env.close()

# Plot Results
plt.figure(figsize=(12, 8))
plt.plot(
    range(0, max_trajectories, trajectory_step),
    results["CEM"],
    label=f"CEM (lr={cem_params['learning_rate']}, q={cem_params['q_param']})",
)

for i, (label, rewards) in enumerate(results.items()):
    if label.startswith("DQN") and i < len(dqn_params_list):
        dqn_params = dqn_params_list[i]
        gamma = dqn_params["gamma"]
        lr = dqn_params["learning_rate"]
        eps_dec = dqn_params["epsilon_decrease"]
        t_max = dqn_params["t_max"]
        dqn_label = (
            f"DQN {i} (gamma={gamma}, lr={lr}, eps_dec={eps_dec}, t_max={t_max})"
        )
        plt.plot(
            range(0, max_trajectories, trajectory_step),
            rewards,
            label=dqn_label,
        )

plt.xlabel("Number of Trajectories")
plt.ylabel("Mean Total Reward")
plt.title("Training Curves: Deep CEM vs DQN")
plt.legend()
plt.grid()
plt.show()
