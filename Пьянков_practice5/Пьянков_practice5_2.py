import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


# DQN Network
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


# Base DQN Class
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


# Modified DQN Class
class DQNModified(DQN):
    def __init__(
        self,
        state_dim,
        action_n,
        epsilon_decrease,
        gamma=0.99,
        batch_size=128,
        lr=1e-3,
        epsilon_min=1e-2,
        target_update="hard",
        update_freq=100,
        tau=0.01,
    ):
        super().__init__(
            state_dim, action_n, epsilon_decrease, gamma, batch_size, lr, epsilon_min
        )
        self.target_model = Network(self.state_dim, self.action_n)
        self.target_model.load_state_dict(self.q_model.state_dict())
        self.target_model.eval()
        self.update_freq = update_freq
        self.tau = tau
        self.target_update = target_update
        self.steps = 0

    def soft_update(self):
        for target_param, main_param in zip(
            self.target_model.parameters(), self.q_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * main_param.data + (1 - self.tau) * target_param.data
            )

    def hard_update(self):
        self.target_model.load_state_dict(self.q_model.state_dict())

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(
                torch.tensor, zip(*batch)
            )

            if self.target_update == "double":
                next_actions = torch.argmax(self.q_model(next_states), dim=1)
                targets = (
                    rewards
                    + (1 - dones)
                    * self.gamma
                    * self.target_model(next_states)[
                        torch.arange(self.batch_size), next_actions
                    ]
                )
            else:
                targets = (
                    rewards
                    + (1 - dones)
                    * self.gamma
                    * torch.max(self.target_model(next_states), dim=1).values
                )

            q_values = self.q_model(states)[torch.arange(self.batch_size), actions]
            loss = torch.mean((q_values - targets) ** 2)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.steps += 1
            if self.target_update == "hard" and self.steps % self.update_freq == 0:
                self.hard_update()
            elif self.target_update == "soft":
                self.soft_update()

            self.epsilon = max(self.epsilon - self.epsilon_decrease, self.epsilon_min)


# Training Function
def train_dqn(
    state_dim,
    action_n,
    gamma,
    learning_rate,
    epsilon_decrease,
    t_max,
    max_trajectories,
    trajectory_step=50,
    agent_class=DQN,
    **agent_params,
):
    agent = agent_class(
        state_dim,
        action_n,
        epsilon_decrease,
        gamma=gamma,
        lr=learning_rate,
        **agent_params,
    )
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
            f"{agent_class.__name__} - Total Trajectories: {total_trajectories}, "
            f"Mean Reward (Last {trajectory_step}): {mean_reward:.2f}"
        )

    return rewards_history


# Experiment Settings
state_dim = 8
action_n = 4
max_trajectories = 300
trajectory_step = 50

# Gym Environment
env = gym.make("LunarLander-v2")

# Modifications and Hyperparameters
modifications = [
    {
        "name": "Original_DQN",
        "class": DQN,
        "params": {
            "gamma": 0.99,
            "learning_rate": 1e-3,
            "epsilon_decrease": 1e-4,
        },
    },
    {
        "name": "Original_DQN",
        "class": DQN,
        "params": {
            "gamma": 0.9999,
            "learning_rate": 1e-3,
            "epsilon_decrease": 5e-4,
        },
    },
    {
        "name": "DQN_Hard",
        "class": DQNModified,
        "params": {
            "gamma": 0.99,
            "learning_rate": 1e-3,
            "epsilon_decrease": 1e-4,
            "target_update": "hard",
            "update_freq": 100,
        },
    },
    {
        "name": "DQN_Hard",
        "class": DQNModified,
        "params": {
            "gamma": 0.999,
            "learning_rate": 1e-3,
            "epsilon_decrease": 1e-4,
            "target_update": "hard",
            "update_freq": 50,
        },
    },
    {
        "name": "DQN_Hard",
        "class": DQNModified,
        "params": {
            "gamma": 0.99,
            "learning_rate": 1e-3,
            "epsilon_decrease": 1e-4,
            "target_update": "hard",
            "update_freq": 20,
        },
    },
    {
        "name": "DQN_Soft",
        "class": DQNModified,
        "params": {
            "gamma": 0.99,
            "learning_rate": 1e-3,
            "epsilon_decrease": 1e-4,
            "target_update": "soft",
            "tau": 0.01,
        },
    },
    {
        "name": "DQN_Soft",
        "class": DQNModified,
        "params": {
            "gamma": 0.99,
            "learning_rate": 1e-3,
            "epsilon_decrease": 1e-4,
            "target_update": "soft",
            "tau": 0.3,
        },
    },
    {
        "name": "DQN_Soft",
        "class": DQNModified,
        "params": {
            "gamma": 0.99,
            "learning_rate": 1e-3,
            "epsilon_decrease": 1e-4,
            "target_update": "soft",
            "tau": 0.7,
        },
    },
    {
        "name": "Double_DQN",
        "class": DQNModified,
        "params": {
            "gamma": 0.99,
            "learning_rate": 1e-3,
            "epsilon_decrease": 1e-4,
            "target_update": "double",
        },
    },
    {
        "name": "Double_DQN",
        "class": DQNModified,
        "params": {
            "gamma": 0.99,
            "learning_rate": 1e-3,
            "epsilon_decrease": 5e-4,
            "target_update": "double",
        },
    },
    {
        "name": "Double_DQN",
        "class": DQNModified,
        "params": {
            "gamma": 0.9999,
            "learning_rate": 1e-3,
            "epsilon_decrease": 5e-4,
            "target_update": "double",
        },
    },
]

# Collect Results
results = {}
for mod in modifications:
    print(f"Training {mod['name']}")
    rewards = train_dqn(
        state_dim,
        action_n,
        t_max=500,
        max_trajectories=max_trajectories,
        trajectory_step=trajectory_step,
        agent_class=mod["class"],
        **mod["params"],
    )
    results[mod["name"]] = rewards

env.close()

# Plot Results
plt.figure(figsize=(12, 8))
for label, rewards in results.items():
    plt.plot(
        range(0, max_trajectories, trajectory_step),
        rewards,
        label=label,
    )

plt.xlabel("Number of Trajectories")
plt.ylabel("Mean Total Reward")
plt.title("Comparison of DQN Variants")
plt.legend()
plt.grid()
plt.show()
