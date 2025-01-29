import gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import Normal


class PPO(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.9,
        batch_size=128,
        epsilon=0.2,
        epoch_n=30,
        pi_lr=1e-4,
        v_lr=5e-4,
        use_new_method=False,
    ):

        super().__init__()

        self.pi_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * action_dim),
            nn.Tanh(),
        )

        self.v_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)
        self.use_new_method = use_new_method

    def get_action(self, state):
        mean, log_std = self.pi_model(torch.FloatTensor(state))
        dist = Normal(mean, log_std.exp())
        action = dist.sample()
        return action.detach().numpy().reshape(1)

    def fit(self, states, actions, rewards, dones, next_states=None):
        states, actions, rewards, dones = map(
            np.array, [states, actions, rewards, dones]
        )
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        if self.use_new_method:
            next_state_values = self.v_model(torch.FloatTensor(next_states)).detach()
        else:
            returns = np.zeros_like(rewards)
            returns[-1] = rewards[-1]
            for t in range(len(returns) - 2, -1, -1):
                returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]
            returns = torch.FloatTensor(returns)

        rewards, actions, states, dones = map(
            torch.FloatTensor, [rewards, actions, states, dones]
        )
        mean, log_std = self.pi_model(states).T
        mean, log_std = mean.unsqueeze(1), log_std.unsqueeze(1)
        dist = Normal(mean, torch.exp(log_std))
        old_log_probs = dist.log_prob(actions).detach()

        for epoch in range(self.epoch_n):
            idxs = np.random.permutation(dones.shape[0])
            for i in range(0, dones.shape[0], self.batch_size):
                b_idxs = idxs[i : i + self.batch_size]
                b_states = states[b_idxs]
                b_actions = actions[b_idxs]
                b_rewards = rewards[b_idxs]
                b_dones = dones[b_idxs]

                if not self.use_new_method:
                    b_returns = returns[b_idxs]
                    b_advantages = b_returns.detach() - self.v_model(b_states)
                else:
                    b_state_values = self.v_model(b_states)
                    b_next_state_values = torch.FloatTensor(next_state_values[b_idxs])
                    b_advantages = (
                        b_rewards
                        + (1 - b_dones) * self.gamma * b_next_state_values
                        - b_state_values
                    )

                b_old_log_probs = old_log_probs[b_idxs]

                b_mean, b_log_std = self.pi_model(b_states).T
                b_mean, b_log_std = b_mean.unsqueeze(1), b_log_std.unsqueeze(1)
                b_dist = Normal(b_mean, torch.exp(b_log_std))
                b_new_log_probs = b_dist.log_prob(b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantages.detach()
                pi_loss_2 = (
                    torch.clamp(b_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                    * b_advantages.detach()
                )
                pi_loss = -torch.mean(torch.min(pi_loss_1, pi_loss_2))

                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean(b_advantages**2)

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()


def train_agent(agent, use_new_method):
    total_rewards = []

    for episode in range(episode_n):
        states, actions, rewards, dones, next_states = [], [], [], [], []

        for _ in range(trajectory_n):
            total_reward = 0
            state = env.reset()
            for t in range(200):
                states.append(state)

                action = agent.get_action(state)
                actions.append(action)

                next_state, reward, done, _ = env.step(2 * action)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)

        if use_new_method:
            agent.fit(states, actions, rewards, dones, next_states)
        else:
            agent.fit(states, actions, rewards, dones)

    return total_rewards


env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print(f"{state_dim=}")
print(f"{action_dim=}")

episode_n = 50
trajectory_n = 20

agent_old = PPO(state_dim, action_dim, use_new_method=False)
total_rewards_old = train_agent(agent_old, use_new_method=False)

agent_new = PPO(state_dim, action_dim, use_new_method=True)
total_rewards_new = train_agent(agent_new, use_new_method=True)

smoothed_rewards_old = pd.Series(total_rewards_old).ewm(span=10).mean()
smoothed_rewards_new = pd.Series(total_rewards_new).ewm(span=10).mean()

plt.figure(figsize=(12, 6))
plt.plot(smoothed_rewards_old, label="Old Method")
plt.plot(smoothed_rewards_new, label="New Method")
plt.title("Total Rewards Comparison")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.legend()
plt.show()
