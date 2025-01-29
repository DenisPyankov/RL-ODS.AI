import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
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
    ):
        super().__init__()
        self.action_dim = action_dim

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

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self.pi_model(state_tensor)
        mean, log_std = logits.split(self.action_dim, dim=-1)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.tanh(action)
        return action.squeeze(0).numpy()

    def fit(self, states, actions, rewards, dones):
        states, actions, rewards, dones = map(
            np.array, [states, actions, rewards, dones]
        )
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]
        for t in range(returns.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        states, actions, returns = map(torch.FloatTensor, [states, actions, returns])

        logits = self.pi_model(states)
        mean, log_std = logits.split(self.action_dim, dim=-1)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        old_log_probs = dist.log_prob(actions).sum(dim=-1).detach()

        for epoch in range(self.epoch_n):
            idxs = np.random.permutation(returns.shape[0])
            epoch_pi_loss = 0
            epoch_v_loss = 0
            for i in range(0, returns.shape[0], self.batch_size):
                b_idxs = idxs[i : i + self.batch_size]
                b_states = states[b_idxs]
                b_actions = actions[b_idxs]
                b_returns = returns[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                b_advantage = b_returns.detach() - self.v_model(b_states)

                b_logits = self.pi_model(b_states)

                b_mean, b_log_std = b_logits.split(self.action_dim, dim=-1)
                b_std = torch.exp(b_log_std)
                b_dist = Normal(b_mean, b_std)
                b_new_log_probs = b_dist.log_prob(b_actions).sum(dim=-1)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = (
                    torch.clamp(b_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                    * b_advantage.detach()
                )
                pi_loss = -torch.mean(torch.min(pi_loss_1, pi_loss_2))

                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean(b_advantage**2)

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()

                epoch_pi_loss += pi_loss.item()
                epoch_v_loss += v_loss.item()

    def train_agent(self, env, episodes=100, trajectory_n=10):
        total_rewards = []

        for episode in range(episodes):
            states, actions, rewards, dones = [], [], [], []

            for _ in range(trajectory_n):
                total_reward = 0
                state, _ = env.reset()
                for t in range(200):
                    states.append(state)

                    action = self.get_action(state)
                    actions.append(action)

                    state, reward, done, _, _ = env.step(action)
                    rewards.append(reward)
                    dones.append(done)

                    total_reward += reward
                    if done:
                        break

                total_rewards.append(total_reward)

            self.fit(states, actions, rewards, dones)

            print(
                f"Episode {episode+1}/{episodes} | Total Reward: {np.mean(total_rewards[-trajectory_n:]):.4f}"
            )

        return total_rewards


env = gym.make("LunarLander-v2", continuous=True)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = PPO(state_dim, action_dim)

total_rewards = agent.train_agent(env, episodes=70, trajectory_n=60)

plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve")
plt.show()
