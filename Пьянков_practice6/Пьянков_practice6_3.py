import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Categorical


class PPO(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.9,
        epsilon=0.2,
        lr=1e-4,
        batch_size=1024,
        epoch_n=10,
    ):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epoch_n = epoch_n

        # Политика
        self.pi_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )

        # Функция ценности
        self.v_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Оптимизаторы
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=lr)

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.pi_model(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def fit(self, states, actions, rewards, dones):
        # Вычисление дисконтированных возвратов

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        dones = torch.FloatTensor(np.array(dones))

        returns = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * (
                returns[t + 1] if t + 1 < len(rewards) else 0
            )

        # Преобразование в тензоры
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(returns)

        # Нормализация возвратов
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        old_log_probs = Categorical(self.pi_model(states)).log_prob(actions).detach()

        for _ in range(self.epoch_n):
            idxs = np.random.permutation(len(states))
            for i in range(0, len(states), self.batch_size):
                batch_idxs = idxs[i : i + self.batch_size]
                b_states = states[batch_idxs]
                b_actions = actions[batch_idxs]
                b_returns = returns[batch_idxs]
                b_old_log_probs = old_log_probs[batch_idxs]

                # Политика
                logits = self.pi_model(b_states)
                dist = Categorical(logits)
                new_log_probs = dist.log_prob(b_actions)

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                advantage = b_returns - self.v_model(b_states).detach()

                pi_loss = -torch.mean(
                    torch.min(
                        ratio * advantage,
                        torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                        * advantage,
                    )
                )

                self.pi_optimizer.zero_grad()
                pi_loss.backward()
                self.pi_optimizer.step()

                # Функция ценности
                v_loss = torch.mean((self.v_model(b_states) - b_returns) ** 2)
                self.v_optimizer.zero_grad()
                v_loss.backward()
                self.v_optimizer.step()

    def train_agent(self, env, episodes=500, trajectory_n=20):
        total_rewards = []

        for episode in range(episodes):
            states, actions, rewards, dones = [], [], [], []
            for _ in range(trajectory_n):
                state, _ = env.reset()
                total_reward = 0
                max_steps = 1000
                for step in range(max_steps):
                    action, log_prob = self.get_action(state)
                    next_state, reward, done, _, _ = env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    dones.append(done)

                    state = next_state
                    total_reward += reward
                    if done:
                        break

                total_rewards.append(total_reward)

            self.fit(states, actions, rewards, dones)

            print(
                f"Episode {episode + 1}/{episodes}, Avg Reward: {np.mean(total_rewards[-trajectory_n:]):.2f}"
            )

        return total_rewards


env = gym.make("Acrobot-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPO(state_dim, action_dim)
rewards = agent.train_agent(env, episodes=100, trajectory_n=20)


plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training PPO on Acrobot")
plt.show()
