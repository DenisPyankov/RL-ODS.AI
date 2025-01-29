import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Создаём окружение
env = gym.make("LunarLander-v2")
state_dim = env.observation_space.shape[0]
action_n = env.action_space.n

# Настройки для дискретизации состояния
number_of_buckets = (5, 5, 5, 5, 5, 5, 2, 2)  # количество бакетов для каждого измерения
state_value_bounds = [
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [0, 1],
    [0, 1],
]


def bucketize(state):
    bucket_indexes = []
    for i in range(len(state)):
        if state[i] <= state_value_bounds[i][0]:
            bucket_index = 0
        elif state[i] >= state_value_bounds[i][1]:
            bucket_index = number_of_buckets[i] - 1
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (number_of_buckets[i] - 1) * state_value_bounds[i][0] / bound_width
            scaling = (number_of_buckets[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indexes.append(bucket_index)
    return tuple(bucket_indexes)


# Epsilon-greedy выбор действия
def get_epsilon_greedy_action(q_values, epsilon, action_n):
    prob = np.ones(action_n) * epsilon / action_n
    argmax_action = np.argmax(q_values)
    prob[argmax_action] += 1 - epsilon
    action = np.random.choice(np.arange(action_n), p=prob)
    return action


# Реализация алгоритмов Monte Carlo, SARSA, Q-Learning
def monte_carlo(env, episodes, gamma=0.99, max_trajectories=2000):
    q_values = np.zeros(number_of_buckets + (action_n,))
    counters = np.zeros(number_of_buckets + (action_n,))
    total_rewards, trajectory_counts = [], []
    trajectory_count = 0

    for episode in range(episodes):
        epsilon = 1 - episode / episodes
        trajectory = {"states": [], "actions": [], "rewards": []}
        state, _ = env.reset()
        state = bucketize(state)

        for t in range(500):
            action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
            next_state, reward, done, _, _ = env.step(action)
            next_state = bucketize(next_state)
            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            state = next_state
            if done:
                break

        trajectory_count += 1
        trajectory_counts.append(trajectory_count)
        total_rewards.append(np.sum(trajectory["rewards"]))

        returns = np.zeros(len(trajectory["rewards"]) + 1)
        for t in reversed(range(len(trajectory["rewards"]))):
            returns[t] = trajectory["rewards"][t] + gamma * returns[t + 1]

        for t in range(len(trajectory["rewards"])):
            state, action = trajectory["states"][t], trajectory["actions"][t]
            counters[state][action] += 1
            q_values[state][action] += (
                returns[t] - q_values[state][action]
            ) / counters[state][action]

        if trajectory_count >= max_trajectories:
            break

    return total_rewards, trajectory_counts


def sarsa(env, episodes, alpha=0.5, gamma=0.99, max_trajectories=2000):
    q_values = np.zeros(number_of_buckets + (action_n,))
    total_rewards, trajectory_counts = [], []
    trajectory_count = 0

    for episode in range(episodes):
        epsilon = 1 - episode / episodes
        state, _ = env.reset()
        state = bucketize(state)
        action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
        total_reward = 0

        for t in range(500):
            next_state, reward, done, _, _ = env.step(action)
            next_state = bucketize(next_state)
            next_action = get_epsilon_greedy_action(
                q_values[next_state], epsilon, action_n
            )
            q_values[state][action] += alpha * (
                reward
                + gamma * q_values[next_state][next_action]
                - q_values[state][action]
            )
            total_reward += reward
            state, action = next_state, next_action
            if done:
                break

        trajectory_count += 1
        trajectory_counts.append(trajectory_count)
        total_rewards.append(total_reward)

        if trajectory_count >= max_trajectories:
            break

    return total_rewards, trajectory_counts


def q_learning(env, episodes, alpha=0.5, gamma=0.99, max_trajectories=2000):
    q_values = np.zeros(number_of_buckets + (action_n,))
    total_rewards, trajectory_counts = [], []
    trajectory_count = 0

    for episode in range(episodes):
        epsilon = 1 - episode / episodes
        state, _ = env.reset()
        state = bucketize(state)
        total_reward = 0

        for t in range(500):
            action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
            next_state, reward, done, _, _ = env.step(action)
            next_state = bucketize(next_state)
            q_values[state][action] += alpha * (
                reward + gamma * np.max(q_values[next_state]) - q_values[state][action]
            )
            total_reward += reward
            state = next_state
            if done:
                break

        trajectory_count += 1
        trajectory_counts.append(trajectory_count)
        total_rewards.append(total_reward)

        if trajectory_count >= max_trajectories:
            break

    return total_rewards, trajectory_counts


# Реализация Deep CEM
class CEM(nn.Module):
    def __init__(self, state_dim, action_n, learning_rate):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_n)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, state):
        return self.network(state)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        probs = self.softmax(logits).detach().numpy()
        return np.random.choice(action_n, p=probs)

    def fit(self, elite_trajectories):
        elite_states, elite_actions = [], []
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


def train_deep_cem(learning_rate, q_param, max_trajectories=2000, trajectory_n=100):
    agent = CEM(state_dim, action_n, learning_rate)
    rewards_history, trajectory_counts = [], []
    trajectory_count = 0

    while trajectory_count < max_trajectories:
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
        rewards_history.append(np.mean(total_rewards))
        trajectory_count += trajectory_n
        trajectory_counts.append(trajectory_count)

        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = [
            trajectory
            for trajectory in trajectories
            if np.sum(trajectory["rewards"]) > quantile
        ]

        if elite_trajectories:
            agent.fit(elite_trajectories)

    return rewards_history, trajectory_counts


def get_trajectory(env, agent, max_len=500):
    trajectory = {"states": [], "actions": [], "rewards": []}
    state, _ = env.reset()
    for _ in range(max_len):
        trajectory["states"].append(state)
        action = agent.get_action(state)
        trajectory["actions"].append(action)
        state, reward, done, _, _ = env.step(action)
        trajectory["rewards"].append(reward)
        if done:
            break
    return trajectory


# Запуск всех алгоритмов и построение графиков
mc_rewards, mc_trajectories = monte_carlo(env, episodes=10000, max_trajectories=10000)
sarsa_rewards, sarsa_trajectories = sarsa(env, episodes=10000, max_trajectories=10000)
q_learning_rewards, q_learning_trajectories = q_learning(
    env, episodes=10000, max_trajectories=10000
)
cem_rewards, cem_trajectories = train_deep_cem(
    learning_rate=0.01, q_param=0.7, max_trajectories=10000, trajectory_n=100
)

plt.figure(figsize=(12, 8))
plt.plot(mc_trajectories, mc_rewards, label="Monte Carlo")
plt.plot(sarsa_trajectories, sarsa_rewards, label="SARSA")
plt.plot(q_learning_trajectories, q_learning_rewards, label="Q-Learning")
plt.plot(cem_trajectories, cem_rewards, label="Deep CEM")
plt.xlabel("Количество траекторий")
plt.ylabel("Средняя награда")
plt.title("Сравнение алгоритмов на основе траекторий")
plt.legend()
plt.grid()
plt.show()
env.close()
