import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time


# Определение функции epsilon-greedy для выбора действий
def get_epsilon_greedy_action(q_values, epsilon, action_n):
    prob = np.ones(action_n) * epsilon / action_n
    argmax_action = np.argmax(q_values)
    prob[argmax_action] += 1 - epsilon
    action = np.random.choice(np.arange(action_n), p=prob)
    return action


# Реализация алгоритма Cross-Entropy
class CrossEntropyAgent:
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)

    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model


def get_trajectory(env, agent, max_len=1000):
    trajectory = {"states": [], "actions": [], "rewards": []}
    obs, _ = env.reset()
    state = obs

    for _ in range(max_len):
        trajectory["states"].append(state)
        action = agent.get_action(state)
        trajectory["actions"].append(action)
        obs, reward, done, _, _ = env.step(action)
        trajectory["rewards"].append(reward)
        state = obs
        if done:
            break

    return trajectory


# Инициализация сред
env = gym.make("Taxi-v3")
state_n = env.observation_space.n
action_n = env.action_space.n


# Реализация Cross-Entropy
def run_cross_entropy(env, episode_n, trajectory_n=200, q_param=0.5):
    agent = CrossEntropyAgent(state_n, action_n)
    rewards = []
    trajectory_counts = []

    for episode in range(episode_n):
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
        total_trajectories = (episode + 1) * trajectory_n
        trajectory_counts.append(total_trajectories)
        total_rewards = [np.sum(t["rewards"]) for t in trajectories]
        rewards.append(np.mean(total_rewards))
        quantile = np.quantile(total_rewards, q_param)
        elite_trajectories = [
            t for t in trajectories if np.sum(t["rewards"]) > quantile
        ]
        agent.fit(elite_trajectories)
        print(
            f"Cross-Entropy: Средняя награда после {total_trajectories} траекторий: {rewards[-1]}"
        )

    return rewards, trajectory_counts


# Реализация Monte Carlo
def MonteCarlo(env, episode_n, trajectory_len=500, gamma=0.99):
    q_values = np.zeros((state_n, action_n))
    counters = np.zeros((state_n, action_n))
    total_rewards = []
    trajectory_counts = []

    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n
        trajectory = {"states": [], "actions": [], "rewards": []}
        state, _ = env.reset()

        for t in range(trajectory_len):
            action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
            next_state, reward, done, _, _ = env.step(action)
            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            state = next_state
            if done:
                break

        total_trajectories = episode + 1
        trajectory_counts.append(total_trajectories)
        total_rewards.append(np.sum(trajectory["rewards"]))
        print(
            f"Monte Carlo: Средняя награда после {total_trajectories} траекторий: {total_rewards[-1]}"
        )

        real_trajectory_len = len(trajectory["rewards"])
        returns = np.zeros(real_trajectory_len + 1)
        for t in range(real_trajectory_len - 1, -1, -1):
            returns[t] = trajectory["rewards"][t] + gamma * returns[t + 1]

        for t in range(real_trajectory_len):
            state = trajectory["states"][t]
            action = trajectory["actions"][t]
            counters[state][action] += 1
            q_values[state][action] += (
                returns[t] - q_values[state][action]
            ) / counters[state][action]

    return total_rewards, trajectory_counts


# Реализация SARSA
def SARSA(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    q_values = np.zeros((state_n, action_n))
    total_rewards = []
    trajectory_counts = []

    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n
        total_reward = 0
        state, _ = env.reset()
        action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)

        for t in range(trajectory_len):
            next_state, reward, done, _, _ = env.step(action)
            next_action = get_epsilon_greedy_action(
                q_values[next_state], epsilon, action_n
            )
            q_values[state][action] += alpha * (
                reward
                + gamma * q_values[next_state][next_action]
                - q_values[state][action]
            )
            total_reward += reward
            state = next_state
            action = next_action
            if done:
                break

        total_trajectories = episode + 1
        trajectory_counts.append(total_trajectories)
        total_rewards.append(total_reward)
        print(
            f"SARSA: Средняя награда после {total_trajectories} траекторий: {total_reward}"
        )

    return total_rewards, trajectory_counts


# Реализация Q-Learning
def QLearning(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    q_values = np.zeros((state_n, action_n))
    total_rewards = []
    trajectory_counts = []

    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n
        total_reward = 0
        state, _ = env.reset()

        for t in range(trajectory_len):
            action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
            next_state, reward, done, _, _ = env.step(action)
            q_values[state][action] += alpha * (
                reward + gamma * np.max(q_values[next_state]) - q_values[state][action]
            )
            total_reward += reward
            state = next_state
            if done:
                break

        total_trajectories = episode + 1
        trajectory_counts.append(total_trajectories)
        total_rewards.append(total_reward)
        print(
            f"Q-Learning: Средняя награда после {total_trajectories} траекторий: {total_reward}"
        )

    return total_rewards, trajectory_counts


# Запуск алгоритмов
episode_n = 2000
rewards_cross_entropy, trajectory_counts_ce = run_cross_entropy(env, episode_n=10)
rewards_monte_carlo, trajectory_counts_mc = MonteCarlo(env, episode_n)
rewards_sarsa, trajectory_counts_sarsa = SARSA(env, episode_n)
rewards_q_learning, trajectory_counts_ql = QLearning(env, episode_n)

# Построение графиков
plt.plot(trajectory_counts_ce, rewards_cross_entropy, label="Cross-Entropy")
plt.plot(trajectory_counts_mc, rewards_monte_carlo, label="Monte Carlo")
plt.plot(trajectory_counts_sarsa, rewards_sarsa, label="SARSA")
plt.plot(trajectory_counts_ql, rewards_q_learning, label="Q-Learning")

plt.xlabel("Количество траекторий")
plt.ylabel("Накопленное вознаграждение")
plt.legend()
plt.show()
