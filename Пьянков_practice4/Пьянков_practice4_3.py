import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Определение функции epsilon-greedy для выбора действий
def get_epsilon_greedy_action(q_values, epsilon, action_n):
    prob = np.ones(action_n) * epsilon / action_n
    argmax_action = np.argmax(q_values)
    prob[argmax_action] += 1 - epsilon
    action = np.random.choice(np.arange(action_n), p=prob)
    return action


# Инициализация среды
env = gym.make("Taxi-v3")
state_n = env.observation_space.n
action_n = env.action_space.n


# Реализация Monte Carlo с разными стратегиями epsilon
def MonteCarlo(env, episode_n, epsilon_strategy, gamma=0.99, trajectory_len=500):
    q_values = np.zeros((state_n, action_n))
    counters = np.zeros((state_n, action_n))
    total_rewards = []
    trajectory_counts = []

    epsilon_max = 1.0
    epsilon_min = 0.01
    decay_rate = 0.005
    epsilon = epsilon_max

    for episode in range(episode_n):
        # Определяем epsilon в зависимости от стратегии
        if epsilon_strategy == "exp_decay":
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(
                -decay_rate * episode
            )
        elif epsilon_strategy == "linear_decay":
            epsilon = max(
                epsilon_min,
                epsilon_max - (episode / episode_n) * (epsilon_max - epsilon_min),
            )
        elif epsilon_strategy == "adaptive":
            if len(total_rewards) >= 200:
                recent_rewards = total_rewards[-100:]
                previous_rewards = total_rewards[-200:-100]
                if np.mean(recent_rewards) > np.mean(previous_rewards):
                    epsilon = max(
                        epsilon_min, epsilon * 0.99
                    )  # Уменьшаем `ε` на 1% при успехе
                else:
                    epsilon = min(
                        epsilon_max, epsilon * 1.01
                    )  # Увеличиваем на 1% при снижении
            else:
                epsilon = (
                    epsilon_max  # На начальных этапах оставляем epsilon максимальным
                )

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

        total_rewards.append(np.sum(trajectory["rewards"]))
        trajectory_counts.append(episode + 1)

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


# Параметры обучения
episode_n = 2000

# Запуск алгоритма с разными стратегиями epsilon
strategies = ["exp_decay", "linear_decay", "adaptive"]
results = {}

for strategy in strategies:
    print(f"Запуск Monte Carlo с epsilon стратегией: {strategy}")
    rewards, trajectory_counts = MonteCarlo(env, episode_n, epsilon_strategy=strategy)
    results[strategy] = (rewards, trajectory_counts)

# Построение графиков для каждой стратегии
plt.figure(figsize=(12, 8))

for strategy, (rewards, counts) in results.items():
    plt.plot(counts, rewards, label=strategy)

plt.xlabel("Количество траекторий")
plt.ylabel("Средняя награда")
plt.title("Сравнение стратегий epsilon в Monte Carlo для задачи Taxi-v3")
plt.legend()
plt.grid()
plt.show()
