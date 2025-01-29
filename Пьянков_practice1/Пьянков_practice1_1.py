import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt


env = gym.make("Taxi-v3", render_mode="ansi")
state_n = env.observation_space.n  # Количество состояний
action_n = env.action_space.n  # Количество действий


class CrossEntropyAgent:
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        # Выбор действия на основе текущей модели вероятностей
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)

    def fit(self, elite_trajectories):
        # Обновление модели на основе элитных траекторий
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                new_model[state][action] += 1

        # Нормализация для получения вероятностей
        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model
        return None


def get_trajectory(env, agent, max_len=1000, visualize=True):
    trajectory = {"states": [], "actions": [], "rewards": []}

    obs, _ = env.reset()  # Сброс среды и получение начального состояния
    state = obs

    for _ in range(max_len):
        trajectory["states"].append(state)

        action = agent.get_action(state)
        trajectory["actions"].append(action)

        obs, reward, done, _, _ = env.step(action)
        trajectory["rewards"].append(reward)

        state = obs

        # if visualize:
        #    time.sleep(0.1)  # Опциональная пауза
        #    print(env.render())  # Вывод текста рендеринга

        if done:
            break

    return trajectory


# Параметры
iteration_n = 100
max_len = 100
# params = {"sizes": [50], "quantiles": [0.8]}
params = {"sizes": [10, 50, 100], "quantiles": [0.7, 0.8, 0.9]}
# params = {"sizes": [10], "quantiles": [0.7, 0.8, 0.9]}
# params = {"sizes": [50], "quantiles": [0.7, 0.8, 0.9]}
# params = {"sizes": [100], "quantiles": [0.7, 0.8, 0.9]}
# params = {"sizes": [10, 50, 100], "quantiles": [0.7]}
# params = {"sizes": [10, 50, 100], "quantiles": [0.8]}
# params = {"sizes": [10, 50, 100], "quantiles": [0.9]}


def run_experiment(size, quantile):
    agent = CrossEntropyAgent(state_n, action_n)
    rewards_per_iteration = []

    for iteration in range(iteration_n):
        # Генерация траекторий
        trajectories = [get_trajectory(env, agent, max_len) for _ in range(size)]
        total_rewards = [np.sum(t["rewards"]) for t in trajectories]
        rewards_per_iteration.append(np.mean(total_rewards))
        print(f"Iteration: {iteration}, Mean Total Reward: {np.mean(total_rewards)}")

        # Отбор элитных траекторий
        quantile_value = np.quantile(total_rewards, quantile)
        elite_trajectories = [
            t for t, r in zip(trajectories, total_rewards) if r > quantile_value
        ]

        # Обновление модели
        agent.fit(elite_trajectories)

    # Тестирование обученного агента
    print("size = ", size, "quantile = ", quantile)
    trajectory = get_trajectory(env, agent, max_len=100, visualize=True)
    print("Total reward:", sum(trajectory["rewards"]))
    print("Final model probabilities:")
    print(agent.model)

    return rewards_per_iteration


# Сбор данных
results = {}
for size in params["sizes"]:
    for quantile in params["quantiles"]:
        key = f"size={size}, quantile={quantile}"
        results[key] = run_experiment(size, quantile)

# Визуализация на одном графике
plt.figure(figsize=(10, 6))
for key, rewards in results.items():
    plt.plot(rewards, label=key)

plt.xlabel("Iteration")
plt.ylabel("Mean Total Reward")
plt.title("Taxi-v3 Cross-Entropy: Hyperparameter Tuning")
plt.legend()
plt.show()
