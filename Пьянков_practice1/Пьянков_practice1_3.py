import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Создание среды Taxi-v3
env = gym.make("Taxi-v3")
state_n = env.observation_space.n
action_n = env.action_space.n


class CrossEntropyAgent:
    def __init__(self, state_n, action_n, n_samples=3):
        self.state_n = state_n
        self.action_n = action_n
        self.model = (
            np.ones((self.state_n, self.action_n)) / self.action_n
        )  # Изначальная политика
        self.n_samples = n_samples  # Количество сэмплов детерминированных политик

    def sample_deterministic_policy(self):
        deterministic_policies = []
        for state in range(self.state_n):
            # Сэмплирование действий на основе текущих вероятностей
            actions = np.random.choice(
                self.action_n, self.n_samples, p=self.model[state]
            )
            # Преобразование в детерминированные политики
            for action in actions:
                deterministic_policy = np.zeros(self.action_n)
                deterministic_policy[action] = 1.0
                deterministic_policies.append((state, deterministic_policy))
        return deterministic_policies

    def get_trajectories(self, env, policies, max_len=1000):
        trajectories = []
        for state, policy in policies:
            trajectory = {"states": [], "actions": [], "rewards": []}
            obs, _ = env.reset()
            for _ in range(max_len):
                trajectory["states"].append(obs)
                action = np.argmax(
                    policy
                )  # Получение действия из детерминированной политики
                trajectory["actions"].append(action)
                obs, reward, done, _, _ = env.step(action)
                trajectory["rewards"].append(reward)
                if done:
                    break
            trajectories.append(trajectory)
        return trajectories

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

    def run(self, iterations=100, trajectories_per_iteration=100, elite_fraction=0.2):
        rewards_per_iteration = []

        for iteration in range(iterations):
            # Сэмплирование детерминированных политик
            policies = self.sample_deterministic_policy()

            # Получение траекторий
            trajectories = self.get_trajectories(env, policies)

            # Оценка траекторий
            total_rewards = [
                np.sum(trajectory["rewards"]) for trajectory in trajectories
            ]
            rewards_per_iteration.append(np.mean(total_rewards))

            # Отбор элитных траекторий
            threshold = np.quantile(total_rewards, 1 - elite_fraction)
            elite_trajectories = [
                trajectory
                for trajectory in trajectories
                if np.sum(trajectory["rewards"]) > threshold
            ]

            # Обучение модели на элитных траекториях
            self.fit(elite_trajectories)

            print(
                f"Iteration: {iteration}, Mean Total Reward: {np.mean(total_rewards)}"
            )

        return rewards_per_iteration


# Инициализация агента
agent = CrossEntropyAgent(state_n, action_n)

# Запуск обучения
rewards = agent.run(iterations=100)

# Визуализация результатов
plt.plot(rewards)
plt.xlabel("Iterations")
plt.ylabel("Mean Total Reward")
plt.title("Training Progress of Cross Entropy Agent on Taxi-v3")
plt.show()
