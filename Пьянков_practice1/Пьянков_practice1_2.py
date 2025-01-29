import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt


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
        return None


class LaplaceSmoothingAgent(CrossEntropyAgent):
    def __init__(self, state_n, action_n, alpha=1):
        super().__init__(state_n, action_n)
        self.alpha = alpha  # Параметр сглаживания Лапласа

    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                new_model[state][action] += 1

        # Применение Лапласова сглаживания
        for state in range(self.state_n):
            new_model[state] = (new_model[state] + self.alpha) / (
                np.sum(new_model[state]) + self.alpha * self.action_n
            )

        self.model = new_model
        return None


class PolicySmoothingAgent(CrossEntropyAgent):
    def __init__(self, state_n, action_n, epsilon=0.1):
        super().__init__(state_n, action_n)
        self.epsilon = epsilon  # Параметр случайности для сглаживания

    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                new_model[state][action] += 1

        # Применение сглаживания политики
        for state in range(self.state_n):
            total = np.sum(new_model[state])
            if total > 0:
                new_model[state] /= total
            else:
                new_model[state] = self.model[state].copy()

            # Добавление случайности
            new_model[state] = (1 - self.epsilon) * new_model[
                state
            ] + self.epsilon / self.action_n

        self.model = new_model
        return None


def get_trajectory(env, agent, max_len=1000, visualize=False):
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

        if visualize:
            time.sleep(0.1)
            print(env.render())

        if done:
            break

    return trajectory


# Инициализация среды
env = gym.make("Taxi-v3")
state_n = env.observation_space.n
action_n = env.action_space.n

# Параметры
q_param = 0.8
iteration_n = 100
trajectory_n = 100

# Для хранения результатов
rewards_no_smoothing = []
rewards_laplace = []
rewards_policy = []

# Обучение без сглаживания
agent_no_smoothing = CrossEntropyAgent(state_n, action_n)
for iteration in range(iteration_n):
    trajectories = [
        get_trajectory(env, agent_no_smoothing) for _ in range(trajectory_n)
    ]
    total_rewards = [np.sum(t["rewards"]) for t in trajectories]
    rewards_no_smoothing.append(np.mean(total_rewards))
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = [t for t in trajectories if np.sum(t["rewards"]) > quantile]
    agent_no_smoothing.fit(elite_trajectories)

# Обучение с Laplace smoothing
agent_laplace = LaplaceSmoothingAgent(state_n, action_n, alpha=1)
for iteration in range(iteration_n):
    trajectories = [get_trajectory(env, agent_laplace) for _ in range(trajectory_n)]
    total_rewards = [np.sum(t["rewards"]) for t in trajectories]
    rewards_laplace.append(np.mean(total_rewards))
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = [t for t in trajectories if np.sum(t["rewards"]) > quantile]
    agent_laplace.fit(elite_trajectories)

# Обучение с Policy smoothing
agent_policy = PolicySmoothingAgent(state_n, action_n, epsilon=0.1)
for iteration in range(iteration_n):
    trajectories = [get_trajectory(env, agent_policy) for _ in range(trajectory_n)]
    total_rewards = [np.sum(t["rewards"]) for t in trajectories]
    rewards_policy.append(np.mean(total_rewards))
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = [t for t in trajectories if np.sum(t["rewards"]) > quantile]
    agent_policy.fit(elite_trajectories)

# Визуализация результатов
plt.plot(rewards_no_smoothing, label="No Smoothing")
plt.plot(rewards_laplace, label="Laplace Smoothing")
plt.plot(rewards_policy, label="Policy Smoothing")
plt.xlabel("Iterations")
plt.ylabel("Mean Total Reward")
plt.title("Comparison of Cross Entropy Methods")
plt.legend()
plt.show()
