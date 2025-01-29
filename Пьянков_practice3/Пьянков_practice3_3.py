import numpy as np
import matplotlib.pyplot as plt
from Frozen_Lake import FrozenLakeEnv

env = FrozenLakeEnv()


# Функция для получения Q-значений
def get_q_values(values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                transition_prob = env.get_transition_prob(state, action, next_state)
                reward = env.get_reward(state, action, next_state)
                q_values[state][action] += transition_prob * (
                    reward + gamma * values[next_state]
                )
    return q_values


# Инициализация политики
def init_policy():
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        actions = env.get_possible_actions(state)
        for action in actions:
            policy[state][action] = 1 / len(actions)
    return policy


# Инициализация значений
def init_values():
    values = {}
    for state in env.get_all_states():
        values[state] = 0
    return values


# Оценка политики
def policy_evaluation(policy, gamma, L):
    values = init_values()
    for _ in range(L):
        new_values = init_values()
        for state in env.get_all_states():
            for action in env.get_possible_actions(state):
                q_values = get_q_values(values, gamma)
                new_values[state] += policy[state][action] * q_values[state][action]
        values = new_values
    return values


# Улучшение политики
def policy_improvement(q_values):
    policy = init_policy()
    for state in env.get_all_states():
        if len(env.get_possible_actions(state)) > 0:
            max_action = max(q_values[state], key=q_values[state].get)
            for action in env.get_possible_actions(state):
                policy[state][action] = 1 if action == max_action else 0
    return policy


# Функция для Policy Iteration с ограничением итераций
def policy_iteration(gamma, L, max_iterations=250):
    policy = init_policy()
    total_steps = 0
    rewards_per_step = []

    for iteration in range(max_iterations):
        # Оценка политики
        values = policy_evaluation(policy, gamma, L)

        # Получаем Q-значения
        q_values = get_q_values(values, gamma)

        # Улучшение политики
        new_policy = policy_improvement(q_values)

        # Проверка на сходимость
        if new_policy == policy:
            print(f"Policy iteration converged at iteration {iteration + 1}")
            break

        policy = new_policy

        # Запись результатов
        total_steps += len(env.get_all_states())
        avg_reward = test_policy(policy, episodes=100)
        rewards_per_step.append((total_steps, avg_reward))
        print(f"Policy Iteration Step {iteration + 1}: Avg Reward = {avg_reward}")

    return rewards_per_step


# Функция для Value Iteration с промежуточными выводами
def value_iteration(gamma, threshold=1e-6, max_iterations=1000):
    values = {state: 0 for state in env.get_all_states()}
    total_steps = 0
    rewards_per_step = []

    for iteration in range(max_iterations):
        new_values = values.copy()
        for state in env.get_all_states():
            possible_actions = env.get_possible_actions(state)
            if not possible_actions:
                continue

            q_values = get_q_values(values, gamma)
            max_q_value = max(q_values[state].values()) if q_values[state] else 0
            new_values[state] = max_q_value

        total_steps += len(env.get_all_states())

        q_values = get_q_values(new_values, gamma)
        test_policy_dict = {}
        for state in q_values:
            if q_values[state]:
                max_action = max(q_values[state], key=q_values[state].get)
                test_policy_dict[state] = {
                    a: 1 if a == max_action else 0
                    for a in env.get_possible_actions(state)
                }

        avg_reward = test_policy(test_policy_dict, episodes=100)
        rewards_per_step.append((total_steps, avg_reward))
        print(f"Value Iteration Step {iteration + 1}: Avg Reward = {avg_reward}")

        if (
            np.max(
                np.abs(
                    np.array(list(new_values.values()))
                    - np.array(list(values.values()))
                )
            )
            < threshold
        ):
            print(f"Value iteration converged at iteration {iteration + 1}")
            break

        values = new_values

    return rewards_per_step


def test_policy(policy_or_values, gamma=None, episodes=100):
    if isinstance(policy_or_values, dict) and all(
        isinstance(val, float) for val in policy_or_values.values()
    ):

        values = policy_or_values
        q_values = get_q_values(values, gamma)
        policy = {}
        for state in q_values:
            if q_values[state]:
                max_action = max(q_values[state], key=q_values[state].get)
                policy[state] = {
                    a: 1 if a == max_action else 0
                    for a in env.get_possible_actions(state)
                }
    else:
        policy = policy_or_values

    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(1000):
            action = np.random.choice(
                env.get_possible_actions(state), p=list(policy[state].values())
            )
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)
    return np.mean(total_rewards)


gamma = 0.99
L = 20

policy_rewards = policy_iteration(gamma, L)
value_rewards = value_iteration(gamma)

policy_steps, policy_avg_rewards = zip(*policy_rewards)
value_steps, value_avg_rewards = zip(*value_rewards)

plt.figure(figsize=(12, 6))
plt.plot(policy_steps, policy_avg_rewards, label="Policy Iteration", color="blue")
plt.plot(value_steps, value_avg_rewards, label="Value Iteration", color="orange")
plt.xlabel("Количество обращений к среде")
plt.ylabel("Средняя награда")
plt.title("Кривая обучения для Policy Iteration и Value Iteration")
plt.legend()
plt.grid()
plt.show()


"""
# ПЕРВАЯ ЧАСТЬ


import numpy as np
import matplotlib.pyplot as plt
from Frozen_Lake import FrozenLakeEnv

env = FrozenLakeEnv()


# Функция для получения Q-значений
def get_q_values(values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                q_values[state][action] += env.get_transition_prob(
                    state, action, next_state
                ) * (
                    env.get_reward(state, action, next_state)
                    + gamma * values[next_state]
                )
    return q_values


# Функция для Value Iteration
def value_iteration(gamma, threshold=1e-6):
    values = {state: 0 for state in env.get_all_states()}
    num_steps = 0

    while True:
        new_values = values.copy()
        for state in env.get_all_states():
            possible_actions = env.get_possible_actions(state)
            if not possible_actions:
                continue

            q_values = get_q_values(values, gamma)
            max_q_value = max(q_values[state].values()) if q_values[state] else 0
            new_values[state] = max_q_value

        num_steps += 1
        if (
            np.max(
                np.abs(
                    np.array(list(new_values.values()))
                    - np.array(list(values.values()))
                )
            )
            < threshold
        ):
            break
        values = new_values

    return values, num_steps


# Функция для тестирования политики
def test_policy(values, gamma):
    q_values = get_q_values(values, gamma)
    policy = {}
    for state in q_values:
        if q_values[state]:
            max_action = max(q_values[state], key=q_values[state].get)
            policy[state] = max_action
        else:
            policy[state] = None

    total_rewards = []
    for _ in range(1000):
        state = env.reset()
        total_reward = 0
        for _ in range(100):
            action = policy[state]
            if action is None:
                break
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)
    return np.mean(total_rewards)


# Основные параметры
gammas = [
    0.5,
    0.7,
    0.9,
    0.99,
    0.999,
    1.0,
]  # Примеры значений gamma для тестирования
thresholds = [
    1e-1,
    1e-2,
    1e-4,
    1e-5,
    1e-6,
    1e-9,
    1e-11,
]  # Примеры значений threshold для тестирования

results = {}

for gamma in gammas:
    results[gamma] = []
    for threshold in thresholds:
        values, _ = value_iteration(gamma, threshold)
        avg_reward = test_policy(values, gamma)
        results[gamma].append(avg_reward)

plt.figure(figsize=(10, 6))
for gamma, rewards in results.items():
    plt.plot(thresholds, rewards, marker="o", label=f"Gamma={gamma}")

plt.xlabel("Threshold")
plt.ylabel("Average Total Reward")
plt.title("Value Iteration Performance for Different Gammas and Thresholds")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.show()

"""
