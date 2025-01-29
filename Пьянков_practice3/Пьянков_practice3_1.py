import numpy as np
import matplotlib.pyplot as plt
from Frozen_Lake import FrozenLakeEnv

env = FrozenLakeEnv()


# Функции для Policy Iteration
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


def init_policy():
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        for action in env.get_possible_actions(state):
            policy[state][action] = 1 / len(env.get_possible_actions(state))
    return policy


def init_values():
    values = {}
    for state in env.get_all_states():
        values[state] = 0
    return values


def policy_evaluation(policy, gamma, L):
    values = init_values()
    for _ in range(L):
        new_values = init_values()
        for state in env.get_all_states():
            for action in env.get_possible_actions(state):
                q_values = get_q_values(values, gamma)
                new_values[state] += policy[state][action] * q_values[state][action]
        values = new_values
    return get_q_values(values, gamma)


def policy_improvement(q_values):
    policy = init_policy()
    for state in env.get_all_states():
        if len(env.get_possible_actions(state)) > 0:
            max_i = np.argmax(list(q_values[state].values()))
            max_action = env.get_possible_actions(state)[max_i]
            for action in env.get_possible_actions(state):
                policy[state][action] = 1 if action == max_action else 0
    return policy


# Функция для тестирования политики
def test_policy(policy, episodes=1000):
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(1000):
            action_i = np.random.choice(np.arange(4), p=list(policy[state].values()))
            action = env.get_possible_actions(state)[action_i]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)
    return np.mean(total_rewards)


# Основные параметры
K = 20  # Число итераций улучшения политики
L = 20  # Число итераций оценки политики
gammas = np.arange(0.999, 1.0, 0.0001)  # Значения gamma для тестирования

avg_rewards = []

for gamma in gammas:
    policy = init_policy()

    for k in range(K):
        q_values = policy_evaluation(policy, gamma, L)
        policy = policy_improvement(q_values)

    avg_reward = test_policy(policy)
    avg_rewards.append(avg_reward)
    print(f"Gamma: {gamma:.3f}, Avg Reward: {avg_reward:.3f}")

plt.plot(gammas, avg_rewards, marker="o")
plt.xlabel("Gamma")
plt.ylabel("Average Total Reward")
plt.title("Effect of Gamma on Policy Performance")
plt.xticks(gammas)
plt.grid(True)
plt.show()
