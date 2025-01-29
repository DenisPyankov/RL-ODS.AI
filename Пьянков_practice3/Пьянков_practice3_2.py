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


def policy_evaluation(policy, gamma, L, values_init=None):
    values = values_init if values_init else init_values()
    for _ in range(L):
        new_values = init_values()
        for state in env.get_all_states():
            for action in env.get_possible_actions(state):
                q_values = get_q_values(values, gamma)
                new_values[state] += policy[state][action] * q_values[state][action]
        values = new_values
    return values, (
        get_q_values(values, gamma) if values_init else get_q_values(values, gamma)
    )


def policy_improvement(q_values):
    policy = init_policy()
    for state in env.get_all_states():
        if len(env.get_possible_actions(state)) > 0:
            max_i = np.argmax(list(q_values[state].values()))
            max_action = env.get_possible_actions(state)[max_i]
            for action in env.get_possible_actions(state):
                policy[state][action] = 1 if action == max_action else 0
    return policy


# Основные параметры
K = 20  # Число итераций улучшения политики
L = 20  # Число итераций оценки политики
gamma = 0.9999

policy = init_policy()
values_prev = init_values()

for k in range(K):
    values_prev, q_values = policy_evaluation(policy, gamma, L, values_prev)
    policy = policy_improvement(q_values)

num_runs = 25
episodes_per_run = 1000
avg_rewards = []

for run in range(num_runs):
    total_rewards = []
    for _ in range(episodes_per_run):
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
    avg_rewards.append(np.mean(total_rewards))

plt.plot(range(1, num_runs + 1), avg_rewards, marker="o")
plt.xlabel("Run")
plt.ylabel("Average Total Reward")
plt.title("Average Total Reward for 50 Runs")
plt.grid(True)
plt.xticks(range(1, num_runs + 1))
plt.show()

print(f"Mean total reward over {num_runs} runs: {np.mean(avg_rewards):.3f}")
