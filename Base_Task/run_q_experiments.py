import os
import random
import numpy as np
import matplotlib.pyplot as plt
from env.dungeon_env import DungeonEnv


def state_to_key(state):
    return tuple(state.tolist())


def moving_average(values, window=100):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def train_q_learning(alpha, gamma, epsilon_decay, episodes=2000):
    env = DungeonEnv()

    epsilon = 1.0
    epsilon_min = 0.05

    q_table = {}
    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    def get_q_values(state):
        key = state_to_key(state)

        if key not in q_table:
            q_table[key] = np.zeros(env.action_space.n)

        return q_table[key]

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        total_reward = 0
        steps = 0
        result = "running"

        while not done:
            q_values = get_q_values(state)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_values))

            next_state, reward, done, _, info = env.step(action)
            next_q_values = get_q_values(next_state)

            q_values[action] = q_values[action] + alpha * (
                reward + gamma * np.max(next_q_values) - q_values[action]
            )

            state = next_state
            total_reward += reward
            steps += 1

            result = info["result"]

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_successes.append(1 if result == "cleared" else 0)

    return episode_rewards, episode_lengths, episode_successes


experiments = {
    "Baseline α=0.1 γ=0.95 decay=0.995": {
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon_decay": 0.995
    },
    "High alpha α=0.5": {
        "alpha": 0.5,
        "gamma": 0.95,
        "epsilon_decay": 0.995
    },
    "Low gamma γ=0.8": {
        "alpha": 0.1,
        "gamma": 0.8,
        "epsilon_decay": 0.995
    },
    "Faster decay ε=0.99": {
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon_decay": 0.99
    }
}


os.makedirs("results", exist_ok=True)

all_results = {}

for name, params in experiments.items():
    print(f"Running experiment: {name}")

    rewards, lengths, successes = train_q_learning(
        alpha=params["alpha"],
        gamma=params["gamma"],
        epsilon_decay=params["epsilon_decay"]
    )

    all_results[name] = {
        "rewards": rewards,
        "lengths": lengths,
        "successes": successes
    }


# Reward plot
plt.figure(figsize=(12, 7))

for name, results in all_results.items():
    plt.plot(moving_average(results["rewards"]), label=name)

plt.xlabel("Episode")
plt.ylabel("Average Reward over 100 Episodes")
plt.title("Q-learning Reward Comparison")
plt.legend()
plt.grid(True)
plt.savefig("results/q_learning_reward_comparison.png")
plt.show()


# Episode length plot
plt.figure(figsize=(12, 7))

for name, results in all_results.items():
    plt.plot(moving_average(results["lengths"]), label=name)

plt.xlabel("Episode")
plt.ylabel("Average Episode Length over 100 Episodes")
plt.title("Q-learning Episode Length Comparison")
plt.legend()
plt.grid(True)
plt.savefig("results/q_learning_episode_length_comparison.png")
plt.show()


# Success rate plot
plt.figure(figsize=(12, 7))

for name, results in all_results.items():
    success_rate = moving_average(results["successes"]) * 100
    plt.plot(success_rate, label=name)

plt.xlabel("Episode")
plt.ylabel("Room Clear Success Rate (%) over 100 Episodes")
plt.title("Q-learning Success Rate Comparison")
plt.legend()
plt.grid(True)
plt.savefig("results/q_learning_success_rate_comparison.png")
plt.show()


print("\nExperiment Summary")
print("------------------")

for name, results in all_results.items():
    final_avg_reward = np.mean(results["rewards"][-100:])
    final_avg_length = np.mean(results["lengths"][-100:])
    final_success_rate = np.mean(results["successes"][-100:]) * 100

    print(name)
    print(f"Final average reward: {final_avg_reward:.2f}")
    print(f"Final average episode length: {final_avg_length:.2f}")
    print(f"Final success rate: {final_success_rate:.2f}%")
    print()