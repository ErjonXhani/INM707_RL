import os
import random
import numpy as np
import matplotlib.pyplot as plt
from env.dungeon_env import DungeonEnv


def state_to_key(state):
    return tuple(state.tolist())


def train_q_learning(alpha, gamma, epsilon_decay, episodes=2000):
    env = DungeonEnv()

    epsilon = 1.0
    epsilon_min = 0.05

    q_table = {}
    episode_rewards = []
    episode_lengths = []
    success_count = 0

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
        cleared_room = False

        while not done:
            q_values = get_q_values(state)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_values))

            next_state, reward, done, _, _ = env.step(action)
            next_q_values = get_q_values(next_state)

            q_values[action] = q_values[action] + alpha * (
                reward + gamma * np.max(next_q_values) - q_values[action]
            )

            state = next_state
            total_reward += reward
            steps += 1

            if sum(env.enemies_alive) == 0:
                cleared_room = True

        if cleared_room:
            success_count += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    success_rate = success_count / episodes

    return episode_rewards, episode_lengths, success_rate


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

summary_results = {}

plt.figure(figsize=(12, 7))

for name, params in experiments.items():
    print(f"Running experiment: {name}")

    rewards, lengths, success_rate = train_q_learning(
        alpha=params["alpha"],
        gamma=params["gamma"],
        epsilon_decay=params["epsilon_decay"]
    )

    moving_average = np.convolve(rewards, np.ones(100) / 100, mode="valid")

    plt.plot(moving_average, label=name)

    summary_results[name] = {
        "final_average_reward": np.mean(rewards[-100:]),
        "average_episode_length": np.mean(lengths[-100:]),
        "success_rate": success_rate
    }

plt.xlabel("Episode")
plt.ylabel("Average Reward over 100 Episodes")
plt.title("Q-learning Parameter Experiment Comparison")
plt.legend()
plt.grid(True)
plt.savefig("results/q_learning_experiment_comparison.png")
plt.show()


print("\nExperiment Summary")
print("------------------")

for name, results in summary_results.items():
    print(name)
    print(f"Final average reward: {results['final_average_reward']:.2f}")
    print(f"Average episode length: {results['average_episode_length']:.2f}")
    print(f"Success rate: {results['success_rate'] * 100:.2f}%")
    print()