import random
import numpy as np
from env.dungeon_env import DungeonEnv
import matplotlib.pyplot as plt
import pickle
import os

env = DungeonEnv()

# Q-learning settings
episodes = 2000
alpha = 0.1       # learning rate
gamma = 0.95      # future reward importance
epsilon = 1.0     # exploration rate
epsilon_decay = 0.995
epsilon_min = 0.05

# Q-table: maps state -> action values
q_table = {}

episode_rewards = []


def state_to_key(state):
    return tuple(state.tolist())


def get_q_values(state):
    key = state_to_key(state)

    if key not in q_table:
        q_table[key] = np.zeros(env.action_space.n)

    return q_table[key]


for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        q_values = get_q_values(state)

        # epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_values)

        next_state, reward, done, truncated, info = env.step(action)

        next_q_values = get_q_values(next_state)

        # Q-learning update rule
        q_values[action] = q_values[action] + alpha * (
            reward + gamma * np.max(next_q_values) - q_values[action]
        )

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")


print("Training finished.")

os.makedirs("models", exist_ok=True)

with open("models/q_table.pkl", "wb") as file:
    pickle.dump(q_table, file)

print("Q-table saved to models/q_table.pkl")

plt.plot(episode_rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Q-learning Performance")
plt.show()