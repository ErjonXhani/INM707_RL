import random
from env.dungeon_env import DungeonEnv

env = DungeonEnv()

episodes = 5

for episode in range(episodes):
    state, _ = env.reset()
    done = False

    print(f"\nEpisode {episode + 1}")

    while not done:
        env.render()

        action = random.randint(0, 7)  # random action
        state, reward, done, _, _ = env.step(action)

        print("Action:", action, "Reward:", reward)

    print("Episode finished")