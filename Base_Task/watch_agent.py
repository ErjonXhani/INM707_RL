import time
import os
import random
from env.dungeon_env import DungeonEnv

env = DungeonEnv()
state, _ = env.reset()
done = False

while not done:
    os.system("cls")  # Windows clear screen
    env.render()
    time.sleep(0.4)

    action = random.randint(0, 7)
    state, reward, done, _, _ = env.step(action)

    print("Action:", action)
    print("Reward:", reward)

print("Episode finished")
env.render()