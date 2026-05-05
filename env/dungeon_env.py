import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class DungeonEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 8
        self.max_steps = 80
        self.action_space = spaces.Discrete(8)

        # State: agent_x, agent_y, enemy1_alive, enemy2_alive, enemy3_alive, enemy4_alive, enemy5_alive
        self.observation_space = spaces.Box(
            low=0,
            high=8,
            shape=(7,),
            dtype=np.int32
        )

        self.obstacles = [
            (1, 1), (1, 2), (3, 3), (4, 3),
            (6, 1), (2, 6), (5, 5), (6, 6)
        ]

        self.enemy_positions = [
            (7, 7),
            (2, 5),
            (6, 4),
            (4, 1),
            (0, 6)
        ]

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        possible_spawns = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.obstacles and (x, y) not in self.enemy_positions
        ]

        self.agent_pos = random.choice(possible_spawns)
        self.enemies_alive = [1 for _ in self.enemy_positions]
        self.steps = 0

        return self._get_state(), {}

    def _get_state(self):
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            *self.enemies_alive
        ], dtype=np.int32)

    def step(self, action):
        self.steps += 1
        reward = -1
        done = False

        if action in [0, 1, 2, 3]:
            reward += self._move(action)

        elif action in [4, 5, 6, 7]:
            reward += self._shoot(action)
        
        # death condition

        if self.agent_pos in self.enemy_positions:
            enemy_index = self.enemy_positions.index(self.agent_pos)

            if self.enemies_alive[enemy_index] == 1:
                reward -= 100
                done = True

        if sum(self.enemies_alive) == 0:
            reward += 100
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, False, {}

    def _move(self, action):
        x, y = self.agent_pos

        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1

        new_pos = (x, y)

        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return -10

        if new_pos in self.obstacles:
            return -10

        self.agent_pos = new_pos
        return 0

    def _shoot(self, action):
        reward = -2

        ax, ay = self.agent_pos

        if action == 4:      # shoot up
            dx, dy = 0, -1
        elif action == 5:    # shoot down
            dx, dy = 0, 1
        elif action == 6:    # shoot left
            dx, dy = -1, 0
        elif action == 7:    # shoot right
            dx, dy = 1, 0
        else:
            return reward

        x = ax + dx
        y = ay + dy

        while 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            current_pos = (x, y)

        # Bullet stops when it hits an obstacle
            if current_pos in self.obstacles:
                break

        # Bullet kills first living enemy in its path
            if current_pos in self.enemy_positions:
                enemy_index = self.enemy_positions.index(current_pos)

                if self.enemies_alive[enemy_index] == 1:
                    self.enemies_alive[enemy_index] = 0
                    reward += 30
                    break

            x += dx
            y += dy

        return reward

    def render(self):
        for y in range(self.grid_size):
            row = ""
            for x in range(self.grid_size):
                pos = (x, y)

                if pos == self.agent_pos:
                    row += "A "
                elif pos in self.obstacles:
                    row += "X "
                elif pos in self.enemy_positions:
                    enemy_index = self.enemy_positions.index(pos)
                    if self.enemies_alive[enemy_index] == 1:
                        row += "E "
                    else:
                        row += ". "
                else:
                    row += ". "
            print(row)
        print()