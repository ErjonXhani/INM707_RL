import pickle
import pygame
import numpy as np
from env.dungeon_env import DungeonEnv

CELL_SIZE = 70
GRID_SIZE = 8
WIDTH = CELL_SIZE * GRID_SIZE
HEIGHT = CELL_SIZE * GRID_SIZE

pygame.init()
font = pygame.font.SysFont("arial", 24)
big_font = pygame.font.SysFont("arial", 42, bold=True)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Trained Q-learning Agent")
clock = pygame.time.Clock()

env = DungeonEnv()

with open("models/q_table.pkl", "rb") as file:
    q_table = pickle.load(file)


def state_to_key(state):
    return tuple(state.tolist())


def choose_best_action(state):
    key = state_to_key(state)

    if key not in q_table:
        return env.action_space.sample()

    return int(np.argmax(q_table[key]))


def draw_robot(x, y):
    px = x * CELL_SIZE
    py = y * CELL_SIZE

    body = pygame.Rect(px + 22, py + 28, 36, 34)
    head = pygame.Rect(px + 25, py + 10, 30, 25)

    pygame.draw.rect(screen, (60, 100, 220), body, border_radius=8)
    pygame.draw.rect(screen, (90, 140, 255), head, border_radius=6)

    pygame.draw.circle(screen, (255, 255, 255), (px + 34, py + 22), 4)
    pygame.draw.circle(screen, (255, 255, 255), (px + 46, py + 22), 4)

    pygame.draw.line(screen, (40, 40, 40), (px + 40, py + 10), (px + 40, py + 2), 3)
    pygame.draw.circle(screen, (255, 180, 0), (px + 40, py + 2), 4)


def draw_monster(x, y):
    px = x * CELL_SIZE
    py = y * CELL_SIZE
    center = (px + CELL_SIZE // 2, py + CELL_SIZE // 2)

    pygame.draw.circle(screen, (160, 40, 40), center, 28)
    pygame.draw.polygon(screen, (100, 20, 20), [(px + 20, py + 18), (px + 30, py + 2), (px + 38, py + 24)])
    pygame.draw.polygon(screen, (100, 20, 20), [(px + 60, py + 18), (px + 50, py + 2), (px + 42, py + 24)])

    pygame.draw.circle(screen, (255, 255, 255), (px + 32, py + 36), 6)
    pygame.draw.circle(screen, (255, 255, 255), (px + 48, py + 36), 6)
    pygame.draw.circle(screen, (0, 0, 0), (px + 32, py + 36), 3)
    pygame.draw.circle(screen, (0, 0, 0), (px + 48, py + 36), 3)

    pygame.draw.arc(screen, (0, 0, 0), (px + 28, py + 42, 24, 16), 0, 3.14, 3)


def draw_bullet(action):
    if action not in [4, 5, 6, 7]:
        return

    ax, ay = env.agent_pos

    if action == 4:
        dx, dy = 0, -1
    elif action == 5:
        dx, dy = 0, 1
    elif action == 6:
        dx, dy = -1, 0
    else:
        dx, dy = 1, 0

    x = ax + dx
    y = ay + dy

    while 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
        current_pos = (x, y)

        if current_pos in env.obstacles:
            break

        bullet_center = (
            x * CELL_SIZE + CELL_SIZE // 2,
            y * CELL_SIZE + CELL_SIZE // 2
        )

        pygame.draw.circle(screen, (255, 200, 0), bullet_center, 7)

        # stop at monster coordinate, even if it has just been killed
        if current_pos in env.enemy_positions:
            break

        x += dx
        y += dy


def draw_grid():
    screen.fill((240, 240, 240))

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)

    for ox, oy in env.obstacles:
        rect = pygame.Rect(ox * CELL_SIZE, oy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (80, 80, 80), rect)

    for i, (ex, ey) in enumerate(env.enemy_positions):
        if env.enemies_alive[i] == 1:
            draw_monster(ex, ey)

    ax, ay = env.agent_pos
    draw_robot(ax, ay)

def draw_text(text, x, y, color=(20, 20, 20), large=False):
    selected_font = big_font if large else font
    surface = selected_font.render(text, True, color)
    screen.blit(surface, (x, y))


def draw_hud(episode, total_reward, steps):
    enemies_left = sum(env.enemies_alive)

    draw_text(f"Episode: {episode}", 10, 10)
    draw_text(f"Reward: {total_reward}", 10, 35)
    draw_text(f"Steps: {steps}", 10, 60)
    draw_text(f"Enemies left: {enemies_left}", 10, 85)


def draw_end_message(message):
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(170)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    text_surface = big_font.render(message, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text_surface, text_rect)
    pygame.display.flip()

def draw_explosion(x, y):
    center = (
        x * CELL_SIZE + CELL_SIZE // 2,
        y * CELL_SIZE + CELL_SIZE // 2
    )

    for radius in [10, 20, 30]:
        draw_grid()
        pygame.draw.circle(screen, (255, 120, 0), center, radius)
        pygame.draw.circle(screen, (255, 220, 0), center, radius // 2)

        draw_hud(episode, total_reward, steps)
        pygame.display.flip()
        pygame.time.delay(120)


state, _ = env.reset()
done = False
last_action = None

episode = 1
total_reward = 0
steps = 0
end_message = ""

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not done:
        action = choose_best_action(state)

        old_enemies_alive = env.enemies_alive.copy()

        state, reward, done, _, _ = env.step(action)

        total_reward += reward
        steps += 1
        last_action = action

        draw_grid()

        if last_action is not None:
            draw_bullet(last_action)

        draw_hud(episode, total_reward, steps)
        pygame.display.flip()
        pygame.time.delay(180)

        # Monster death explosion
        for i, alive_before in enumerate(old_enemies_alive):
            if alive_before == 1 and env.enemies_alive[i] == 0:
                ex, ey = env.enemy_positions[i]
                draw_explosion(ex, ey)

        # Robot death explosion
        if done and sum(env.enemies_alive) > 0:
            ax, ay = env.agent_pos
            draw_explosion(ax, ay)

        # End message + reset
        if done:
            if sum(env.enemies_alive) == 0:
                end_message = "ROOM CLEARED!"
            else:
                end_message = "ROBOT DESTROYED!"

            draw_end_message(end_message)
            pygame.time.delay(1200)

            episode += 1
            total_reward = 0
            steps = 0
            state, _ = env.reset()
            done = False
            last_action = None

    clock.tick(2)

pygame.quit()