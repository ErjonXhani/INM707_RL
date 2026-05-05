import pickle
import pygame
import numpy as np
from env.dungeon_env import DungeonEnv

CELL_SIZE = 70
GRID_SIZE = 8

GRID_WIDTH = CELL_SIZE * GRID_SIZE
PANEL_WIDTH = 220

WIDTH = GRID_WIDTH + PANEL_WIDTH
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
    screen.fill((25, 25, 30))

    # floor tiles
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if (x + y) % 2 == 0:
                pygame.draw.rect(screen, (55, 55, 65), rect)
            else:
                pygame.draw.rect(screen, (48, 48, 58), rect)

            pygame.draw.rect(screen, (80, 80, 90), rect, 1)

    # obstacles as wall blocks
    for ox, oy in env.obstacles:
        rect = pygame.Rect(ox * CELL_SIZE + 4, oy * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8)
        pygame.draw.rect(screen, (95, 95, 105), rect, border_radius=6)
        pygame.draw.rect(screen, (45, 45, 55), rect, 3, border_radius=6)

    # enemies
    for i, (ex, ey) in enumerate(env.enemy_positions):
        if env.enemies_alive[i] == 1:
            draw_monster(ex, ey)

    # agent
    ax, ay = env.agent_pos
    draw_robot(ax, ay)

def draw_text(text, x, y, color=(20, 20, 20), large=False):
    selected_font = big_font if large else font
    surface = selected_font.render(text, True, color)
    screen.blit(surface, (x, y))


def draw_hud(episode, total_reward, steps):
    panel_x = GRID_WIDTH

    panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, HEIGHT)
    pygame.draw.rect(screen, (35, 35, 42), panel_rect)

    pygame.draw.line(screen, (120, 120, 130), (panel_x, 0), (panel_x, HEIGHT), 3)

    enemies_left = sum(env.enemies_alive)

    draw_text("RL AGENT", panel_x + 25, 30, color=(255, 255, 255), large=True)

    draw_text(f"Episode:", panel_x + 25, 110, color=(180, 180, 190))
    draw_text(str(episode), panel_x + 25, 140, color=(255, 255, 255))

    draw_text(f"Reward:", panel_x + 25, 190, color=(180, 180, 190))
    draw_text(str(total_reward), panel_x + 25, 220, color=(255, 220, 120))

    draw_text(f"Steps:", panel_x + 25, 270, color=(180, 180, 190))
    draw_text(str(steps), panel_x + 25, 300, color=(255, 255, 255))

    draw_text(f"Enemies left:", panel_x + 25, 350, color=(180, 180, 190))
    draw_text(str(enemies_left), panel_x + 25, 380, color=(255, 120, 120))


def draw_end_message(message):
    overlay = pygame.Surface((GRID_WIDTH, HEIGHT))
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    if message == "ROOM CLEARED!":
        color = (120, 255, 150)
    else:
        color = (255, 100, 100)

    text_surface = big_font.render(message, True, color)
    text_rect = text_surface.get_rect(center=(GRID_WIDTH // 2, HEIGHT // 2))
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

        state, reward, done, _, info = env.step(action)

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
            if info["result"] == "cleared":
                end_message = "ROOM CLEARED!"
            elif info["result"] == "dead":
                end_message = "ROBOT DESTROYED!"
            elif info["result"] == "timeout":
                end_message = "TIME OUT!"
            else:
                end_message = "EPISODE ENDED!"

            draw_end_message(end_message)
            pygame.time.delay(1200)

            episode += 1
            total_reward = 0
            steps = 0
            state, _ = env.reset()
            done = False
            last_action = None

    clock.tick(3)

pygame.quit()