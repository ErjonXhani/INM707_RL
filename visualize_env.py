import pygame
import random
from env.dungeon_env import DungeonEnv

CELL_SIZE = 80
GRID_SIZE = 6
WIDTH = CELL_SIZE * GRID_SIZE
HEIGHT = CELL_SIZE * GRID_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dungeon RL Environment")
clock = pygame.time.Clock()

env = DungeonEnv()
state, _ = env.reset()
done = False

def draw_robot(x, y):
    px = x * CELL_SIZE
    py = y * CELL_SIZE

    body = pygame.Rect(px + 22, py + 28, 36, 34)
    head = pygame.Rect(px + 25, py + 10, 30, 25)

    pygame.draw.rect(screen, (60, 100, 220), body, border_radius=8)
    pygame.draw.rect(screen, (90, 140, 255), head, border_radius=6)

    # eyes
    pygame.draw.circle(screen, (255, 255, 255), (px + 34, py + 22), 4)
    pygame.draw.circle(screen, (255, 255, 255), (px + 46, py + 22), 4)

    # antenna
    pygame.draw.line(screen, (40, 40, 40), (px + 40, py + 10), (px + 40, py + 2), 3)
    pygame.draw.circle(screen, (255, 180, 0), (px + 40, py + 2), 4)


def draw_monster(x, y):
    px = x * CELL_SIZE
    py = y * CELL_SIZE

    center = (px + CELL_SIZE // 2, py + CELL_SIZE // 2)

    # body
    pygame.draw.circle(screen, (160, 40, 40), center, 28)

    # horns
    pygame.draw.polygon(screen, (100, 20, 20), [(px + 20, py + 18), (px + 30, py + 2), (px + 38, py + 24)])
    pygame.draw.polygon(screen, (100, 20, 20), [(px + 60, py + 18), (px + 50, py + 2), (px + 42, py + 24)])

    # eyes
    pygame.draw.circle(screen, (255, 255, 255), (px + 32, py + 36), 6)
    pygame.draw.circle(screen, (255, 255, 255), (px + 48, py + 36), 6)
    pygame.draw.circle(screen, (0, 0, 0), (px + 32, py + 36), 3)
    pygame.draw.circle(screen, (0, 0, 0), (px + 48, py + 36), 3)

    # mouth
    pygame.draw.arc(screen, (0, 0, 0), (px + 28, py + 42, 24, 16), 0, 3.14, 3)


def draw_grid():
    screen.fill((240, 240, 240))

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)

    # obstacles
    for ox, oy in env.obstacles:
        rect = pygame.Rect(ox * CELL_SIZE, oy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (80, 80, 80), rect)

    # enemies
        # enemies
    for i, (ex, ey) in enumerate(env.enemy_positions):
        if env.enemies_alive[i] == 1:
            draw_monster(ex, ey)

    # agent
        # agent
    ax, ay = env.agent_pos
    draw_robot(ax, ay)
    

def draw_bullet(action):
    if action not in [4, 5, 6, 7]:
        return

    ax, ay = env.agent_pos
    bx = ax * CELL_SIZE + CELL_SIZE // 2
    by = ay * CELL_SIZE + CELL_SIZE // 2

    if action == 4:
        end = (bx, 0)
    elif action == 5:
        end = (bx, HEIGHT)
    elif action == 6:
        end = (0, by)
    elif action == 7:
        end = (WIDTH, by)

    pygame.draw.line(screen, (255, 200, 0), (bx, by), end, 5)

last_action = None
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not done:
        last_action = random.randint(0, 7)
        state, reward, done, _, _ = env.step(last_action)
    else:
        pygame.time.delay(1000)
        state, _ = env.reset()
        done = False
        last_action = None

    draw_grid()

    if last_action is not None:
        draw_bullet(last_action)

    pygame.display.flip()
    clock.tick(2)

pygame.quit()