import pygame
import math
import random
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 0

# Game States
GAME_STATE_PLAYING = "PLAYING"
GAME_STATE_GAME_OVER = "GAME_OVER"
GAME_STATE_ROOM_CLEARED = "ROOM_CLEARED"
GAME_STATE_PAUSED = "PAUSED"
game_state = GAME_STATE_PLAYING

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (150, 150, 150)
DARK_GRAY = (100, 100, 100)
PURPLE = (150, 0, 150)

# --- Image Asset Loading ---
IMAGES = {}
try:
    pygame.init()
    IMAGES['player'] = pygame.transform.scale(pygame.image.load('player.png'), (50, 50))
    IMAGES['bullet'] = pygame.transform.scale(pygame.image.load('bullet.png'), (10, 10))
    IMAGES['enemy_bullet'] = pygame.transform.scale(pygame.image.load('enemy_bullet.png'), (12, 12))
    IMAGES['normal'] = pygame.transform.scale(pygame.image.load('enemy_normal.png'), (40, 40))
    IMAGES['fast'] = pygame.transform.scale(pygame.image.load('enemy_fast.png'), (30, 30))
    IMAGES['shooter'] = pygame.transform.scale(pygame.image.load('enemy_shooter.png'), (50, 50))
    print("Assets loaded successfully.")
except Exception as e:
    print(f"Asset loading failed ({e}). Falling back to colored shapes.")

# Player Setup
PLAYER_SIZE = 50
PLAYER_SPEED = 5
player_x_initial = float((SCREEN_WIDTH - PLAYER_SIZE) // 2)
player_y_initial = float((SCREEN_HEIGHT - PLAYER_SIZE) // 2)
player_x, player_y = player_x_initial, player_y_initial
player_rect = pygame.Rect(int(player_x), int(player_y), PLAYER_SIZE, PLAYER_SIZE)
player_health_initial = 3
player_health = player_health_initial
last_hit_time = 0
INVULNERABILITY_TIME = 1000

# Movement/Shooting flags
move_left = move_right = move_up = move_down = False
is_shooting_left = is_shooting_right = is_shooting_up = is_shooting_down = False

# Enemy/Obstacle Setup
ENEMY_TYPES = {
    'normal': {'size': 40, 'speed': 2,   'health': 2, 'color': GREEN,  'can_shoot': False, 'fire_rate': 0},
    'fast':   {'size': 30, 'speed': 3.5, 'health': 1, 'color': BLUE,   'can_shoot': False, 'fire_rate': 0},
    'shooter':{'size': 50, 'speed': 1,   'health': 3, 'color': PURPLE,  'can_shoot': False,  'fire_rate': 2000}
}
OBSTACLE_TYPES = {
    'destructible':   {'size': 30, 'health': 2,  'color': DARK_GRAY},
    'indestructible': {'size': 70, 'health': -1, 'color': GRAY}
}
MAX_ENEMIES =2
ROOM_ENEMIES_TOTAL = 2
MAX_OBSTACLES = 5
MIN_OBSTACLES = 2
enemies = []
obstacles = []

# Enemy type encoding for state
TYPE_MAP = {'normal': 0.0, 'fast': 0.5, 'shooter': 1.0}

# Score Setup
score_initial = 2000
score = score_initial
SCORE_DECAY_INTERVAL = 100
SCORE_DECAY_AMOUNT = 20
last_score_decay_time = 0

# Stagnation
STAGNATION_THRESHOLD = 5.0

BULLET_SIZE, BULLET_SPEED, SHOOT_COOLDOWN = 10, 10, 200
bullets = []
last_shot_time = 0
ENEMY_BULLET_SIZE, ENEMY_BULLET_SPEED = 12, 7
enemy_bullets = []

# --- Action Space ---
# 5 move options x 5 shoot options = 25 combined actions
# Move: 0=none, 1=up, 2=down, 3=left, 4=right
# Shoot: 0=none, 1=up, 2=down, 3=left, 4=right
MOVE_OPTIONS = 5
SHOOT_OPTIONS = 5
ACTION_DIM = MOVE_OPTIONS * SHOOT_OPTIONS  # 25

# --- DQN Neural Network ---
class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.fc(x)

# --- Hyperparameters ---
# State: px, py, ph + (x, y, type) per enemy + (x, y) per obstacle + ecount + bx, by, bdx, bdy
INPUT_DIM = 3 + (3 * MAX_ENEMIES) + (2 * MAX_OBSTACLES) + 1 + 4
ALPHA = 0.0001
GAMMA = 0.95
MEMORY_SIZE, BATCH_SIZE = 200000, 64
EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.05, 0.995
TARGET_UPDATE = 20

policy_net = DQNNet(INPUT_DIM, ACTION_DIM)
target_net = DQNNet(INPUT_DIM, ACTION_DIM)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)
memory = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_START


def get_state():
    px = player_x / SCREEN_WIDTH
    py = player_y / SCREEN_HEIGHT
    ph = player_health / player_health_initial

    # Sort enemies by distance to player so index 0 is always closest
    sorted_enemies = sorted(
        enemies,
        key=lambda e: math.sqrt((e['x'] - player_x)**2 + (e['y'] - player_y)**2)
    )

    # Enemy coords: x, y, type (3 values per enemy)
    enemy_coords = []
    for i in range(MAX_ENEMIES):
        if i < len(sorted_enemies):
            e = sorted_enemies[i]
            enemy_coords.extend([
                e['x'] / SCREEN_WIDTH,
                e['y'] / SCREEN_HEIGHT,
                TYPE_MAP[e['type']]
            ])
        else:
            enemy_coords.extend([-1.0, -1.0, -1.0])

    ecount = len(enemies) / ROOM_ENEMIES_TOTAL

    # Obstacle coords
    obs_coords = []
    for i in range(MAX_OBSTACLES):
        if i < len(obstacles):
            obs_coords.extend([
                obstacles[i]['rect'].x / SCREEN_WIDTH,
                obstacles[i]['rect'].y / SCREEN_HEIGHT
            ])
        else:
            obs_coords.extend([-1.0, -1.0])

    # Nearest enemy bullet: position + direction
    if enemy_bullets:
        closest = min(
            enemy_bullets,
            key=lambda b: math.sqrt((b['rect'].x - player_x)**2 + (b['rect'].y - player_y)**2)
        )
        bx  = closest['rect'].x / SCREEN_WIDTH
        by  = closest['rect'].y / SCREEN_HEIGHT
        bdx = closest['direction'][0]
        bdy = closest['direction'][1]
    else:
        bx, by, bdx, bdy = -1.0, -1.0, 0.0, 0.0

    return np.array(
        [px, py, ph] + enemy_coords + obs_coords + [ecount, bx, by, bdx, bdy],
        dtype=np.float32
    )


def choose_action(state):
    global epsilon
    if mode == 'train' and random.random() < epsilon:
        return random.randrange(ACTION_DIM)
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        return policy_net(state_t).argmax().item()


def decode_action(a_idx):
    """Decode combined action index into (move_idx, shoot_idx)."""
    move_idx  = a_idx // SHOOT_OPTIONS
    shoot_idx = a_idx % SHOOT_OPTIONS
    return move_idx, shoot_idx


def apply_action(a_idx):
    global move_left, move_right, move_up, move_down
    global is_shooting_left, is_shooting_right, is_shooting_up, is_shooting_down

    move_left = move_right = move_up = move_down = False
    is_shooting_left = is_shooting_right = is_shooting_up = is_shooting_down = False

    move_idx, shoot_idx = decode_action(a_idx)

    # Apply movement
    if   move_idx == 1: move_up    = True
    elif move_idx == 2: move_down  = True
    elif move_idx == 3: move_left  = True
    elif move_idx == 4: move_right = True
    # move_idx == 0 means no movement

    # Apply shooting
    if   shoot_idx == 1: is_shooting_up    = True
    elif shoot_idx == 2: is_shooting_down  = True
    elif shoot_idx == 3: is_shooting_left  = True
    elif shoot_idx == 4: is_shooting_right = True
    # shoot_idx == 0 means no shooting


def train_step():
    if len(memory) < BATCH_SIZE: return
    batch = random.sample(memory, BATCH_SIZE)
    s, a, r, ns, d = zip(*batch)

    s  = torch.FloatTensor(np.array(s))
    a  = torch.LongTensor(a)
    r  = torch.FloatTensor(r)
    ns = torch.FloatTensor(np.array(ns))
    d  = torch.FloatTensor(d)

    q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Standard DQN: target_net picks max Q of next state
        next_q_values = target_net(ns).max(1)[0]
        target_q_values = r + (1 - d) * GAMMA * next_q_values

    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def spawn_obstacles():
    global obstacles
    obstacles.clear()
    for _ in range(random.randint(MIN_OBSTACLES, MAX_OBSTACLES)):
        t = random.choice(list(OBSTACLE_TYPES.keys()))
        size = OBSTACLE_TYPES[t]['size']
        while True:
            new_rect = pygame.Rect(
                random.randint(0, SCREEN_WIDTH - size),
                random.randint(0, SCREEN_HEIGHT - size),
                size, size
            )
            if not new_rect.colliderect(player_rect) and \
               not any(new_rect.colliderect(o['rect']) for o in obstacles):
                obstacles.append({'rect': new_rect, 'health': OBSTACLE_TYPES[t]['health'], 'type': t})
                break


def spawn_enemy(etype):
    data = ENEMY_TYPES[etype]
    size = data['size']
    while True:
        sx = float(random.randint(0, SCREEN_WIDTH - size))
        sy = float(random.randint(0, SCREEN_HEIGHT - size))
        new_rect = pygame.Rect(int(sx), int(sy), size, size)
        if not new_rect.colliderect(player_rect) and \
           not any(new_rect.colliderect(o['rect']) for o in obstacles) and \
           not any(new_rect.colliderect(e['rect']) for e in enemies):
            enemies.append({
                'rect': new_rect,
                'x': sx, 'y': sy,
                'health': data['health'],
                'type': etype,
                'last_shot_time': 0
            })
            break


def reset_game():
    global player_x, player_y, player_health, score
    global bullets, enemy_bullets, enemies, game_state
    global last_hit_time, last_score_decay_time

    player_x, player_y = player_x_initial, player_y_initial
    player_health = player_health_initial
    score = score_initial
    last_hit_time = 0
    last_score_decay_time = pygame.time.get_ticks()
    bullets.clear()
    enemy_bullets.clear()
    enemies.clear()
    game_state = GAME_STATE_PLAYING
    spawn_obstacles()
    
    spawn_enemy('normal')
    spawn_enemy('shooter')


def handle_input():
    global running, game_state
    global move_left, move_right, move_up, move_down
    global is_shooting_left, is_shooting_right, is_shooting_up, is_shooting_down
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r: reset_game()
            if event.key == pygame.K_p:
                game_state = GAME_STATE_PAUSED if game_state == GAME_STATE_PLAYING else GAME_STATE_PLAYING
            if game_state == GAME_STATE_PLAYING:
                if event.key == pygame.K_a: move_left  = True
                if event.key == pygame.K_d: move_right = True
                if event.key == pygame.K_w: move_up    = True
                if event.key == pygame.K_s: move_down  = True
                if event.key == pygame.K_LEFT:  is_shooting_left  = True
                if event.key == pygame.K_RIGHT: is_shooting_right = True
                if event.key == pygame.K_UP:    is_shooting_up    = True
                if event.key == pygame.K_DOWN:  is_shooting_down  = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a: move_left  = False
            if event.key == pygame.K_d: move_right = False
            if event.key == pygame.K_w: move_up    = False
            if event.key == pygame.K_s: move_down  = False
            if event.key == pygame.K_LEFT:  is_shooting_left  = False
            if event.key == pygame.K_RIGHT: is_shooting_right = False
            if event.key == pygame.K_UP:    is_shooting_up    = False
            if event.key == pygame.K_DOWN:  is_shooting_down  = False


def update_game_logic():
    global player_x, player_y, player_health, score
    global last_shot_time, last_hit_time, game_state
    global obstacles, last_score_decay_time

    t = pygame.time.get_ticks()

    # Score decay
    if t - last_score_decay_time > SCORE_DECAY_INTERVAL:
        score -= SCORE_DECAY_AMOUNT
        last_score_decay_time = t

    # Player shooting
    if t - last_shot_time > SHOOT_COOLDOWN:
        d = None
        if is_shooting_left:  d = (-1, 0)
        elif is_shooting_right: d = (1, 0)
        elif is_shooting_up:    d = (0, -1)
        elif is_shooting_down:  d = (0, 1)
        if d:
            bullets.append({'rect': pygame.Rect(player_x+20, player_y+20, 10, 10), 'direction': d})
            last_shot_time = t

    # Player movement
    if move_left:  player_x -= PLAYER_SPEED
    if move_right: player_x += PLAYER_SPEED
    if move_up:    player_y -= PLAYER_SPEED
    if move_down:  player_y += PLAYER_SPEED
    player_x = max(0, min(player_x, SCREEN_WIDTH - PLAYER_SIZE))
    player_y = max(0, min(player_y, SCREEN_HEIGHT - PLAYER_SIZE))
    player_rect.topleft = (int(player_x), int(player_y))

    # Player-obstacle collision
    for obs in obstacles:
        if player_rect.colliderect(obs['rect']):
            overlap = player_rect.clip(obs['rect'])
            if overlap.width < overlap.height:
                if player_rect.centerx < obs['rect'].centerx: player_x -= overlap.width
                else: player_x += overlap.width
            else:
                if player_rect.centery < obs['rect'].centery: player_y -= overlap.height
                else: player_y += overlap.height
            player_rect.topleft = (int(player_x), int(player_y))

    # Enemy update
    for i, e in enumerate(enemies):
        dx = player_x - e['x']
        dy = player_y - e['y']
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 0:
            e['x'] += (dx / dist) * ENEMY_TYPES[e['type']]['speed']
            e['y'] += (dy / dist) * ENEMY_TYPES[e['type']]['speed']
        e['rect'].topleft = (int(e['x']), int(e['y']))

        # Enemy-obstacle collision
        for obs in obstacles:
            if e['rect'].colliderect(obs['rect']):
                overlap = e['rect'].clip(obs['rect'])
                if overlap.width < overlap.height:
                    if e['rect'].centerx < obs['rect'].centerx: e['x'] -= overlap.width
                    else: e['x'] += overlap.width
                else:
                    if e['rect'].centery < obs['rect'].centery: e['y'] -= overlap.height
                    else: e['y'] += overlap.height
                e['rect'].topleft = (int(e['x']), int(e['y']))

        # Enemy-enemy separation
        for j, e2 in enumerate(enemies):
            if i != j and e['rect'].colliderect(e2['rect']):
                overlap = e['rect'].clip(e2['rect'])
                if overlap.width < overlap.height:
                    if e['rect'].centerx < e2['rect'].centerx:
                        e['x'] -= overlap.width/2; e2['x'] += overlap.width/2
                    else:
                        e['x'] += overlap.width/2; e2['x'] -= overlap.width/2
                else:
                    if e['rect'].centery < e2['rect'].centery:
                        e['y'] -= overlap.height/2; e2['y'] += overlap.height/2
                    else:
                        e['y'] += overlap.height/2; e2['y'] -= overlap.height/2
                e['rect'].topleft  = (int(e['x']),  int(e['y']))
                e2['rect'].topleft = (int(e2['x']), int(e2['y']))

        # Enemy shooting
        if ENEMY_TYPES[e['type']]['can_shoot'] and \
           t - e['last_shot_time'] > ENEMY_TYPES[e['type']]['fire_rate']:
            if dist > 0:
                enemy_bullets.append({
                    'rect': pygame.Rect(e['x']+20, e['y']+20, 12, 12),
                    'direction': (dx/dist, dy/dist)
                })
                e['last_shot_time'] = t

        # Enemy-player contact damage
        if e['rect'].colliderect(player_rect) and t - last_hit_time > INVULNERABILITY_TIME:
            player_health -= 1
            score -= 500
            last_hit_time = t

    # Player bullets
    for b in bullets[:]:
        b['rect'].x += b['direction'][0] * BULLET_SPEED
        b['rect'].y += b['direction'][1] * BULLET_SPEED

        # Check out of bounds
        if not (0 <= b['rect'].x <= SCREEN_WIDTH and 0 <= b['rect'].y <= SCREEN_HEIGHT):
            if b in bullets: bullets.remove(b)
            continue

        hit_enemy = False
        for e in enemies[:]:
            if b['rect'].colliderect(e['rect']):
                # FIX: properly decrement health, only remove when health reaches 0
                e['health'] -= 1
                if b in bullets: bullets.remove(b)
                if e['health'] <= 0:
                    if e in enemies: enemies.remove(e)
                    score += 200
                hit_enemy = True
                break

        if not hit_enemy and b in bullets:
            for obs in obstacles[:]:
                if b['rect'].colliderect(obs['rect']):
                    if b in bullets: bullets.remove(b)
                    if obs['type'] == 'destructible':
                        obs['health'] -= 1
                        if obs['health'] <= 0:
                            obstacles.remove(obs)
                            score += 50
                    break

    # Enemy bullets
    for eb in enemy_bullets[:]:
        eb['rect'].x += eb['direction'][0] * ENEMY_BULLET_SPEED
        eb['rect'].y += eb['direction'][1] * ENEMY_BULLET_SPEED

        # Check out of bounds
        if not (0 <= eb['rect'].x <= SCREEN_WIDTH and 0 <= eb['rect'].y <= SCREEN_HEIGHT):
            if eb in enemy_bullets: enemy_bullets.remove(eb)
            continue

        if eb['rect'].colliderect(player_rect) and t - last_hit_time > INVULNERABILITY_TIME:
            player_health -= 1
            score -= 500
            last_hit_time = t
            if eb in enemy_bullets: enemy_bullets.remove(eb)
        elif any(eb['rect'].colliderect(obs['rect']) for obs in obstacles):
            if eb in enemy_bullets: enemy_bullets.remove(eb)

    # Terminal conditions
    if player_health <= 0 or score <= 0:
        game_state = GAME_STATE_GAME_OVER
    elif not enemies:
        score += 4000
        game_state = GAME_STATE_ROOM_CLEARED


def draw_game_elements(screen, font, large_font):
    screen.fill(BLACK)
    if game_state in (GAME_STATE_PLAYING, GAME_STATE_PAUSED):
        for obs in obstacles:
            pygame.draw.rect(screen, OBSTACLE_TYPES[obs['type']]['color'], obs['rect'])

        # Player
        if 'player' in IMAGES:
            screen.blit(IMAGES['player'], player_rect)
            if pygame.time.get_ticks() - last_hit_time < INVULNERABILITY_TIME and \
               (pygame.time.get_ticks() // 100) % 2 == 0:
                s = pygame.Surface((50, 50), pygame.SRCALPHA)
                s.fill((255, 0, 0, 128))
                screen.blit(s, player_rect)
        else:
            color = RED if (pygame.time.get_ticks() - last_hit_time > 100 or
                           (pygame.time.get_ticks()//100)%2==0) else WHITE
            pygame.draw.rect(screen, color, player_rect)

        # Enemies — show health bar above each
        for e in enemies:
            etype = e['type']
            if etype in IMAGES:
                screen.blit(IMAGES[etype], e['rect'])
            else:
                pygame.draw.rect(screen, ENEMY_TYPES[etype]['color'], e['rect'])
            # Health bar
            max_hp = ENEMY_TYPES[etype]['health']
            bar_w = e['rect'].width
            hp_w = int(bar_w * e['health'] / max_hp)
            pygame.draw.rect(screen, RED,   (e['rect'].x, e['rect'].y - 6, bar_w, 4))
            pygame.draw.rect(screen, GREEN, (e['rect'].x, e['rect'].y - 6, hp_w,  4))

        # Player bullets
        for b in bullets:
            if 'bullet' in IMAGES:
                screen.blit(IMAGES['bullet'], b['rect'])
            else:
                pygame.draw.rect(screen, YELLOW, b['rect'])

        # Enemy bullets
        for eb in enemy_bullets:
            if 'enemy_bullet' in IMAGES:
                screen.blit(IMAGES['enemy_bullet'], eb['rect'])
            else:
                pygame.draw.rect(screen, ORANGE, eb['rect'])

    txt = font.render(
        f"Score: {score}  HP: {player_health}  Enemies: {len(enemies)}  Epsilon: {epsilon:.3f}",
        True, WHITE
    )
    screen.blit(txt, (10, 10))

    if game_state == GAME_STATE_GAME_OVER:
        msg = large_font.render("GAME OVER", True, RED)
        screen.blit(msg, (SCREEN_WIDTH//2 - 140, SCREEN_HEIGHT//2 - 50))
        sub = font.render("Press R to Restart", True, WHITE)
        screen.blit(sub, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 + 30))
    elif game_state == GAME_STATE_ROOM_CLEARED:
        msg = large_font.render("VICTORY!", True, GREEN)
        screen.blit(msg, (SCREEN_WIDTH//2 - 120, SCREEN_HEIGHT//2 - 50))
        sub = font.render("Press R to Reset", True, WHITE)
        screen.blit(sub, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 + 30))
    elif game_state == GAME_STATE_PAUSED:
        msg = large_font.render("PAUSED", True, WHITE)
        screen.blit(msg, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 50))

    pygame.display.flip()


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("DQN v2 - Combat Agent")
    font    = pygame.font.Font(None, 36)
    l_font  = pygame.font.Font(None, 74)
    clock   = pygame.time.Clock()

    mode         = sys.argv[1] if len(sys.argv) > 1 else 'manual'
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else (1000 if mode == 'train' else 10)

    if mode == 'inference':
        try:
            policy_net.load_state_dict(torch.load('dqn_v2_mixed2e_model.pth'))
            policy_net.eval()
            epsilon = 0
            print("Loaded weights from dqn_v2_model.pth for inference.")
        except Exception as e:
            print(f"Failed to load weights: {e}")

    current_episode     = 0
    win_streak          = 0
    total_wins          = 0
    target_update_counter = 0
    best_win_rate       = 0.0

    reset_game()
    running = True

    while running:
        handle_input()

        if game_state == GAME_STATE_PLAYING:
            if mode != 'manual':
                s = get_state()
                a = choose_action(s)

                prev_h = player_health
                prev_s = score
                old_player_x, old_player_y = player_x, player_y

                apply_action(a)
                update_game_logic()

                ns = get_state()

                if mode == 'train':
                    raw_diff   = score - prev_s
                    reward     = (max(0, raw_diff + SCORE_DECAY_AMOUNT)) / 10.0
                    dist_moved = math.sqrt((player_x - old_player_x)**2 + (player_y - old_player_y)**2)
                    is_terminal = (game_state == GAME_STATE_GAME_OVER or
                                   game_state == GAME_STATE_ROOM_CLEARED)

                    # Damage penalty
                    if player_health < prev_h:
                        reward -= 30

                    # Terminal rewards
                    if game_state == GAME_STATE_ROOM_CLEARED:
                        reward += 100
                    if game_state == GAME_STATE_GAME_OVER:
                        reward -= 100

                    # Movement pressure when enemies are close
                    if enemies:
                        nearest_dist = min(
                            math.sqrt((player_x - e['x'])**2 + (player_y - e['y'])**2)
                            for e in enemies
                        )
                        if nearest_dist < 200 and dist_moved < STAGNATION_THRESHOLD:
                            reward -= 2.0  # must move when threatened
                        if nearest_dist > 400:
                            reward -= 1.0  # penalise hiding far away

                    memory.append((s, a, reward, ns, is_terminal))
                    train_step()
                    target_update_counter += 1

                    if target_update_counter % TARGET_UPDATE == 0:
                        target_net.load_state_dict(policy_net.state_dict())

            else:
                update_game_logic()

        # Only render during manual and inference
        if mode != 'train':
            draw_game_elements(screen, font, l_font)

        # Episode handling
        if game_state == GAME_STATE_ROOM_CLEARED:
            win_streak += 1
            total_wins += 1
            current_episode += 1

            if mode == 'train':
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                # Save best weights
                if current_episode >= 50:
                    wr = total_wins / current_episode
                    if wr > best_win_rate:
                        best_win_rate = wr
                        torch.save(policy_net.state_dict(), 'dqn_v2_mixed2e_best.pth')
                print(f'WIN  | Episode: {current_episode}/{max_episodes} | '
                      f'Win Streak: {win_streak} | '
                      f'Total Wins: {total_wins} | '
                      f'Win Rate: {total_wins/current_episode*100:.1f}% | '
                      f'Epsilon: {epsilon:.3f}')
            else:
                print(f'WIN  | Episode: {current_episode}/{max_episodes} | '
                      f'Win Streak: {win_streak} | '
                      f'Total Wins: {total_wins} | '
                      f'Win Rate: {total_wins/current_episode*100:.1f}%')

            if current_episode < max_episodes:
                reset_game()
            else:
                running = False

        elif game_state == GAME_STATE_GAME_OVER:
            win_streak = 0
            current_episode += 1

            if mode != 'manual':
                if mode == 'train':
                    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                    if current_episode >= 50:
                        wr = total_wins / current_episode
                        if wr > best_win_rate:
                            best_win_rate = wr
                            torch.save(policy_net.state_dict(), 'dqn_v2_mixed2e_best.pth')
                print(f'LOSS | Episode: {current_episode}/{max_episodes} | '
                      f'Total Wins: {total_wins} | '
                      f'Win Rate: {total_wins/current_episode*100:.1f}%' +
                      (f' | Epsilon: {epsilon:.3f}' if mode == 'train' else ''))

                if current_episode < max_episodes:
                    reset_game()
                else:
                    running = False

        clock.tick(FPS)

    if mode == 'train':
        torch.save(policy_net.state_dict(), 'dqn_v2_mixed2e_model.pth')
        print(f'\nTraining complete. Final model saved to dqn_v2_mixed2e_model.pth')
        print(f'Best model saved to dqn_v2_mixed2e_best.pth (best win rate: {best_win_rate*100:.1f}%)')

    pygame.quit()
