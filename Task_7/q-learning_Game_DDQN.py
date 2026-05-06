import pygame
import math
import random
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import seaborn as sns
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
    'normal': {'size': 40, 'speed': 2, 'health': 2, 'color': GREEN, 'can_shoot': False, 'fire_rate': 0},
    'fast': {'size': 30, 'speed': 3.5, 'health': 1, 'color': BLUE, 'can_shoot': False, 'fire_rate': 0},
    'shooter': {'size': 50, 'speed': 1, 'health': 2, 'color': (150, 0, 150), 'can_shoot': True, 'fire_rate': 2000}
}
OBSTACLE_TYPES = {'destructible': {'size': 30, 'health': 2, 'color': DARK_GRAY}, 'indestructible': {'size': 70, 'health': -1, 'color': GRAY}}
MAX_ENEMIES = 10
ROOM_ENEMIES_TOTAL = 5
MAX_OBSTACLES = 5
MIN_OBSTACLES = 2
enemies = []
obstacles = []

# Score Setup
score_initial = 2000
score = score_initial
SCORE_DECAY_INTERVAL = 100
SCORE_DECAY_AMOUNT = 20
last_score_decay_time = 0

# RL Reward adjustments
STAGNATION_THRESHOLD = 5.0  # Minimum distance to be considered "moving" (in pixels)
STAGNATION_PENALTY = -0.5   # Penalty for general stagnation

BULLET_SIZE, BULLET_SPEED, SHOOT_COOLDOWN = 10, 10, 200
bullets = []
last_shot_time = 0
ENEMY_BULLET_SIZE, ENEMY_BULLET_SPEED = 12, 7
enemy_bullets = []

# --- DDQN Neural Network Architecture ---
class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, x): return self.fc(x)

# --- DDQN Hyperparameters and Initialization ---
INPUT_DIM, ACTION_DIM = 3 + (2 * MAX_ENEMIES) + (2 * MAX_OBSTACLES) + 1, 9
ALPHA,GAMMA = 0.001, 0.95
MEMORY_SIZE, BATCH_SIZE = 200000, 64
EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.05, 0.995
TARGET_UPDATE = 20
policy_net = DQNNet(INPUT_DIM, ACTION_DIM)
target_net = DQNNet(INPUT_DIM, ACTION_DIM)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)
memory = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_START
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), 'shoot_up', 'shoot_down', 'shoot_left', 'shoot_right', 'none']

def get_state():
    px = player_x / SCREEN_WIDTH
    py = player_y / SCREEN_HEIGHT
    ph = player_health / player_health_initial
    enemy_coords = []
    for i in range(MAX_ENEMIES):
        if i < len(enemies):
            enemy_coords.extend([enemies[i]['x'] / SCREEN_WIDTH, enemies[i]['y'] / SCREEN_HEIGHT])
        else:
            enemy_coords.extend([-1.0, -1.0])  # Padding for missing enemies
    ecount = len(enemies) / ROOM_ENEMIES_TOTAL

    obs_coords = []
    for i in range(MAX_OBSTACLES):
        if i < len(obstacles):
            obs_coords.extend([obstacles[i]['rect'].x / SCREEN_WIDTH, obstacles[i]['rect'].y / SCREEN_HEIGHT])
        else:
            obs_coords.extend([-1.0, -1.0])

    return np.array([px, py, ph]+enemy_coords+obs_coords+[ecount], dtype=np.float32)

def choose_action(state):
    global epsilon
    if mode == 'train' and random.random() < epsilon:
        return random.randrange(ACTION_DIM)
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        return policy_net(state_t).argmax().item()

def train_step():
    if len(memory) < BATCH_SIZE: return
    batch = random.sample(memory, BATCH_SIZE)
    s, a, r, ns, d = zip(*batch)

    s = torch.FloatTensor(np.array(s))
    a = torch.LongTensor(a)
    r = torch.FloatTensor(r)
    ns = torch.FloatTensor(np.array(ns))
    d = torch.FloatTensor(d)

    q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Double DQN logic: use policy_net to pick action, target_net to get Q value
        next_actions = policy_net(ns).argmax(1).unsqueeze(1)
        next_q_values = target_net(ns).gather(1, next_actions).squeeze(1)
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
            new_rect = pygame.Rect(random.randint(0, SCREEN_WIDTH-size), random.randint(0, SCREEN_HEIGHT-size), size, size)
            if not new_rect.colliderect(player_rect) and not any(new_rect.colliderect(o['rect']) for o in obstacles):
                obstacles.append({'rect': new_rect, 'health': OBSTACLE_TYPES[t]['health'], 'type': t})
                break

def reset_game():
    global player_x, player_y, player_health, score, bullets, enemy_bullets, enemies, game_state, last_hit_time, last_score_decay_time
    player_x, player_y, player_health, score = player_x_initial, player_y_initial, player_health_initial, score_initial
    last_hit_time, last_score_decay_time = 0, pygame.time.get_ticks()
    bullets.clear(); enemy_bullets.clear(); enemies.clear()
    game_state = GAME_STATE_PLAYING
    spawn_obstacles()
    for _ in range(ROOM_ENEMIES_TOTAL): spawn_enemy(random.choice(list(ENEMY_TYPES.keys())))

def spawn_enemy(etype):
    data = ENEMY_TYPES[etype]
    size = data['size']
    while True:
        sx, sy = float(random.randint(0, SCREEN_WIDTH-size)), float(random.randint(0, SCREEN_HEIGHT-size))
        new_rect = pygame.Rect(int(sx), int(sy), size, size)
        if not new_rect.colliderect(player_rect) and not any(new_rect.colliderect(o['rect']) for o in obstacles) and not any(new_rect.colliderect(e['rect']) for e in enemies):
            enemies.append({'rect': new_rect, 'x': sx, 'y': sy, 'health': data['health'], 'type': etype, 'last_shot_time': 0})
            break

def handle_input():
    global running, game_state, move_left, move_right, move_up, move_down, is_shooting_left, is_shooting_right, is_shooting_up, is_shooting_down
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r: reset_game()
            if event.key == pygame.K_p: game_state = GAME_STATE_PAUSED if game_state == GAME_STATE_PLAYING else GAME_STATE_PLAYING
            if game_state == GAME_STATE_PLAYING:
                if event.key == pygame.K_a: move_left = True
                if event.key == pygame.K_d: move_right = True
                if event.key == pygame.K_w: move_up = True
                if event.key == pygame.K_s: move_down = True
                if event.key == pygame.K_LEFT: is_shooting_left = True
                if event.key == pygame.K_RIGHT: is_shooting_right = True
                if event.key == pygame.K_UP: is_shooting_up = True
                if event.key == pygame.K_DOWN: is_shooting_down = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a: move_left = False
            if event.key == pygame.K_d: move_right = False
            if event.key == pygame.K_w: move_up = False
            if event.key == pygame.K_s: move_down = False
            if event.key == pygame.K_LEFT: is_shooting_left = False
            if event.key == pygame.K_RIGHT: is_shooting_right = False
            if event.key == pygame.K_UP: is_shooting_up = False
            if event.key == pygame.K_DOWN: is_shooting_down = False

def update_game_logic():
    global player_x, player_y, player_health, score, last_shot_time, last_hit_time, game_state, obstacles, last_score_decay_time
    t = pygame.time.get_ticks()
    if t - last_score_decay_time > SCORE_DECAY_INTERVAL: score -= SCORE_DECAY_AMOUNT; last_score_decay_time = t
    if t - last_shot_time > SHOOT_COOLDOWN:
        d = None
        if is_shooting_left: d = (-1, 0)
        elif is_shooting_right: d = (1, 0)
        elif is_shooting_up: d = (0, -1)
        elif is_shooting_down: d = (0, 1)
        if d:
            bullets.append({'rect': pygame.Rect(player_x+20, player_y+20, 10, 10), 'direction': d})
            last_shot_time = t
    if move_left: player_x -= PLAYER_SPEED
    if move_right: player_x += PLAYER_SPEED
    if move_up: player_y -= PLAYER_SPEED
    if move_down: player_y += PLAYER_SPEED
    player_x = max(0, min(player_x, SCREEN_WIDTH-50))
    player_y = max(0, min(player_y, SCREEN_HEIGHT-50))
    player_rect.topleft = (int(player_x), int(player_y))

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

    for i, e in enumerate(enemies):
        dx, dy = player_x - e['x'], player_y - e['y']
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 0:
            e['x'] += (dx/dist)*ENEMY_TYPES[e['type']]['speed']
            e['y'] += (dy/dist)*ENEMY_TYPES[e['type']]['speed']
        e['rect'].topleft = (int(e['x']), int(e['y']))

        # Enemy-Obstacle Collision Resolution
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

        # Enemy-Enemy Collision Resolution (Separation)
        for j, e2 in enumerate(enemies):
            if i != j and e['rect'].colliderect(e2['rect']):
                overlap = e['rect'].clip(e2['rect'])
                if overlap.width < overlap.height:
                    if e['rect'].centerx < e2['rect'].centerx: e['x'] -= overlap.width/2; e2['x'] += overlap.width/2
                    else: e['x'] += overlap.width/2; e2['x'] -= overlap.width/2
                else:
                    if e['rect'].centery < e2['rect'].centery: e['y'] -= overlap.height/2; e2['y'] += overlap.height/2
                    else: e['y'] += overlap.height/2; e2['y'] -= overlap.height/2
                e['rect'].topleft = (int(e['x']), int(e['y']))
                e2['rect'].topleft = (int(e2['x']), int(e2['y']))

        if ENEMY_TYPES[e['type']]['can_shoot'] and t - e['last_shot_time'] > ENEMY_TYPES[e['type']]['fire_rate']:
            if dist > 0: enemy_bullets.append({'rect': pygame.Rect(e['x']+20, e['y']+20, 12, 12), 'direction': (dx/dist, dy/dist)}); e['last_shot_time'] = t
        if e['rect'].colliderect(player_rect) and t - last_hit_time > INVULNERABILITY_TIME:
            player_health -= 1; score -= 500; last_hit_time = t

    for b in bullets[:]:
        b['rect'].x += b['direction'][0]*BULLET_SPEED; b['rect'].y += b['direction'][1]*BULLET_SPEED
        for e in enemies[:]:
            if b['rect'].colliderect(e['rect']):
                if e in enemies: enemies.remove(e)
                if b in bullets: bullets.remove(b)
                score += 200; break
        if b in bullets:
            for obs in obstacles[:]:
                if b['rect'].colliderect(obs['rect']):
                    bullets.remove(b)
                    if obs['type'] == 'destructible':
                        obs['health'] -= 1
                        if obs['health'] <= 0: obstacles.remove(obs); score += 50
                    break
    for eb in enemy_bullets[:]:
        eb['rect'].x += eb['direction'][0]*ENEMY_BULLET_SPEED; eb['rect'].y += eb['direction'][1]*ENEMY_BULLET_SPEED
        if eb['rect'].colliderect(player_rect) and t - last_hit_time > INVULNERABILITY_TIME:
            player_health -= 1; score -= 500; last_hit_time = t; enemy_bullets.remove(eb)
        elif any(eb['rect'].colliderect(obs['rect']) for obs in obstacles): enemy_bullets.remove(eb)
    if player_health <= 0 or score <= 0: game_state = GAME_STATE_GAME_OVER
    elif not enemies:
        score += 4000
        game_state = GAME_STATE_ROOM_CLEARED

def draw_game_elements(screen, font, large_font):
    screen.fill(BLACK)
    if game_state == GAME_STATE_PLAYING or game_state == GAME_STATE_PAUSED:
        for obs in obstacles: pygame.draw.rect(screen, OBSTACLE_TYPES[obs['type']]['color'], obs['rect'])
        # Draw Player
        if 'player' in IMAGES:
            screen.blit(IMAGES['player'], player_rect)
            if pygame.time.get_ticks() - last_hit_time < INVULNERABILITY_TIME and (pygame.time.get_ticks() // 100) % 2 == 0:
                s = pygame.Surface((50,50), pygame.SRCALPHA)
                s.fill((255,0,0,128))
                screen.blit(s, player_rect)
        else:
            pygame.draw.rect(screen, RED if (pygame.time.get_ticks() - last_hit_time > 100 or (pygame.time.get_ticks()//100)%2==0) else WHITE, player_rect)

        # Draw Enemies
        for e in enemies:
            etype = e['type']
            if etype in IMAGES:
                screen.blit(IMAGES[etype], e['rect'])
            else:
                pygame.draw.rect(screen, ENEMY_TYPES[etype]['color'], e['rect'])

        # Draw Bullets
        for b in bullets:
            if 'bullet' in IMAGES:
                screen.blit(IMAGES['bullet'], b['rect'])
            else:
                pygame.draw.rect(screen, YELLOW, b['rect'])

        # Draw Enemy Bullets
        for eb in enemy_bullets:
            if 'enemy_bullet' in IMAGES:
                screen.blit(IMAGES['enemy_bullet'], eb['rect'])
            else:
                pygame.draw.rect(screen, ORANGE, eb['rect'])

    txt = font.render(f"Score: {score}  HP: {player_health}  Epsilon: {epsilon:.2f}", True, WHITE)
    screen.blit(txt, (10, 10))
    if game_state == GAME_STATE_GAME_OVER:
        msg = large_font.render("GAME OVER", True, RED)
        screen.blit(msg, (SCREEN_WIDTH//2-140, SCREEN_HEIGHT//2-50))
        sub = font.render("Press R to Restart", True, WHITE)
        screen.blit(sub, (SCREEN_WIDTH//2-100, SCREEN_HEIGHT//2+30))
    elif game_state == GAME_STATE_ROOM_CLEARED:
        msg = large_font.render("VICTORY!", True, GREEN)
        screen.blit(msg, (SCREEN_WIDTH//2-120, SCREEN_HEIGHT//2-50))
        sub = font.render("Press R to Reset", True, WHITE)
        screen.blit(sub, (SCREEN_WIDTH//2-100, SCREEN_HEIGHT//2+30))
    elif game_state == GAME_STATE_PAUSED:
        msg = large_font.render("PAUSED", True, WHITE)
        screen.blit(msg, (SCREEN_WIDTH//2-100, SCREEN_HEIGHT//2-50))
    pygame.display.flip()

def apply_action(a_idx):
    global move_left, move_right, move_up, move_down, is_shooting_left, is_shooting_right, is_shooting_up, is_shooting_down
    move_left = move_right = move_up = move_down = False
    is_shooting_left = is_shooting_right = is_shooting_up = is_shooting_down = False
    a = ACTIONS[a_idx]
    if a == (0, -1): move_up = True
    elif a == (0, 1): move_down = True
    elif a == (-1, 0): move_left = True
    elif a == (1, 0): move_right = True
    elif a == 'shoot_up': is_shooting_up = True
    elif a == 'shoot_down': is_shooting_down = True
    elif a == 'shoot_left': is_shooting_left = True
    elif a == 'shoot_right': is_shooting_right = True

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font, l_font = pygame.font.Font(None, 36), pygame.font.Font(None, 74)
    clock = pygame.time.Clock()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'manual'
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else (1000 if mode == 'train' else 1)
    if mode == 'inference':
        try:
            policy_net.load_state_dict(torch.load('ddqn_model-2.pth'))
            policy_net.eval()
            epsilon = 0
            print("Loaded weights from ddqn_model.pth for inference.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            pass
    current_episode = 0
    win_streak = 0
    target_update_counter = 0 # Initialize counter for target network updates
    reset_game()
    running = True
    while running:
        handle_input()
        if game_state == GAME_STATE_PLAYING:
            if mode != 'manual':
                s = get_state()
                a = choose_action(s)

                prev_h, prev_s = player_health, score
                # Capture player's position before applying the action for stagnation check
                old_player_x, old_player_y = player_x, player_y

                apply_action(a)
                update_game_logic()

                ns = get_state()

                if mode == 'train':
                    raw_diff = score - prev_s
                    reward = (max(0, raw_diff + SCORE_DECAY_AMOUNT)) / 10.0

                    dist = math.sqrt((player_x - old_player_x)**2 + (player_y - old_player_y)**2)
                    is_terminal = (game_state == GAME_STATE_GAME_OVER or game_state == GAME_STATE_ROOM_CLEARED)
                    if a in [0, 1, 2, 3] and dist < STAGNATION_THRESHOLD: ## only when moving
                        reward -= 0.5
                    if game_state == GAME_STATE_ROOM_CLEARED: 
                        reward += 100
                    
                    memory.append((s,a,reward,ns,is_terminal))
                    train_step(); target_update_counter += 1
                    
                    if target_update_counter % TARGET_UPDATE == 0: target_net.load_state_dict(policy_net.state_dict())
            else:
                update_game_logic()
        draw_game_elements(screen, font, l_font)
        if game_state == GAME_STATE_ROOM_CLEARED:
            win_streak += 1
            print(f'Win Streak: {win_streak}')
            if win_streak >=5:
                if mode != 'manual':
                    running = False
            else:
                reset_game()
        elif game_state == GAME_STATE_GAME_OVER:
            if mode != 'manual':
                current_episode += 1
                win_streak = 0
                if current_episode < max_episodes:
                    if mode == 'train':
                        print(f'Episode: {current_episode}/{max_episodes}')
                        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                    reset_game()
                else: running = False
        clock.tick(FPS)
    if mode == 'train':
        torch.save(policy_net.state_dict(),"ddqn_model-2.pth")
    pygame.quit()