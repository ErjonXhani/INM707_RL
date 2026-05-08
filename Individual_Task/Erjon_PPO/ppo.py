import pygame
import math
import random
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 144

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
    'shooter':{'size': 50, 'speed': 1,   'health': 3, 'color': PURPLE,  'can_shoot': False, 'fire_rate': 2000}
}
OBSTACLE_TYPES = {
    'destructible':   {'size': 30, 'health': 2,  'color': DARK_GRAY},
    'indestructible': {'size': 70, 'health': -1, 'color': GRAY}
}
MAX_ENEMIES = 2
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
# Same as DQN: 5 move x 5 shoot = 25 combined actions
MOVE_OPTIONS = 5
SHOOT_OPTIONS = 5
ACTION_DIM = MOVE_OPTIONS * SHOOT_OPTIONS  # 25

# --- State Dimensions ---
INPUT_DIM = 3 + (3 * MAX_ENEMIES) + (2 * MAX_OBSTACLES) + 1 + 4

# --- PPO Hyperparameters ---
ALPHA        = 0.0003    # learning rate
GAMMA        = 0.95      # discount factor
GAE_LAMBDA   = 0.95      # GAE smoothing parameter
CLIP_EPS     = 0.2       # PPO clip epsilon
ENTROPY_COEF = 0.01      # entropy bonus to encourage exploration
VALUE_COEF   = 0.5       # value loss coefficient
MAX_GRAD_NORM = 0.5      # gradient clipping
PPO_EPOCHS   = 4         # how many times to reuse each batch
ROLLOUT_STEPS = 512      # steps to collect before each PPO update
BATCH_SIZE   = 64        # minibatch size for PPO updates


# --- Actor-Critic Network ---
# Shared backbone with separate actor and critic heads
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Actor head: outputs logits for each action
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Critic head: outputs a single state value estimate
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.shared(x)
        logits   = self.actor(features)
        value    = self.critic(features)
        return logits, value

    def get_action(self, state):
        """Sample an action from the policy and return action, log_prob, value."""
        logits, value = self.forward(state)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze()

    def evaluate(self, states, actions):
        """Evaluate actions for PPO update — returns log_probs, values, entropy."""
        logits, values = self.forward(states)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values.squeeze(), entropy


# Initialise network and optimizer
model     = ActorCritic(INPUT_DIM, ACTION_DIM)
optimizer = optim.Adam(model.parameters(), lr=ALPHA)


# --- Rollout Buffer ---
# PPO is on-policy: collect ROLLOUT_STEPS steps, update, discard, repeat
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.values    = []
        self.log_probs = []
        self.dones     = []

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)


rollout = RolloutBuffer()


def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Generalised Advantage Estimation (GAE).
    Computes advantages and returns for the collected rollout.
    GAE balances bias vs variance in advantage estimates using lambda.
    """
    advantages = []
    gae = 0
    values = values + [last_value]

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae   = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


def ppo_update(last_value):
    """Run PPO_EPOCHS of minibatch updates on the collected rollout."""
    advantages, returns = compute_gae(
        rollout.rewards,
        [v.item() if torch.is_tensor(v) else v for v in rollout.values],
        rollout.dones,
        last_value
    )

    # Convert everything to tensors
    states     = torch.FloatTensor(np.array(rollout.states))
    actions    = torch.LongTensor(rollout.actions)
    returns    = torch.FloatTensor(returns)
    advantages = torch.FloatTensor(advantages)
    old_log_probs = torch.stack(rollout.log_probs).detach()

    # Normalise advantages for training stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(rollout.states)

    for _ in range(PPO_EPOCHS):
        # Shuffle indices for minibatch sampling
        indices = np.random.permutation(n)

        for start in range(0, n, BATCH_SIZE):
            end  = start + BATCH_SIZE
            mb   = indices[start:end]

            mb_states     = states[mb]
            mb_actions    = actions[mb]
            mb_returns    = returns[mb]
            mb_advantages = advantages[mb]
            mb_old_lp     = old_log_probs[mb]

            # Evaluate current policy on minibatch
            new_log_probs, values, entropy = model.evaluate(mb_states, mb_actions)

            # PPO clipped surrogate objective
            ratio      = torch.exp(new_log_probs - mb_old_lp)
            surr1      = ratio * mb_advantages
            surr2      = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            critic_loss = nn.MSELoss()(values, mb_returns)

            # Entropy bonus encourages exploration
            entropy_loss = -entropy.mean()

            # Combined PPO loss
            loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    rollout.clear()


def get_state():
    px = player_x / SCREEN_WIDTH
    py = player_y / SCREEN_HEIGHT
    ph = player_health / player_health_initial

    sorted_enemies = sorted(
        enemies,
        key=lambda e: math.sqrt((e['x'] - player_x)**2 + (e['y'] - player_y)**2)
    )

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

    obs_coords = []
    for i in range(MAX_OBSTACLES):
        if i < len(obstacles):
            obs_coords.extend([
                obstacles[i]['rect'].x / SCREEN_WIDTH,
                obstacles[i]['rect'].y / SCREEN_HEIGHT
            ])
        else:
            obs_coords.extend([-1.0, -1.0])

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
    """
    During training: sample stochastically from policy distribution.
    During inference: take the greedy argmax action.
    """
    state_t = torch.FloatTensor(state).unsqueeze(0)
    if mode == 'train':
        action, log_prob, value = model.get_action(state_t)
        return action, log_prob, value
    else:
        with torch.no_grad():
            logits, value = model(state_t)
            # Sample with low temperature instead of pure argmax
            # This preserves the policy distribution while being mostly greedy
            dist = Categorical(logits=logits)
            action = dist.sample().item()
        return action, None, None


def decode_action(a_idx):
    move_idx  = a_idx // SHOOT_OPTIONS
    shoot_idx = a_idx % SHOOT_OPTIONS
    return move_idx, shoot_idx


def apply_action(a_idx):
    global move_left, move_right, move_up, move_down
    global is_shooting_left, is_shooting_right, is_shooting_up, is_shooting_down

    move_left = move_right = move_up = move_down = False
    is_shooting_left = is_shooting_right = is_shooting_up = is_shooting_down = False

    move_idx, shoot_idx = decode_action(a_idx)

    if   move_idx == 1: move_up    = True
    elif move_idx == 2: move_down  = True
    elif move_idx == 3: move_left  = True
    elif move_idx == 4: move_right = True

    if   shoot_idx == 1: is_shooting_up    = True
    elif shoot_idx == 2: is_shooting_down  = True
    elif shoot_idx == 3: is_shooting_left  = True
    elif shoot_idx == 4: is_shooting_right = True


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

    if t - last_score_decay_time > SCORE_DECAY_INTERVAL:
        score -= SCORE_DECAY_AMOUNT
        last_score_decay_time = t

    if t - last_shot_time > SHOOT_COOLDOWN:
        d = None
        if is_shooting_left:    d = (-1, 0)
        elif is_shooting_right: d = (1, 0)
        elif is_shooting_up:    d = (0, -1)
        elif is_shooting_down:  d = (0, 1)
        if d:
            bullets.append({'rect': pygame.Rect(player_x+20, player_y+20, 10, 10), 'direction': d})
            last_shot_time = t

    if move_left:  player_x -= PLAYER_SPEED
    if move_right: player_x += PLAYER_SPEED
    if move_up:    player_y -= PLAYER_SPEED
    if move_down:  player_y += PLAYER_SPEED
    player_x = max(0, min(player_x, SCREEN_WIDTH - PLAYER_SIZE))
    player_y = max(0, min(player_y, SCREEN_HEIGHT - PLAYER_SIZE))
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
        dx = player_x - e['x']
        dy = player_y - e['y']
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 0:
            e['x'] += (dx / dist) * ENEMY_TYPES[e['type']]['speed']
            e['y'] += (dy / dist) * ENEMY_TYPES[e['type']]['speed']
        e['rect'].topleft = (int(e['x']), int(e['y']))

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

        if ENEMY_TYPES[e['type']]['can_shoot'] and \
           t - e['last_shot_time'] > ENEMY_TYPES[e['type']]['fire_rate']:
            if dist > 0:
                enemy_bullets.append({
                    'rect': pygame.Rect(e['x']+20, e['y']+20, 12, 12),
                    'direction': (dx/dist, dy/dist)
                })
                e['last_shot_time'] = t

        if e['rect'].colliderect(player_rect) and t - last_hit_time > INVULNERABILITY_TIME:
            player_health -= 1
            score -= 500
            last_hit_time = t

    for b in bullets[:]:
        b['rect'].x += b['direction'][0] * BULLET_SPEED
        b['rect'].y += b['direction'][1] * BULLET_SPEED

        if not (0 <= b['rect'].x <= SCREEN_WIDTH and 0 <= b['rect'].y <= SCREEN_HEIGHT):
            if b in bullets: bullets.remove(b)
            continue

        hit_enemy = False
        for e in enemies[:]:
            if b['rect'].colliderect(e['rect']):
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

    for eb in enemy_bullets[:]:
        eb['rect'].x += eb['direction'][0] * ENEMY_BULLET_SPEED
        eb['rect'].y += eb['direction'][1] * ENEMY_BULLET_SPEED

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

        for e in enemies:
            etype = e['type']
            if etype in IMAGES:
                screen.blit(IMAGES[etype], e['rect'])
            else:
                pygame.draw.rect(screen, ENEMY_TYPES[etype]['color'], e['rect'])
            max_hp = ENEMY_TYPES[etype]['health']
            bar_w  = e['rect'].width
            hp_w   = int(bar_w * e['health'] / max_hp)
            pygame.draw.rect(screen, RED,   (e['rect'].x, e['rect'].y - 6, bar_w, 4))
            pygame.draw.rect(screen, GREEN, (e['rect'].x, e['rect'].y - 6, hp_w,  4))

        for b in bullets:
            if 'bullet' in IMAGES:
                screen.blit(IMAGES['bullet'], b['rect'])
            else:
                pygame.draw.rect(screen, YELLOW, b['rect'])

        for eb in enemy_bullets:
            if 'enemy_bullet' in IMAGES:
                screen.blit(IMAGES['enemy_bullet'], eb['rect'])
            else:
                pygame.draw.rect(screen, ORANGE, eb['rect'])

    txt = font.render(
        f"Score: {score}  HP: {player_health}  Enemies: {len(enemies)}  Steps: {total_steps}",
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
    pygame.display.set_caption("PPO - Combat Agent")
    font   = pygame.font.Font(None, 36)
    l_font = pygame.font.Font(None, 74)
    clock  = pygame.time.Clock()

    mode         = sys.argv[1] if len(sys.argv) > 1 else 'manual'
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else (1000 if mode == 'train' else 10)

    if mode == 'inference':
        try:
            model.load_state_dict(torch.load('ppo_model.pth'))
            model.eval()
            print("Loaded weights from ppo_model.pth for inference.")
        except Exception as e:
            print(f"Failed to load weights: {e}")

    current_episode = 0
    win_streak      = 0
    total_wins      = 0
    total_steps     = 0
    best_win_rate   = 0.0

    reset_game()
    running = True

    while running:
        handle_input()

        if game_state == GAME_STATE_PLAYING:
            if mode != 'manual':
                s = get_state()

                if mode == 'train':
                    action, log_prob, value = choose_action(s)
                else:
                    action, _, _ = choose_action(s)

                prev_h = player_health
                prev_s = score
                old_player_x, old_player_y = player_x, player_y

                apply_action(action)
                update_game_logic()

                ns = get_state()

                if mode == 'train':
                    raw_diff   = score - prev_s
                    reward     = (max(0, raw_diff + SCORE_DECAY_AMOUNT)) / 10.0
                    dist_moved = math.sqrt((player_x - old_player_x)**2 + (player_y - old_player_y)**2)
                    is_terminal = (game_state == GAME_STATE_GAME_OVER or
                                   game_state == GAME_STATE_ROOM_CLEARED)

                    if player_health < prev_h:
                        reward -= 30

                    if game_state == GAME_STATE_ROOM_CLEARED:
                        reward += 100
                    if game_state == GAME_STATE_GAME_OVER:
                        reward -= 100

                    if enemies:
                        nearest_dist = min(
                            math.sqrt((player_x - e['x'])**2 + (player_y - e['y'])**2)
                            for e in enemies
                        )
                        if nearest_dist < 200 and dist_moved < STAGNATION_THRESHOLD:
                            reward -= 2.0
                        if nearest_dist > 400:
                            reward -= 1.0

                    # Penalise shooting in a direction with no enemy
                    move_idx, shoot_idx = decode_action(action)
                    if shoot_idx != 0 and enemies:
                        shoot_dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
                        if shoot_idx in shoot_dirs:
                            sdx, sdy = shoot_dirs[shoot_idx]
                            shooting_at_enemy = False
                            for e in enemies:
                                ex = e['x'] - player_x
                                ey = e['y'] - player_y
                                dist = math.sqrt(ex**2 + ey**2)
                                if dist > 0:
                                    ex, ey = ex/dist, ey/dist
                                    alignment = ex*sdx + ey*sdy
                                    if alignment > 0.5:
                                        shooting_at_enemy = True
                                        break
                            if not shooting_at_enemy:
                                reward -= 1.0

                    rollout.add(s, action, reward, value, log_prob, float(is_terminal))

                    total_steps += 1

                    # When rollout buffer is full, run PPO update
                    if len(rollout) >= ROLLOUT_STEPS:
                        # Bootstrap last value for GAE
                        with torch.no_grad():
                            ns_t = torch.FloatTensor(ns).unsqueeze(0)
                            _, last_value = model(ns_t)
                            last_value = last_value.item()
                        ppo_update(last_value)

            else:
                update_game_logic()

        if mode != 'train':
            draw_game_elements(screen, font, l_font)

        # Episode handling
        if game_state == GAME_STATE_ROOM_CLEARED:
            win_streak += 1
            total_wins += 1
            current_episode += 1

            if mode == 'train':
                if current_episode >= 50:
                    wr = total_wins / current_episode
                    if wr > best_win_rate:
                        best_win_rate = wr
                        torch.save(model.state_dict(), 'ppo_model_best.pth')
                print(f'WIN  | Episode: {current_episode}/{max_episodes} | '
                      f'Win Streak: {win_streak} | '
                      f'Total Wins: {total_wins} | '
                      f'Win Rate: {total_wins/current_episode*100:.1f}% | '
                      f'Steps: {total_steps}')
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
                    if current_episode >= 50:
                        wr = total_wins / current_episode
                        if wr > best_win_rate:
                            best_win_rate = wr
                            torch.save(model.state_dict(), 'ppo_model_best.pth')
                print(f'LOSS | Episode: {current_episode}/{max_episodes} | '
                      f'Total Wins: {total_wins} | '
                      f'Win Rate: {total_wins/current_episode*100:.1f}%' +
                      (f' | Steps: {total_steps}' if mode == 'train' else ''))

                if current_episode < max_episodes:
                    reset_game()
                else:
                    running = False

        clock.tick(FPS)

    if mode == 'train':
        torch.save(model.state_dict(), 'ppo_model.pth')
        print(f'\nTraining complete. Final model saved to ppo_model.pth')
        print(f'Best model saved to ppo_model_best.pth (best win rate: {best_win_rate*100:.1f}%)')

    pygame.quit()