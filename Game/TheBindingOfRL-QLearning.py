import pygame
import math
import random
import numpy as np
import sys
import os

# Random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

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

# Pygame Init
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("The Binding of Learning - Pygame Combat Agent")

# Image Asset Loading (Robust individual loading)
IMAGES = {}
image_assets = {
    'player': ('Assests/player.png', (50, 50)),
    'bullet': ('Assests/bullet.png', (10, 10)),
    'enemy_bullet': ('Assests/enemy_bullet.png', (12, 12)),
    'normal': ('Assests/enemy_normal.png', (40, 40)),
    'fast': ('Assests/enemy_fast.png', (30, 30)),
    'shooter': ('Assests/enemy_shooter.png', (50, 50)),
    'destructible': ('Assests/obstacle_destructible.png', (30, 30)),
    'indestructible': ('Assests/obstacle_indestructible.png', (70, 70))
}

for key, (filename, size) in image_assets.items():
    if os.path.exists(filename):
        try:
            img = pygame.image.load(filename).convert_alpha()
            IMAGES[key] = pygame.transform.scale(img, size)
        except Exception as e:
            print(f"Warning: Could not load {filename} - {e}")
    else:
        print(f"Warning: {filename} not found on disk. Falling back to color for '{key}'.")

print(f"Loaded {len(IMAGES)} images successfully: {list(IMAGES.keys())}")

# Sound Asset Loading
SOUNDS = {}
try:
    pygame.mixer.init()
    SOUNDS['shoot'] = pygame.mixer.Sound('Assests/shoot.wav')
    SOUNDS['enemy_death'] = pygame.mixer.Sound('Assests/enemy_death.wav')
    SOUNDS['game_over'] = pygame.mixer.Sound('Assests/game_over.wav')
    SOUNDS['victory'] = pygame.mixer.Sound('Assests/victory.wav')
    pygame.mixer.music.load('Assests/background_music.mp3')
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play(-1) # Play indefinitely
    print("Sounds and music loaded successfully.")
except Exception as e:
    print(f"Sound/Music loading failed ({e}). Running silently.")
    SOUNDS['shoot'] = None
    SOUNDS['enemy_death'] = None
    SOUNDS['game_over'] = None
    SOUNDS['victory'] = None

# Player Setup
PLAYER_SIZE = 50
PLAYER_SPEED = 5
player_x_initial = float((SCREEN_WIDTH - PLAYER_SIZE) // 2)
player_y_initial = float((SCREEN_HEIGHT - PLAYER_SIZE) // 2)
player_x, player_y = player_x_initial, player_y_initial
player_rect = pygame.Rect(int(player_x), int(player_y), PLAYER_SIZE, PLAYER_SIZE)
player_health_initial = 2
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
    'shooter':{'size': 50, 'speed': 1,   'health': 3, 'color': PURPLE,  'can_shoot': True,  'fire_rate': 2000}
}
OBSTACLE_TYPES = {
    'destructible':   {'size': 30, 'health': 2,  'color': DARK_GRAY},
    'indestructible': {'size': 70, 'health': -1, 'color': GRAY}
}
MAX_ENEMIES = 5
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

BULLET_SIZE, BULLET_SPEED, SHOOT_COOLDOWN = 10, 10, 200
bullets = []
last_shot_time = 0
ENEMY_BULLET_SIZE, ENEMY_BULLET_SPEED = 12, 7
enemy_bullets = []


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
    for _ in range(ROOM_ENEMIES_TOTAL): spawn_enemy(random.choice(list(ENEMY_TYPES.keys())))


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
            if SOUNDS.get('shoot'):
                try:
                    SOUNDS['shoot'].play()
                except Exception:
                    pass

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
                e['health'] -= 1
                if b in bullets: bullets.remove(b)
                if e['health'] <= 0:
                    if e in enemies: enemies.remove(e)
                    score += 200
                    if SOUNDS.get('enemy_death'):
                        try:
                            SOUNDS['enemy_death'].play()
                        except Exception:
                            pass
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
        if SOUNDS.get('game_over'):
            try:
                SOUNDS['game_over'].play()
            except Exception:
                pass
    elif not enemies:
        score += 4000
        game_state = GAME_STATE_ROOM_CLEARED
        if SOUNDS.get('victory'):
            try:
                SOUNDS['victory'].play()
            except Exception:
                pass


def draw_game_elements(screen, font, large_font):
    screen.fill(BLACK)
    if game_state in (GAME_STATE_PLAYING, GAME_STATE_PAUSED):
        for obs in obstacles:
            if obs['type'] in IMAGES:
                screen.blit(IMAGES[obs['type']], obs['rect'])
            else:
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
        f"Score: {score}  HP: {player_health}  Enemies: {len(enemies)}",
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
    font    = pygame.font.Font(None, 36)
    l_font  = pygame.font.Font(None, 74)
    clock   = pygame.time.Clock()

    reset_game()
    running = True

    while running:
        handle_input()

        if game_state == GAME_STATE_PLAYING:
            update_game_logic()

        draw_game_elements(screen, font, l_font)

        clock.tick(FPS)

    pygame.quit()