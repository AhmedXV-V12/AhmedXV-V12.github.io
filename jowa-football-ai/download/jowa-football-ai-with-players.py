import pygame
import random
import math
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if torch.cuda.is_available() and torch.cuda.get_device_name(0).lower().startswith("nvidia"):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 600, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3v3 Soccer AI - Enhanced")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# Game settings
PLAYER_RADIUS = 15
BALL_RADIUS = 10
MAX_SPEED = 3
KICK_FORCE = 5
GOAL_WIDTH = 100
FIELD_MARGIN = 50
score_left = 0  # Blue team score
score_right = 0  # Red team score

# Game timing
HALF_DURATION = 120  # 2 minutes per half in seconds
MATCH_BREAK = 5  # 5 seconds between matches

# Movement actions
ACTIONS = [
    (0, 0),     # 0: stop
    (-1, 0),    # 1: left
    (1, 0),     # 2: right
    (0, -1),    # 3: up
    (0, 1),     # 4: down
    (-1, -1),   # 5: left-up
    (1, -1),    # 6: right-up
    (-1, 1),    # 7: left-down
    (1, 1),     # 8: right-down
]

# Enhanced Q-Learning Agent
class AdvancedQAgent:

    def __init__(self, name, team_color):
        self.name = name
        self.team_color = team_color
        self.q_table = self.load_q_table()
        self.lr = 0.5
        qfile = f"{self.name}_qtable.json"  # اسم الملف حسب اللاعب


        if os.path.exists(qfile) and os.path.getsize(qfile) > 100*1024:  # 100KB
            self.discount = 0.99995
            self.epsilon = 0.2
            self.epsilon_decay = 0.99995
        else:
            self.discount = 0.95
            self.epsilon = 0.2
            self.epsilon_decay = 0.95
        self.min_epsilon = 0.01
        self.last_state = None
        self.last_action = None
        self.role = "midfielder"  # defender, midfielder, attacker
        self.save_counter = 0

    def load_q_table(self,):
        json_file = f"{self.name}_qtable.json"
        if os.path.exists(json_file):
            try:
                with open(json_file, "r") as f:
                    raw_table = json.load(f)
                return {eval(k): v for k, v in raw_table.items()}
            except:
                return {}
        return {}

    def save_q_table(self):
        json_file = f"{self.name}_qtable.json"
        try:
            with open(json_file, "w") as f:
                json.dump({str(k): v for k, v in self.q_table.items()}, f)
        except Exception as e:
            print(f"Error saving Q-table for {self.name}: {e}")


    def get_enhanced_state(self, player, ball, teammates, opponents):
        """Enhanced state with tactical information"""
        # Ball position relative to player
        ball_dx = int((ball.x - player.x) / 30)
        ball_dy = int((ball.y - player.y) / 30)
        ball_dx = max(-10, min(10, ball_dx))
        ball_dy = max(-10, min(10, ball_dy))
        
        # Distance to target goal
        target_goal_x = WIDTH - FIELD_MARGIN if self.team_color == RED else FIELD_MARGIN
        goal_dx = int((target_goal_x - player.x) / 50)
        goal_dy = int((HEIGHT // 2 - player.y) / 50)
        goal_dx = max(-5, min(5, goal_dx))
        goal_dy = max(-5, min(5, goal_dy))
        
        # Closest opponent
        if opponents:
            closest_opp = min(opponents, key=lambda p: math.hypot(p.x - player.x, p.y - player.y))
            opp_dx = int((closest_opp.x - player.x) / 40)
            opp_dy = int((closest_opp.y - player.y) / 40)
            opp_dx = max(-5, min(5, opp_dx))
            opp_dy = max(-5, min(5, opp_dy))
        else:
            opp_dx, opp_dy = 0, 0
        
        # Closest teammate
        if teammates:
            closest_team = min(teammates, key=lambda p: math.hypot(p.x - player.x, p.y - player.y))
            team_dx = int((closest_team.x - player.x) / 40)
            team_dy = int((closest_team.y - player.y) / 40)
            team_dx = max(-3, min(3, team_dx))
            team_dy = max(-3, min(3, team_dy))
        else:
            team_dx, team_dy = 0, 0
        
        # Is player closest to ball?
        ball_dist = math.hypot(ball.x - player.x, ball.y - player.y)
        is_closest = 1 if self.is_closest_to_ball(player, ball, teammates) else 0
        
        # Field zone
        field_zone = 0  # defense
        if self.team_color == RED:
            if player.x > WIDTH * 0.33:
                field_zone = 1  # midfield
            if player.x > WIDTH * 0.66:
                field_zone = 2  # attack
        else:
            if player.x < WIDTH * 0.66:
                field_zone = 1
            if player.x < WIDTH * 0.33:
                field_zone = 2
        
        return (ball_dx, ball_dy, goal_dx, goal_dy, opp_dx, opp_dy, 
                team_dx, team_dy, is_closest, field_zone)

    def is_closest_to_ball(self, player, ball, teammates):
        """Check if player is closest to the ball among teammates"""
        player_dist = math.hypot(ball.x - player.x, ball.y - player.y)
        for teammate in teammates:
            teammate_dist = math.hypot(ball.x - teammate.x, ball.y - teammate.y)
            if teammate_dist < player_dist:
                return False
        return True

    def choose_action(self, state):
        self.last_state = state
        
        # Gradually decrease epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        
        if random.random() < self.epsilon:
            self.last_action = random.choice(range(len(ACTIONS)))
            return self.last_action
        
        q_values = self.q_table.get(state, [0] * len(ACTIONS))
        max_q = max(q_values)
        
        # Random choice among best actions
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        self.last_action = random.choice(best_actions)
        return self.last_action

    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0] * len(ACTIONS)
        if next_state not in self.q_table:
            self.q_table[next_state] = [0] * len(ACTIONS)
        
        old_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state])
        new_q = old_q + self.lr * (reward + self.discount * next_max_q - old_q)
        self.q_table[state][action] = new_q
        
        # Periodic saving
        self.save_counter += 1
        if self.save_counter % 1000 == 0:
            self.save_q_table()

# Enhanced Player class
class AdvancedPlayer:
    def __init__(self, x, y, color, agent=None, player_id=0, is_human=False, control_scheme=None):
        self.x = x
        self.y = y
        self.initial_x = x
        self.initial_y = y
        self.color = color
        self.agent = agent
        self.player_id = player_id
        self.vx = 0
        self.vy = 0
        self.has_ball = False
        self.is_human = is_human
        self.control_scheme = control_scheme
        
        # Assign role based on position
        if color == RED:
            if x < WIDTH * 0.25:
                self.role = "defender"
            elif x < WIDTH * 0.5:
                self.role = "midfielder"
            else:
                self.role = "attacker"
        else:
            if x > WIDTH * 0.75:
                self.role = "defender"
            elif x > WIDTH * 0.5:
                self.role = "midfielder"
            else:
                self.role = "attacker"
        
        if agent:
            agent.role = self.role

    def get_teammates(self, all_players):
        return [p for p in all_players if p.color == self.color and p != self]

    def get_opponents(self, all_players):
        return [p for p in all_players if p.color != self.color]

    def calculate_reward(self, ball, all_players, old_pos):
        """Enhanced reward system"""
        reward = 0
        
        # Distance to ball
        ball_dist = math.hypot(ball.x - self.x, ball.y - self.y)
        old_ball_dist = math.hypot(ball.x - old_pos[0], ball.y - old_pos[1])
        
        # Reward for approaching ball (only for closest)
        teammates = self.get_teammates(all_players)
        if self.agent and self.agent.is_closest_to_ball(self, ball, teammates):
            if ball_dist < old_ball_dist:
                reward += 0.1
            if ball_dist < PLAYER_RADIUS + BALL_RADIUS + 5:
                reward += 0.5  # Reward for touching ball
        elif self.agent:
            # Other players should maintain tactical positions
            reward += self.positional_reward()
        
        # Role-specific rewards
        if self.role == "defender":
            # Defender should stay near own goal
            own_goal_x = FIELD_MARGIN if self.color == RED else WIDTH - FIELD_MARGIN
            goal_dist = abs(self.x - own_goal_x)
            if goal_dist < WIDTH * 0.3:
                reward += 0.05
        
        elif self.role == "attacker":
            # Attacker should advance toward opponent goal
            target_goal_x = WIDTH - FIELD_MARGIN if self.color == RED else FIELD_MARGIN
            goal_dist = abs(self.x - target_goal_x)
            if goal_dist < WIDTH * 0.4:
                reward += 0.05
        
        # Penalty for being out of position
        if self.color == RED and self.role == "defender" and self.x > WIDTH * 0.4:
            reward -= 0.1
        elif self.color == BLUE and self.role == "defender" and self.x < WIDTH * 0.6:
            reward -= 0.1
        
        # Reward for good field distribution
        teammates = self.get_teammates(all_players)
        if teammates:
            min_teammate_dist = min(math.hypot(t.x - self.x, t.y - self.y) for t in teammates)
            if min_teammate_dist > 50:  # Good distance between players
                reward += 0.02
            elif min_teammate_dist < 30:  # Too close
                reward -= 0.02
        
        return reward

    def positional_reward(self):
        """Positional reward based on role"""
        reward = 0
        
        if self.role == "defender":
            # Defender prefers to stay in defensive half
            if self.color == RED and self.x < WIDTH * 0.4:
                reward += 0.01
            elif self.color == BLUE and self.x > WIDTH * 0.6:
                reward += 0.01
        
        elif self.role == "attacker":
            # Attacker prefers to advance
            if self.color == RED and self.x > WIDTH * 0.6:
                reward += 0.01
            elif self.color == BLUE and self.x < WIDTH * 0.4:
                reward += 0.01
        
        return reward

    def human_move(self):
        """Handle human player movement based on control scheme"""
        if not self.is_human or not self.control_scheme:
            return
        
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        
        if self.control_scheme == "arrows":
            # Arrow keys
            if keys[pygame.K_LEFT]:
                dx = -1
            if keys[pygame.K_RIGHT]:
                dx = 1
            if keys[pygame.K_UP]:
                dy = -1
            if keys[pygame.K_DOWN]:
                dy = 1
                
        elif self.control_scheme == "ujkh":
            # U/J/K/H keys
            if keys[pygame.K_h]:
                dx = -1
            if keys[pygame.K_k]:
                dx = 1
            if keys[pygame.K_u]:
                dy = -1
            if keys[pygame.K_j]:
                dy = 1
                
        elif self.control_scheme == "wasd":
            # W/A/S/D keys
            if keys[pygame.K_a]:
                dx = -1
            if keys[pygame.K_d]:
                dx = 1
            if keys[pygame.K_w]:
                dy = -1
            if keys[pygame.K_s]:
                dy = 1
        
        # Normalize diagonal movement
        if dx != 0 and dy != 0:
            dx *= 0.7071  # 1/sqrt(2)
            dy *= 0.7071
        
        self.vx, self.vy = dx * MAX_SPEED, dy * MAX_SPEED
        self.x += self.vx
        self.y += self.vy

    def move(self, ball, all_players):
        old_pos = (self.x, self.y)
        
        if self.is_human:
            self.human_move()
        elif self.agent:
            teammates = self.get_teammates(all_players)
            opponents = self.get_opponents(all_players)
            
            state = self.agent.get_enhanced_state(self, ball, teammates, opponents)
            action = self.agent.choose_action(state)
            
            dx, dy = ACTIONS[action]
            self.vx, self.vy = dx * MAX_SPEED, dy * MAX_SPEED
            self.x += self.vx
            self.y += self.vy
            
            # Ball interaction
            ball_dist = math.hypot(self.x - ball.x, self.y - ball.y)
            if ball_dist < PLAYER_RADIUS + BALL_RADIUS:
                # الاستجابة للاصطدام: الكرة تندفع بواسطة اللاعب
                
                # حساب اتجاه الدفع (من اللاعب إلى الكرة)
                push_dx = ball.x - self.x
                push_dy = ball.y - self.y
                push_dist = math.hypot(push_dx, push_dy)
                
                # تطبيع اتجاه الدفع
                if push_dist > 0:
                    push_dx /= push_dist
                    push_dy /= push_dist

                # الكرة تكتسب سرعة اللاعب بالإضافة إلى دفعة صغيرة
                ball.vx = self.vx + push_dx * 1.5
                ball.vy = self.vy + push_dy * 1.5

                # منع الكرة من الالتصاق داخل اللاعب عن طريق إخراجها
                overlap = PLAYER_RADIUS + BALL_RADIUS - ball_dist
                if overlap > 0:
                    ball.x += overlap * push_dx
                    ball.y += overlap * push_dy
            
            # Calculate reward
            reward = self.calculate_reward(ball, all_players, old_pos)
            
            next_state = self.agent.get_enhanced_state(self, ball, teammates, opponents)
            self.agent.update(state, action, reward, next_state)
        
        # Ball interaction for human players
        if self.is_human:
            ball_dist = math.hypot(self.x - ball.x, self.y - ball.y)
            if ball_dist < PLAYER_RADIUS + BALL_RADIUS:
                # الاستجابة للاصطدام: الكرة تندفع بواسطة اللاعب
                
                # حساب اتجاه الدفع (من اللاعب إلى الكرة)
                push_dx = ball.x - self.x
                push_dy = ball.y - self.y
                push_dist = math.hypot(push_dx, push_dy)
                
                # تطبيع اتجاه الدفع
                if push_dist > 0:
                    push_dx /= push_dist
                    push_dy /= push_dist

                # الكرة تكتسب سرعة اللاعب بالإضافة إلى دفعة صغيرة
                ball.vx = self.vx + push_dx * 1.5
                ball.vy = self.vy + push_dy * 1.5

                # منع الكرة من الالتصاق داخل اللاعب عن طريق إخراجها
                overlap = PLAYER_RADIUS + BALL_RADIUS - ball_dist
                if overlap > 0:
                    ball.x += overlap * push_dx
                    ball.y += overlap * push_dy
        
        # Ensure players stay within bounds
        self.x = max(FIELD_MARGIN + PLAYER_RADIUS, 
                    min(WIDTH - FIELD_MARGIN - PLAYER_RADIUS, self.x))
        self.y = max(FIELD_MARGIN + PLAYER_RADIUS, 
                    min(HEIGHT - FIELD_MARGIN - PLAYER_RADIUS, self.y))

    def should_pass(self, teammates, opponents):
        """Determine whether to pass or not"""
        if not teammates:
            return False
        
        # Look for teammate in better position
        target_goal_x = WIDTH - FIELD_MARGIN if self.color == RED else FIELD_MARGIN
        
        for teammate in teammates:
            teammate_goal_dist = abs(teammate.x - target_goal_x)
            my_goal_dist = abs(self.x - target_goal_x)
            
            # If teammate is closer to goal and not surrounded by opponents
            if teammate_goal_dist < my_goal_dist:
                nearby_opponents = sum(1 for opp in opponents 
                                     if math.hypot(opp.x - teammate.x, opp.y - teammate.y) < 60)
                if nearby_opponents < 2:
                    return True
        
        return False

    def pass_ball(self, ball, teammates):
        """Pass ball to best teammate"""
        if not teammates:
            return
        
        target_goal_x = WIDTH - FIELD_MARGIN if self.color == RED else FIELD_MARGIN
        best_teammate = min(teammates, 
                           key=lambda t: abs(t.x - target_goal_x))
        
        dx = best_teammate.x - ball.x
        dy = best_teammate.y - ball.y
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            ball.kick(dx/dist * 0.7, dy/dist * 0.7)

    def kick_toward_goal(self, ball):
        """Kick ball toward goal"""
        target_goal_x = WIDTH - FIELD_MARGIN if self.color == RED else FIELD_MARGIN
        goal_center_y = HEIGHT // 2
        
        dx = target_goal_x - ball.x
        dy = goal_center_y - ball.y
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            ball.kick(dx/dist, dy/dist)

    def draw(self, screen):
        # Draw player with different color based on role
        color = self.color
        if self.role == "defender":
            color = tuple(max(0, c - 50) for c in self.color)  # Darker color
        elif self.role == "attacker":
            color = tuple(min(255, c + 30) for c in self.color)  # Lighter color
        
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), PLAYER_RADIUS)
        
        # Draw player number
        font = pygame.font.SysFont(None, 16)
        text = font.render(str(self.player_id), True, WHITE)
        text_rect = text.get_rect(center=(self.x, self.y))
        screen.blit(text, text_rect)
        
        # Draw indicator for human players
        if self.is_human:
            pygame.draw.circle(screen, YELLOW, (int(self.x), int(self.y)), PLAYER_RADIUS + 3, 2)

# Enhanced Ball class
class AdvancedBall:
    def __init__(self):
        self.trail = []  # Ball trail
        self.reset()

    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.vx = 0
        self.vy = 0
        self.trail.clear()

    def kick(self, dx, dy):
        self.vx += dx * KICK_FORCE
        self.vy += dy * KICK_FORCE
        
        # Maximum velocity
        max_vel = KICK_FORCE * 2
        vel = math.hypot(self.vx, self.vy)
        if vel > max_vel:
            self.vx = (self.vx / vel) * max_vel
            self.vy = (self.vy / vel) * max_vel

    def update(self):
        # Add current position to trail
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 10:
            self.trail.pop(0)
        
        self.x += self.vx
        self.y += self.vy
        
        # Improved friction
        friction = 0.92
        self.vx *= friction
        self.vy *= friction
        
        # Stop ball when velocity is very low
        if abs(self.vx) < 0.1 and abs(self.vy) < 0.1:
            self.vx = 0
            self.vy = 0

        # Bounce from boundaries
        if self.x <= FIELD_MARGIN + BALL_RADIUS:
            if not (HEIGHT // 2 - GOAL_WIDTH // 2 <= self.y <= HEIGHT // 2 + GOAL_WIDTH // 2):
                self.x = FIELD_MARGIN + BALL_RADIUS
                self.vx = abs(self.vx) * 0.8
        
        if self.x >= WIDTH - FIELD_MARGIN - BALL_RADIUS:
            if not (HEIGHT // 2 - GOAL_WIDTH // 2 <= self.y <= HEIGHT // 2 + GOAL_WIDTH // 2):
                self.x = WIDTH - FIELD_MARGIN - BALL_RADIUS
                self.vx = -abs(self.vx) * 0.8
        
        if self.y <= FIELD_MARGIN + BALL_RADIUS:
            self.y = FIELD_MARGIN + BALL_RADIUS
            self.vy = abs(self.vy) * 0.8
        
        if self.y >= HEIGHT - FIELD_MARGIN - BALL_RADIUS:
            self.y = HEIGHT - FIELD_MARGIN - BALL_RADIUS
            self.vy = -abs(self.vy) * 0.8

    def draw(self, screen):
        # Draw ball trail
        for i, pos in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail) * 100
            color = (*WHITE, alpha) if len(self.trail) > 1 else WHITE
            pygame.draw.circle(screen, WHITE, pos, max(1, BALL_RADIUS - (len(self.trail) - i)))
        
        # Draw ball
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), BALL_RADIUS)
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), BALL_RADIUS, 2)

def check_goal_and_reward(players, ball):
    """Check for goals with enhanced reward system"""
    global score_left, score_right
    
    goal_scored = False
    scoring_team = None
    
    # Check left goal (point for BLUE team)
    if (ball.x <= FIELD_MARGIN and 
        HEIGHT // 2 - GOAL_WIDTH // 2 <= ball.y <= HEIGHT // 2 + GOAL_WIDTH // 2):
        score_left += 1
        goal_scored = True
        scoring_team = BLUE
    
    # Check right goal (point for RED team)
    elif (ball.x >= WIDTH - FIELD_MARGIN and 
          HEIGHT // 2 - GOAL_WIDTH // 2 <= ball.y <= HEIGHT // 2 + GOAL_WIDTH // 2):
        score_right += 1
        goal_scored = True
        scoring_team = RED
    
    if goal_scored:
        # Goal rewards and penalties
        for player in players:
            if player.agent:
                teammates = player.get_teammates(players)
                opponents = player.get_opponents(players)
                current_state = player.agent.get_enhanced_state(player, ball, teammates, opponents)
                
                if player.color == scoring_team:
                    # Goal scoring reward
                    ball_dist = math.hypot(ball.x - player.x, ball.y - player.y)
                    if ball_dist < 50:  # Player close to ball
                        reward = 10.0
                    else:
                        reward = 5.0  # Team reward
                else:
                    # Penalty for allowing goal
                    reward = -5.0
                
                player.agent.update(player.agent.last_state, player.agent.last_action, 
                                  reward, current_state)
        
        # Reset positions
        ball.reset()
        for i, player in enumerate(players):
            player.x = player.initial_x + random.randint(-30, 30)
            player.y = player.initial_y + random.randint(-30, 30)

def draw_enhanced_field(screen):
    """Draw enhanced field"""
    screen.fill(GREEN)
    
    # Field outline
    pygame.draw.rect(screen, WHITE, 
                    (FIELD_MARGIN, FIELD_MARGIN, 
                     WIDTH - 2 * FIELD_MARGIN, HEIGHT - 2 * FIELD_MARGIN), 3)
    
    # Center line
    pygame.draw.line(screen, WHITE, 
                    (WIDTH // 2, FIELD_MARGIN), 
                    (WIDTH // 2, HEIGHT - FIELD_MARGIN), 3)
    
    # Center circle
    pygame.draw.circle(screen, WHITE, (WIDTH // 2, HEIGHT // 2), 80, 3)
    pygame.draw.circle(screen, WHITE, (WIDTH // 2, HEIGHT // 2), 5)
    
    # Goals
    goal_height = GOAL_WIDTH
    goal_top = HEIGHT // 2 - goal_height // 2
    goal_bottom = HEIGHT // 2 + goal_height // 2
    
    # Left goal (blue)
    pygame.draw.rect(screen, BLUE, (0, goal_top, FIELD_MARGIN, goal_height))
    pygame.draw.rect(screen, WHITE, (0, goal_top, FIELD_MARGIN, goal_height), 3)
    
    # Right goal (red)  
    pygame.draw.rect(screen, RED, (WIDTH - FIELD_MARGIN, goal_top, FIELD_MARGIN, goal_height))
    pygame.draw.rect(screen, WHITE, (WIDTH - FIELD_MARGIN, goal_top, FIELD_MARGIN, goal_height), 3)
    
    # Penalty areas
    penalty_width = 80
    penalty_height = 150
    
    # Left penalty area
    pygame.draw.rect(screen, WHITE, 
                    (FIELD_MARGIN, HEIGHT // 2 - penalty_height // 2, 
                     penalty_width, penalty_height), 2)
    
    # Right penalty area
    pygame.draw.rect(screen, WHITE, 
                    (WIDTH - FIELD_MARGIN - penalty_width, HEIGHT // 2 - penalty_height // 2, 
                     penalty_width, penalty_height), 2)

def draw_game_info(screen, match_time, current_half, match_number, human_players_count):
    """Draw game information"""
    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 24)
    
    # Score - Blue team on left, Red team on right
    score_text = font.render(f"Blue {score_left} - {score_right} Red", True, WHITE)
    screen.blit(score_text, (WIDTH // 2 - 80, 20))
    
    # Match time
    minutes = int(match_time // 60)
    seconds = int(match_time % 60)
    time_text = font.render(f"Time: {minutes:02d}:{seconds:02d}", True, WHITE)
    screen.blit(time_text, (WIDTH // 2 - 80, 60))
    
    # Half and match info
    half_text = small_font.render(f"Half: {current_half} | Match: {match_number}", True, WHITE)
    screen.blit(half_text, (WIDTH // 2 - 60, 100))
    
    # Human players info
    human_text = small_font.render(f"Human Players: {human_players_count}", True, YELLOW)
    screen.blit(human_text, (WIDTH // 2 - 70, 130))
    
    # Additional info
    info_y = HEIGHT - 40
    
    # Average epsilon for agents
    agents_list = [p.agent for p in players if p.agent]
    if agents_list:
        avg_epsilon = sum(agent.epsilon for agent in agents_list) / len(agents_list)
        epsilon_text = small_font.render(f"Learning: {avg_epsilon:.3f}", True, WHITE)
        screen.blit(epsilon_text, (10, info_y))
    
    # Number of learned states
    total_states = sum(len(p.agent.q_table) for p in players if p.agent)
    states_text = small_font.render(f"States: {total_states}", True, WHITE)
    screen.blit(states_text, (150, info_y))

def draw_player_controls(screen, human_players):
    """Draw control information for human players"""
    font = pygame.font.SysFont(None, 20)
    y_offset = 160
    
    for i, player in enumerate(human_players):
        if player.control_scheme == "arrows":
            controls = "Player 1: Arrow Keys"
        elif player.control_scheme == "ujkh":
            controls = "Player 2: U/J/K/H Keys"
        elif player.control_scheme == "wasd":
            controls = "Player 3: W/A/S/D Keys"
        
        control_text = font.render(controls, True, YELLOW)
        screen.blit(control_text, (WIDTH // 2 - 80, y_offset + i * 25))

def get_player_selection():
    """Get number of human players from user"""
    selection_screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Player Selection")
    
    font_large = pygame.font.SysFont(None, 48)
    font_medium = pygame.font.SysFont(None, 36)
    font_small = pygame.font.SysFont(None, 24)
    
    selected_players = 0
    
    while True:
        selection_screen.fill(BLACK)
        
        title = font_large.render("Select Number of Human Players", True, WHITE)
        selection_screen.blit(title, (WIDTH//2 - title.get_width()//2, 80))
        
        for i in range(4):
            color = GREEN if i == selected_players else WHITE
            text = font_medium.render(f"{i} Players", True, color)
            rect = text.get_rect(center=(WIDTH//2, 180 + i*60))
            selection_screen.blit(text, rect)
            
            # Draw control schemes for each option
            if i == 0:
                controls = "All AI Players"
            elif i == 1:
                controls = "Player 1: Arrow Keys"
            elif i == 2:
                controls = "Player 1: Arrows, Player 2: U/J/K/H"
            elif i == 3:
                controls = "Player 1: Arrows, Player 2: U/J/K/H, Player 3: W/A/S/D"
            
            control_text = font_small.render(controls, True, GRAY)
            control_rect = control_text.get_rect(center=(WIDTH//2, 210 + i*60))
            selection_screen.blit(control_text, control_rect)
        
        instruction = font_small.render("Use UP/DOWN arrows to select, ENTER to confirm", True, WHITE)
        selection_screen.blit(instruction, (WIDTH//2 - instruction.get_width()//2, HEIGHT - 50))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_players = (selected_players - 1) % 4
                elif event.key == pygame.K_DOWN:
                    selected_players = (selected_players + 1) % 4
                elif event.key == pygame.K_RETURN:
                    return selected_players

# Enhanced game setup
def setup_game(human_players_count):
    """Setup game with specified number of human players"""
    agents = [AdvancedQAgent(f"agent{i}", RED if i < 3 else BLUE) for i in range(6)]
    
    players = []
    human_control_schemes = ["arrows", "ujkh", "wasd"]
    
    # Red team (starts on left)
    for i in range(3):
        is_human = (i < human_players_count)
        control_scheme = human_control_schemes[i] if is_human else None
        agent = None if is_human else agents[i]
        
        if i == 0:
            player = AdvancedPlayer(100, HEIGHT // 2 - 80, RED, agent, i+1, is_human, control_scheme)
        elif i == 1:
            player = AdvancedPlayer(200, HEIGHT // 1.32 - 120, RED, agent, i+1, is_human, control_scheme)
        else:
            player = AdvancedPlayer(300, HEIGHT // 3.5 + 120, RED, agent, i+1, is_human, control_scheme)
        
        players.append(player)
    
    # Blue team (starts on right)  
    for i in range(3):
        is_human = (i + 3 < human_players_count)  # Only if we have more than 3 human players
        control_scheme = human_control_schemes[i] if is_human else None
        agent = None if is_human else agents[i+3]
        
        if i == 0:
            player = AdvancedPlayer(WIDTH - 100, HEIGHT // 2 - 80, BLUE, agent, i+4, is_human, control_scheme)
        elif i == 1:
            player = AdvancedPlayer(WIDTH - 200, HEIGHT // 1.32 - 120, BLUE, agent, i+4, is_human, control_scheme)
        else:
            player = AdvancedPlayer(WIDTH - 300, HEIGHT // 4.5 + 121, BLUE, agent, i+4, is_human, control_scheme)
        
        players.append(player)
    
    return players, agents

# Game statistics
game_stats = {
    'total_goals': 0,
    'red_possession': 0,
    'blue_possession': 0,
    'game_time': 0,
    'last_save_time': time.time(),
    'match_number': 1,
    'current_half': 1,
    'match_start_time': time.time(),
    'match_elapsed_time': 0,
    'is_break': False,
    'break_start_time': 0
}

def update_possession(ball, players):
    """Update ball possession statistics"""
    closest_player = min(players, key=lambda p: math.hypot(p.x - ball.x, p.y - ball.y))
    ball_dist = math.hypot(closest_player.x - ball.x, closest_player.y - ball.y)
    
    if ball_dist < PLAYER_RADIUS + BALL_RADIUS + 20:
        if closest_player.color == RED:
            game_stats['red_possession'] += 1
        else:
            game_stats['blue_possession'] += 1

def auto_save_progress(agents):
    """Auto-save progress"""
    current_time = time.time()
    if current_time - game_stats['last_save_time'] > 60:  # Save every minute
        for agent in agents:
            agent.save_q_table()
        game_stats['last_save_time'] = current_time
        print(f"Progress saved - Time: {game_stats['game_time']//60:.0f} minutes")

def reset_positions(players):
    """Reset positions tactically"""
    formations = {
        'red': [(100, HEIGHT // 2 - 80),(200, HEIGHT // 1.32 - 120),(300, HEIGHT // 3.5 + 120)],  # Red vertical formation
        'blue': [(WIDTH - 100, HEIGHT // 2 - 80),(WIDTH - 200, HEIGHT // 1.32 - 120), (WIDTH - 300, HEIGHT // 4.5 + 121)]  # Blue vertical formation
    }
    
    red_players = [p for p in players if p.color == RED]
    blue_players = [p for p in players if p.color == BLUE]
    
    for i, player in enumerate(red_players):
        player.x, player.y = formations['red'][i]
        player.initial_x, player.initial_y = formations['red'][i]
    
    for i, player in enumerate(blue_players):
        player.x, player.y = formations['blue'][i]
        player.initial_x, player.initial_y = formations['blue'][i]

def draw_tactical_info(screen):
    """Draw tactical information"""
    font = pygame.font.SysFont(None, 20)
    
    # Draw connection lines between closest players and ball
    closest_red = min([p for p in players if p.color == RED], 
                     key=lambda p: math.hypot(p.x - ball.x, p.y - ball.y))
    closest_blue = min([p for p in players if p.color == BLUE], 
                      key=lambda p: math.hypot(p.x - ball.x, p.y - ball.y))
    
    # Line for closest player to ball
    closest_overall = min(players, key=lambda p: math.hypot(p.x - ball.x, p.y - ball.y))
    if math.hypot(closest_overall.x - ball.x, closest_overall.y - ball.y) < 100:
        pygame.draw.line(screen, closest_overall.color, 
                        (closest_overall.x, closest_overall.y), 
                        (ball.x, ball.y), 2)
    
    # Show roles
    role_colors = {'defender': GRAY, 'midfielder': YELLOW, 'attacker': WHITE}
    for player in players:
        role_text = font.render(player.role[:3].upper(), True, role_colors.get(player.role, WHITE))
        screen.blit(role_text, (player.x - 15, player.y + 20))

def draw_advanced_stats(screen):
    """Draw advanced statistics"""
    font = pygame.font.SysFont(None, 18)
    
    # Calculate possession percentage
    total_possession = game_stats['red_possession'] + game_stats['blue_possession']
    if total_possession > 0:
        red_pct = (game_stats['red_possession'] / total_possession) * 100
        blue_pct = (game_stats['blue_possession'] / total_possession) * 100
        
        poss_text = font.render(f"Possession - Red: {red_pct:.1f}% Blue: {blue_pct:.1f}%", True, WHITE)
        screen.blit(poss_text, (WIDTH - 300, HEIGHT - 60))
    
    # Show game time
    minutes = game_stats['game_time'] // 3600  # 60 FPS * 60 seconds
    time_text = font.render(f"Time: {minutes:.0f} minutes", True, WHITE)
    screen.blit(time_text, (WIDTH - 150, HEIGHT - 20))

def start_new_match(players):
    """Start a new match with reset scores and positions"""
    global score_left, score_right
    score_left = 0
    score_right = 0
    game_stats['match_number'] += 1
    game_stats['current_half'] = 1
    game_stats['match_start_time'] = time.time()
    game_stats['match_elapsed_time'] = 0
    game_stats['is_break'] = False
    ball.reset()
    reset_positions(players)
    print(f"Starting match {game_stats['match_number']}")

def check_match_status():
    """Check if we need to switch halves or start a new match"""
    current_time = time.time()
    game_stats['match_elapsed_time'] = current_time - game_stats['match_start_time']
    
    # Check if we're in break time between halves
    if game_stats['is_break']:
        if current_time - game_stats['break_start_time'] >= MATCH_BREAK:
            game_stats['is_break'] = False
            game_stats['current_half'] = 2 if game_stats['current_half'] == 1 else 1
            game_stats['match_start_time'] = current_time
            ball.reset()
            reset_positions(players)
            print(f"Starting half {game_stats['current_half']} of match {game_stats['match_number']}")
        return False
    
    # Check if half time is over
    if game_stats['match_elapsed_time'] >= HALF_DURATION:
        if game_stats['current_half'] == 1:
            # First half ended, start break
            game_stats['is_break'] = True
            game_stats['break_start_time'] = current_time
            print(f"Half time! Score: Blue {score_left} - {score_right} Red")
            return True
        else:
            # Second half ended, start new match after break
            game_stats['is_break'] = True
            game_stats['break_start_time'] = current_time
            print(f"Match {game_stats['match_number']} ended! Final score: Blue {score_left} - {score_right} Red")
            return True
    
    return False

# Enhanced main game loop
def main_game_loop():
    global running, players, agents, ball
    
    # Get player selection
    human_players_count = get_player_selection()
    
    # Setup game with selected number of human players
    players, agents = setup_game(human_players_count)
    ball = AdvancedBall()
    
    running = True
    
    print("Starting enhanced soccer simulation...")
    print(f"Human players: {human_players_count}")
    print("Press ESC to quit, SPACE to reset, P to pause")
    
    paused = False
    
    while running:
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Reset game
                    ball.reset()
                    reset_positions(players)
                    print("Game reset")
                elif event.key == pygame.K_p:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_r:
                    # Reset score
                    global score_left, score_right
                    score_left = score_right = 0
                    game_stats['total_goals'] = 0
                    print("Score reset")
        
        if not paused:
            # Update game time
            game_stats['game_time'] += 1
            
            # Check if we need to switch halves or start new match
            is_break = check_match_status()
            
            if not game_stats['is_break']:
                # Normal game play
                for player in players:
                    player.move(ball, players)
                
                ball.update()
                update_possession(ball, players)
                check_goal_and_reward(players, ball)
                auto_save_progress(agents)
        
        # Drawing
        draw_enhanced_field(screen)
        draw_tactical_info(screen)
        
        for player in players:
            player.draw(screen)
        
        ball.draw(screen)
        
        # Calculate match time for display
        if game_stats['is_break']:
            match_time = HALF_DURATION  # Show full time during break
        else:
            match_time = game_stats['match_elapsed_time']
        
        draw_game_info(screen, match_time, game_stats['current_half'], game_stats['match_number'], human_players_count)
        
        # Draw player controls info
        human_players = [p for p in players if p.is_human]
        if human_players:
            draw_player_controls(screen, human_players)
        
        draw_advanced_stats(screen)
        
        # Show break message if applicable
        if game_stats['is_break']:
            font = pygame.font.SysFont(None, 48)
            if game_stats['current_half'] == 1:
                break_text = font.render("HALF TIME", True, YELLOW)
            else:
                break_text = font.render("MATCH ENDED", True, YELLOW)
            
            screen.blit(break_text, (WIDTH // 2 - 100, HEIGHT // 2 - 24))
            
            # Countdown for next half/match
            countdown = MATCH_BREAK - (time.time() - game_stats['break_start_time'])
            count_font = pygame.font.SysFont(None, 36)
            count_text = count_font.render(f"Next: {int(countdown)}", True, WHITE)
            screen.blit(count_text, (WIDTH // 2 - 40, HEIGHT // 2 + 24))
        
        if paused:
            font = pygame.font.SysFont(None, 48)
            pause_text = font.render("PAUSED", True, YELLOW)
            screen.blit(pause_text, (WIDTH // 2 - 50, HEIGHT // 2))
        
        pygame.display.flip()
    
    # Final save
    print("Saving final progress...")
    for agent in agents:
        agent.save_q_table()
    
    print("Game statistics:")
    print(f"Total goals: {score_left + score_right}")
    print(f"Winner: {'Red' if score_right > score_left else 'Blue' if score_left > score_right else 'Draw'}")
    
    total_poss = game_stats['red_possession'] + game_stats['blue_possession']
    if total_poss > 0:
        print(f"Possession - Red: {(game_stats['red_possession']/total_poss)*100:.1f}%")
        print(f"Possession - Blue: {(game_stats['blue_possession']/total_poss)*100:.1f}%")

    pygame.quit()

if __name__ == "__main__":
    main_game_loop()