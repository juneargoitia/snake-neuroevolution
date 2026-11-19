import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

Point = namedtuple('Point', 'x, y')

BG_COLOR = (18, 20, 24)
PANEL_COLOR = (28, 32, 38)
GRID_COLOR = (30, 34, 40)
BORDER_COLOR = (60, 68, 80)

WHITE = (240, 244, 248)
GRAY = (140, 150, 165)
ACCENT_BLUE = (65, 155, 249)
ACCENT_YELLOW = (250, 200, 50)
DANGER_RED = (235, 75, 75)
FOOD_RED = (240, 65, 80)
FOOD_LEAF = (80, 210, 100)

SNAKE_HEAD = (50, 230, 120)
SNAKE_BODY_START = (40, 195, 100)
SNAKE_BODY_END = (20, 110, 60)

BLOCK_SIZE = 22
GRID_WIDTH = 20
GRID_HEIGHT = 20
PANEL_WIDTH = 240
SCORE_W = 1000
SHAPING_W = 20

class SnakeGame:
    def __init__(self, render=True, speed=15):
        self.render = render
        self.speed = speed
        self.paused = False
        self.board_width = GRID_WIDTH * BLOCK_SIZE
        self.board_height = GRID_HEIGHT * BLOCK_SIZE
        self.total_width = self.board_width + PANEL_WIDTH
        
        self.generation = 1
        self.best_score = 0
        self.individual = 1

        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.total_width, self.board_height))
            pygame.display.set_caption('Snake - Neuroevolution')
            self.clock = pygame.time.Clock()
            self.font_title = pygame.font.SysFont('segoeui', 20, bold=True)
            self.font_main = pygame.font.SysFont('segoeui', 16)
            self.font_small = pygame.font.SysFont('segoeui', 13)

        self.reset()
    
    def set_training_info(self, generation, best_score, individual=1):
        self.generation = generation
        self.best_score = best_score
        self.individual = individual

    def reset(self):
        self.direction = Direction.RIGHT
        
        start_x = (GRID_WIDTH // 2) * BLOCK_SIZE
        start_y = (GRID_HEIGHT // 2) * BLOCK_SIZE
        self.head = Point(start_x, start_y)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]

        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.shaping = 0.0
        self.food = None
        self._place_food()
        
        self.prev_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        self.distance_reward = 0.0

    def _place_food(self):
        occupied = set(self.snake)
        free = [
            Point(x * BLOCK_SIZE, y * BLOCK_SIZE)
            for x in range(GRID_WIDTH)
            for y in range(GRID_HEIGHT)
            if Point(x * BLOCK_SIZE, y * BLOCK_SIZE) not in occupied
        ]
        self.food = random.choice(free) if free else Point(-BLOCK_SIZE, -BLOCK_SIZE)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x >= self.board_width or pt.x < 0 or pt.y >= self.board_height or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _ray(self, dx, dy):
        pt = self.head
        dist = 0
        while True:
            pt = Point(pt.x + dx * BLOCK_SIZE, pt.y + dy * BLOCK_SIZE)
            dist += 1
            if self.is_collision(pt):
                return 1.0 / dist

    def get_state(self):
        fx, fy = {
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.UP: (0, -1),
        }[self.direction]
        rx, ry = -fy, fx
        lx, ly = fy, -fx

        vx = (self.food.x - self.head.x) / BLOCK_SIZE
        vy = (self.food.y - self.head.y) / BLOCK_SIZE
        ahead = vx * fx + vy * fy
        side = vx * rx + vy * ry

        state = [
            self._ray(fx, fy),
            self._ray(rx, ry),
            self._ray(lx, ly),
            ahead / GRID_WIDTH,
            side / GRID_WIDTH,
            1.0 if ahead > 0 else 0.0,
            1.0 if side > 0 else 0.0,
            1.0 if side < 0 else 0.0,
        ]
        return np.array(state, dtype=float)
    
    def _handle_events(self):
        """Gestiona cierre de ventana y controles de velocidad/pausa en directo."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.speed = min(120, self.speed + 5)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(5, self.speed - 5)

        while self.paused:
            self._update_ui()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.paused = False
            self.clock.tick(10)

    def step(self, action):
        self.steps += 1
        self.steps_without_food += 1

        if self.render:
            self._handle_events()

        dist_before = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        self._move(action)
        dist_after = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        self.shaping += 1.0 if dist_after < dist_before else -1.5
        self.snake.insert(0, self.head)

        current_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        if current_distance < self.prev_distance:
            self.distance_reward += 2.0   # Premia ir directo
        else:
            self.distance_reward -= 3.0
        self.prev_distance = current_distance

        max_steps = 60 * len(self.snake)
        if self.is_collision() or self.steps_without_food > max_steps:
            return True, self.score

        if self.head == self.food:
            self.score += 1
            self.steps_without_food = 0
            if self.score > self.best_score:
                self.best_score = self.score
            self._place_food()
            self.prev_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        else:
            self.snake.pop()

        if self.render:
            self._update_ui()
            self.clock.tick(self.speed)

        return False, self.score

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 0:    # Seguir recto
            new_dir = clock_wise[idx]
        elif action == 1:  # Girar 90° derecha
            new_dir = clock_wise[(idx + 1) % 4]
        elif action == 2:  # Girar 90° izquierda
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_fitness(self):
        return (self.score ** 2) * SCORE_W + (self.score * 500) + SHAPING_W * self.shaping
    
    def _draw_snake(self):
        total = len(self.snake)
        for i, pt in enumerate(self.snake):
            if i == 0:
                color = SNAKE_HEAD
            else:
                ratio = i / max(total, 1)
                r = int(SNAKE_BODY_START[0] * (1 - ratio) + SNAKE_BODY_END[0] * ratio)
                g = int(SNAKE_BODY_START[1] * (1 - ratio) + SNAKE_BODY_END[1] * ratio)
                b = int(SNAKE_BODY_START[2] * (1 - ratio) + SNAKE_BODY_END[2] * ratio)
                color = (r, g, b)

            rect = pygame.Rect(pt.x + 1, pt.y + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2)
            pygame.draw.rect(self.display, color, rect, border_radius=5)

            if i == 0:
                eye_r = 2
                if self.direction == Direction.RIGHT:
                    e1, e2 = (pt.x + 15, pt.y + 6), (pt.x + 15, pt.y + 15)
                elif self.direction == Direction.LEFT:
                    e1, e2 = (pt.x + 6, pt.y + 6), (pt.x + 6, pt.y + 15)
                elif self.direction == Direction.UP:
                    e1, e2 = (pt.x + 6, pt.y + 6), (pt.x + 15, pt.y + 6)
                else:
                    e1, e2 = (pt.x + 6, pt.y + 15), (pt.x + 15, pt.y + 15)
                pygame.draw.circle(self.display, BG_COLOR, e1, eye_r)
                pygame.draw.circle(self.display, BG_COLOR, e2, eye_r)

    def _draw_food(self):
        center = (self.food.x + BLOCK_SIZE // 2, self.food.y + BLOCK_SIZE // 2)
        pygame.draw.circle(self.display, FOOD_RED, center, BLOCK_SIZE // 2 - 2)
        leaf_rect = pygame.Rect(self.food.x + BLOCK_SIZE // 2, self.food.y + 2, 5, 4)
        pygame.draw.ellipse(self.display, FOOD_LEAF, leaf_rect)

    def _draw_panel(self):
        panel_x = self.board_width
        pygame.draw.rect(self.display, PANEL_COLOR, (panel_x, 0, PANEL_WIDTH, self.board_height))
        pygame.draw.line(self.display, BORDER_COLOR, (panel_x, 0), (panel_x, self.board_height), 2)

        x_pad = panel_x + 18
        y = 18

        title = self.font_title.render("NEURO-SNAKE", True, ACCENT_BLUE)
        self.display.blit(title, (x_pad, y))
        y += 38

        metrics = [
            (f"Generación:  {self.generation}", WHITE),
            (f"Individuo:    #{self.individual}", GRAY),
            (f"Manzanas:   {self.score}", SNAKE_HEAD),
            (f"Récord:       {self.best_score}", ACCENT_YELLOW),
            (f"Fitness:      {int(self.get_fitness())}", WHITE),
        ]
        for text_str, color in metrics:
            surf = self.font_main.render(text_str, True, color)
            self.display.blit(surf, (x_pad, y))
            y += 26

        y += 8
        max_steps = 60 * len(self.snake)
        energy_ratio = max(0.0, 1.0 - (self.steps_without_food / max_steps))
        lbl_energy = self.font_small.render("Energía restante:", True, GRAY)
        self.display.blit(lbl_energy, (x_pad, y))
        y += 18
        pygame.draw.rect(self.display, BG_COLOR, (x_pad, y, PANEL_WIDTH - 36, 10), border_radius=4)
        bar_color = SNAKE_HEAD if energy_ratio > 0.3 else DANGER_RED
        pygame.draw.rect(self.display, bar_color, (x_pad, y, int((PANEL_WIDTH - 36) * energy_ratio), 10), border_radius=4)
        y += 24

        state = self.get_state()
        lbl_sens = self.font_small.render("Sensores de peligro:", True, GRAY)
        self.display.blit(lbl_sens, (x_pad, y))
        y += 20

        dangers = [("Recto", state[0] >= 1.0), ("Der", state[1] >= 1.0), ("Izq", state[2] >= 1.0)]
        for idx, (name, active) in enumerate(dangers):
            bx = x_pad + idx * 68
            box_color = DANGER_RED if active else BG_COLOR
            pygame.draw.rect(self.display, box_color, (bx, y, 60, 24), border_radius=4)
            txt = self.font_small.render(name, True, WHITE)
            self.display.blit(txt, (bx + 12, y + 4))
        y += 42

        pygame.draw.line(self.display, BORDER_COLOR, (x_pad, y), (panel_x + PANEL_WIDTH - 18, y), 1)
        y += 12
        status = "PAUSADO (Espacio)" if self.paused else f"Velocidad: {self.speed} FPS"
        ctrl1 = self.font_small.render(status, True, ACCENT_YELLOW if self.paused else GRAY)
        ctrl2 = self.font_small.render("Flechas Arr/Abj: Velocidad", True, GRAY)
        self.display.blit(ctrl1, (x_pad, y))
        self.display.blit(ctrl2, (x_pad, y + 18))

    def _update_ui(self):
        self.display.fill(BG_COLOR)

        for x in range(0, self.board_width, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (x, 0), (x, self.board_height))
        for y in range(0, self.board_height, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (0, y), (self.board_width, y))

        self._draw_food()
        self._draw_snake()
        self._draw_panel()

        pygame.display.flip()


if __name__ == '__main__':
    game = SnakeGame(render=True, speed=12)
    game.set_training_info(generation=1, best_score=0, individual=1)
    game_over = False

    while not game_over:
        state = game.get_state()
        action = random.choice([0, 1, 2])
        game_over, score = game.step(action)

    print(f"Fin de la prueba | Manzanas: {score} | Fitness: {game.get_fitness()}")
    pygame.quit()