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
        """Permite a main.py pasar datos de la evolución para mostrarlos en pantalla."""
        self.generation = generation
        self.best_score = best_score
        self.individual = individual

    def reset(self):
        """Reinicia el juego al estado inicial."""
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
        self.food = None
        self._place_food()

    def _place_food(self):
        """Coloca la manzana en una celda libre aleatoria."""
        while True:
            x = random.randint(0, GRID_WIDTH - 1) * BLOCK_SIZE
            y = random.randint(0, GRID_HEIGHT - 1) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def is_collision(self, pt=None):
        """Comprueba si un punto choca con los bordes o con el cuerpo."""
        if pt is None:
            pt = self.head
        # Choque con paredes
        if pt.x >= self.board_width or pt.x < 0 or pt.y >= self.board_height or pt.y < 0:
            return True
        # Choque consigo misma
        if pt in self.snake[1:]:
            return True
        return False

    def get_state(self):
        """
        Devuelve el vector de 11 sensores (0 o 1) para la Red Neuronal:
        [peligro_recto, peligro_der, peligro_izq,
            dir_izq, dir_der, dir_arriba, dir_abajo,
            comida_izq, comida_der, comida_arriba, comida_abajo]
        """
        head = self.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # 1. Peligro recto
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # 2. Peligro a la derecha relativa
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # 3. Peligro a la izquierda relativa
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # 4-7. Dirección actual
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # 8-11. Posición relativa de la comida
            self.food.x < head.x,  # comida a la izquierda
            self.food.x > head.x,  # comida a la derecha
            self.food.y < head.y,  # comida arriba
            self.food.y > head.y   # comida abajo
        ]

        return np.array(state, dtype=int)
    
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
        """
        Ejecuta un paso del juego dada una acción de la IA:
        action: 0 = recto, 1 = girar derecha, 2 = girar izquierda
        Devuelve: (game_over, score)
        """
        self.steps += 1
        self.steps_without_food += 1

        if self.render:
            self._handle_events()

        # Actualizar dirección
        self._move(action)
        self.snake.insert(0, self.head)

        # Comprobar muerte (choque o bucle infinito sin comer)
        max_steps = 100 * len(self.snake)
        if self.is_collision() or self.steps_without_food > max_steps:
            return True, self.score

        # Comprobar si comió manzana
        if self.head == self.food:
            self.score += 1
            self.steps_without_food = 0
            if self.score > self.best_score:
                self.best_score = self.score
            self._place_food()
        else:
            self.snake.pop()

        if self.render:
            self._update_ui()
            self.clock.tick(self.speed)

        return False, self.score

    def _move(self, action):
        """Calcula la nueva posición de la cabeza según la acción."""
        # Orden horario: RIGHT(0), DOWN(1), LEFT(2), UP(3)
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
        """Calcula la puntuación de aptitud para el Algoritmo Genético."""
        return self.steps + (self.score * 100)
    
    def _draw_snake(self):
        """Dibuja la serpiente con degradado de color y ojos orientados."""
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

            # Dibujar ojos en la cabeza
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
        """Dibuja una manzana redondeada con un pequeño tallo verde."""
        center = (self.food.x + BLOCK_SIZE // 2, self.food.y + BLOCK_SIZE // 2)
        pygame.draw.circle(self.display, FOOD_RED, center, BLOCK_SIZE // 2 - 2)
        leaf_rect = pygame.Rect(self.food.x + BLOCK_SIZE // 2, self.food.y + 2, 5, 4)
        pygame.draw.ellipse(self.display, FOOD_LEAF, leaf_rect)

    def _draw_panel(self):
        """Dibuja el panel lateral con estadísticas y sensores en vivo."""
        panel_x = self.board_width
        pygame.draw.rect(self.display, PANEL_COLOR, (panel_x, 0, PANEL_WIDTH, self.board_height))
        pygame.draw.line(self.display, BORDER_COLOR, (panel_x, 0), (panel_x, self.board_height), 2)

        x_pad = panel_x + 18
        y = 18

        # Título
        title = self.font_title.render("NEURO-SNAKE", True, ACCENT_BLUE)
        self.display.blit(title, (x_pad, y))
        y += 38

        # Métricas principales
        metrics = [
            (f"Generación:  {self.generation}", WHITE),
            (f"Individuo:    #{self.individual}", GRAY),
            (f"Manzanas:   {self.score}", SNAKE_HEAD),
            (f"Récord:        {self.best_score}", ACCENT_YELLOW),
            (f"Fitness:       {self.get_fitness()}", WHITE),
        ]
        for text_str, color in metrics:
            surf = self.font_main.render(text_str, True, color)
            self.display.blit(surf, (x_pad, y))
            y += 26

        # Barra de energía (pasos restantes antes de morir de hambre)
        y += 8
        max_steps = 100 * len(self.snake)
        energy_ratio = max(0.0, 1.0 - (self.steps_without_food / max_steps))
        lbl_energy = self.font_small.render("Energía restante:", True, GRAY)
        self.display.blit(lbl_energy, (x_pad, y))
        y += 18
        pygame.draw.rect(self.display, BG_COLOR, (x_pad, y, PANEL_WIDTH - 36, 10), border_radius=4)
        bar_color = SNAKE_HEAD if energy_ratio > 0.3 else DANGER_RED
        pygame.draw.rect(self.display, bar_color, (x_pad, y, int((PANEL_WIDTH - 36) * energy_ratio), 10), border_radius=4)
        y += 24

        # Visualizador de sensores en tiempo real
        state = self.get_state()
        lbl_sens = self.font_small.render("Sensores de peligro:", True, GRAY)
        self.display.blit(lbl_sens, (x_pad, y))
        y += 20

        dangers = [("Recto", state[0]), ("Der", state[1]), ("Izq", state[2])]
        for idx, (name, active) in enumerate(dangers):
            bx = x_pad + idx * 68
            box_color = DANGER_RED if active else BG_COLOR
            pygame.draw.rect(self.display, box_color, (bx, y, 60, 24), border_radius=4)
            txt = self.font_small.render(name, True, WHITE)
            self.display.blit(txt, (bx + 12, y + 4))
        y += 42

        # Controles en vivo
        pygame.draw.line(self.display, BORDER_COLOR, (x_pad, y), (panel_x + PANEL_WIDTH - 18, y), 1)
        y += 12
        status = "PAUSADO (Espacio)" if self.paused else f"Velocidad: {self.speed} FPS"
        ctrl1 = self.font_small.render(status, True, ACCENT_YELLOW if self.paused else GRAY)
        ctrl2 = self.font_small.render("Flechas Arr/Abj: Velocidad", True, GRAY)
        self.display.blit(ctrl1, (x_pad, y))
        self.display.blit(ctrl2, (x_pad, y + 18))

    def _update_ui(self):
        """Dibuja el tablero, la serpiente y la manzana."""
        self.display.fill(BG_COLOR)

        # Cuadrícula sutil de fondo
        for x in range(0, self.board_width, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (x, 0), (x, self.board_height))
        for y in range(0, self.board_height, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (0, y), (self.board_width, y))

        self._draw_food()
        self._draw_snake()
        self._draw_panel()

        pygame.display.flip()


# --- BLOQUE DE PRUEBA (Se ejecuta solo si lanzas este archivo directamente) ---
if __name__ == '__main__':
    game = SnakeGame(render=True, speed=12)
    game.set_training_info(generation=1, best_score=0, individual=1)
    game_over = False

    while not game_over:
        state = game.get_state()
        # Acción aleatoria para probar que todo funciona
        action = random.choice([0, 1, 2])
        game_over, score = game.step(action)

    print(f"Fin de la prueba | Manzanas: {score} | Fitness: {game.get_fitness()}")
    pygame.quit()