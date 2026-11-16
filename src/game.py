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
BLACK = (20, 20, 20)
WHITE = (255, 255, 255)
RED = (220, 50, 50)
GREEN_HEAD = (50, 220, 100)
GREEN_BODY = (30, 170, 70)
GRID_COLOR = (35, 35, 35)

BLOCK_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20

class SnakeGame:
    def __init__(self, render=True, speed=15):
        self.render = render
        self.speed = speed
        self.width = GRID_WIDTH * BLOCK_SIZE
        self.height = GRID_HEIGHT * BLOCK_SIZE

        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake - Neuroevolution')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('arial', 18)

        self.reset()

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
        if pt.x >= self.width or pt.x < 0 or pt.y >= self.height or pt.y < 0:
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

    def step(self, action):
        """
        Ejecuta un paso del juego dada una acción de la IA:
        action: 0 = recto, 1 = girar derecha, 2 = girar izquierda
        Devuelve: (game_over, score)
        """
        self.steps += 1
        self.steps_without_food += 1

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

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

    def _update_ui(self):
        """Dibuja el tablero, la serpiente y la manzana."""
        self.display.fill(BLACK)

        # Cuadrícula sutil de fondo
        for x in range(0, self.width, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (0, y), (self.width, y))

        # Dibujar serpiente
        for i, pt in enumerate(self.snake):
            color = GREEN_HEAD if i == 0 else GREEN_BODY
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLACK, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE), 1)

        # Dibujar manzana
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Marcador
        text = self.font.render(f"Score: {self.score} | Steps: {self.steps}", True, WHITE)
        self.display.blit(text, [5, 5])
        pygame.display.flip()


# --- BLOQUE DE PRUEBA (Se ejecuta solo si lanzas este archivo directamente) ---
if __name__ == '__main__':
    game = SnakeGame(render=True, speed=10)
    game_over = False

    while not game_over:
        state = game.get_state()
        # Acción aleatoria para probar que todo funciona
        action = random.choice([0, 1, 2])
        game_over, score = game.step(action)

    print(f"Fin de la prueba | Manzanas: {score} | Fitness: {game.get_fitness()}")
    pygame.quit()