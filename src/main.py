import random
import sys

import numpy as np

from src.game import SnakeGame
from src.genetic_algorithm import GeneticAlgorithm
from src.neural_network import NeuralNetwork
from src.plotter import plot_evolution


def play(brain, game, seed):
    random.seed(seed)
    game.reset()
    game_over = False
    while not game_over:
        action = brain.predict(game.get_state())
        game_over, score = game.step(action)
    return score, game.get_fitness()


def evaluate(brain, game, seeds):
    results = [play(brain, game, s) for s in seeds]
    return float(np.mean([r[1] for r in results])), float(np.mean([r[0] for r in results]))


def train(generations=100, population_size=150, games_per_individual=3, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    ga = GeneticAlgorithm(
        population_size=population_size,
        elite_fraction=0.15,
        mutation_rate=0.08,
        mutation_strength=0.3,
    )
    game = SnakeGame(render=False)

    print(f"Iniciando entrenamiento: {generations} generaciones | Población: {population_size} "
          f"| Partidas por individuo: {games_per_individual}")
    print("-" * 70)

    best_overall_apples = 0

    for gen in range(1, generations + 1):
        # Mismas manzanas para todos los individuos de la generación: comparación justa
        gen_seeds = [seed * 100_000 + gen * 100 + k for k in range(games_per_individual)]

        fitness_scores = []
        apples_scores = []
        for idx, brain in enumerate(ga.population, start=1):
            game.set_training_info(generation=gen, best_score=best_overall_apples, individual=idx)
            fit, apples = evaluate(brain, game, gen_seeds)
            fitness_scores.append(fit)
            apples_scores.append(apples)

        best_apples_gen = max(apples_scores)
        best_overall_apples = max(best_overall_apples, best_apples_gen)

        max_fit, avg_fit = ga.evolve(fitness_scores)
        print(
            f"Gen {gen:03d}/{generations} | "
            f"Max Fitness: {max_fit:9.1f} | "
            f"Avg Fitness: {avg_fit:8.1f} | "
            f"Manzanas (media mejor): {best_apples_gen:5.1f} | Récord: {best_overall_apples:.1f}"
        )

    final_seeds = [seed * 100_000 + 99_000 + k for k in range(20)]
    candidates = list(ga.population[: ga.elite_size])
    if ga.best_overall_weights is not None:
        hall = NeuralNetwork(ga.input_size, ga.hidden_size, ga.output_size)
        hall.set_weights(ga.best_overall_weights.copy())
        candidates.append(hall)

    scored = [(evaluate(c, game, final_seeds), c) for c in candidates]
    (best_fit, best_apples), best_brain = max(scored, key=lambda t: t[0][0])

    ga.best_overall_weights = best_brain.get_weights().copy()
    ga.save_best_model("models/best_brain.npy")
    plot_evolution(ga.best_fitness_history, ga.avg_fitness_history, "results/fitness_evolution.png")
    print("-" * 70)
    print(f"¡Entrenamiento finalizado! Mejor modelo: {best_apples:.1f} manzanas de media "
          f"en 20 partidas nuevas | Fitness medio: {best_fit:.1f}")


def watch(speed=18, games=5):
    """Carga el mejor cerebro guardado y muestra partidas en vivo con su panel de sensores."""
    ga = GeneticAlgorithm()
    best_brain = ga.load_model("models/best_brain.npy")

    game = SnakeGame(render=True, speed=speed)

    for n in range(1, games + 1):
        game.reset()
        game.set_training_info(generation=0, best_score=game.best_score, individual=n)
        game_over = False
        while not game_over:
            action = best_brain.predict(game.get_state())
            game_over, score = game.step(action)
        print(f"Partida {n}/{games} | Manzanas: {score} | Pasos: {game.steps} | "
              f"Pasos/manzana: {game.steps / max(score, 1):.1f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "watch":
        watch(speed=18)
    else:
        train(generations=100, population_size=150)