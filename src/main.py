import sys
from src.game import SnakeGame
from src.genetic_algorithm import GeneticAlgorithm
from src.plotter import plot_evolution


def train(generations=100, population_size=150):
    """Entrena la población sin renderizado (máxima velocidad) y guarda el mejor modelo."""
    ga = GeneticAlgorithm(
        population_size=population_size,
        elite_fraction=0.15,
        mutation_rate=0.08,
        mutation_strength=0.3,
    )
    game = SnakeGame(render=False)

    print(f"Iniciando entrenamiento: {generations} generaciones | Población: {population_size}")
    print("-" * 65)

    best_overall_apples = 0

    for gen in range(1, generations + 1):
        fitness_scores = []
        best_score_gen = 0

        for idx, brain in enumerate(ga.population, start=1):
            game.reset()
            game.set_training_info(generation=gen, best_score=best_overall_apples, individual=idx)
            game_over = False

            while not game_over:
                state = game.get_state()
                action = brain.predict(state)
                game_over, score = game.step(action)

            fitness = game.get_fitness()
            fitness_scores.append(fitness)
            if score > best_score_gen:
                best_score_gen = score

        if best_score_gen > best_overall_apples:
            best_overall_apples = best_score_gen

        max_fit, avg_fit = ga.evolve(fitness_scores)
        print(
            f"Gen {gen:03d}/{generations} | "
            f"Max Fitness: {max_fit:8.1f} | "
            f"Avg Fitness: {avg_fit:7.1f} | "
            f"Max Manzanas: {best_score_gen:2d} (Récord: {best_overall_apples})"
        )

    ga.save_best_model("models/best_brain.npy")
    plot_evolution(ga.best_fitness_history, ga.avg_fitness_history, "results/fitness_evolution.png")
    print("-" * 65)
    print(f"¡Entrenamiento finalizado! Récord manzanas: {best_overall_apples} | Fitness: {ga.best_overall_fitness:.1f}")


def watch(speed=18):
    """Carga el mejor cerebro guardado y muestra una partida en vivo con su panel de sensores."""
    ga = GeneticAlgorithm()
    best_brain = ga.load_model("models/best_brain.npy")

    game = SnakeGame(render=True, speed=speed)
    game.set_training_info(generation=100, best_score=0, individual=1)
    game_over = False

    while not game_over:
        state = game.get_state()
        action = best_brain.predict(state)
        game_over, score = game.step(action)

    print(f"Partida de exhibición terminada | Manzanas comidas: {score} | Fitness: {game.get_fitness()}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "watch":
        watch(speed=18)
    else:
        train(generations=100, population_size=150)