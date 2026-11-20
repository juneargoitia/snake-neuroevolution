import numpy as np
import os
from neural_network import NeuralNetwork


class GeneticAlgorithm:
    def __init__(
        self,
        population_size=100,
        elite_fraction=0.2,
        mutation_rate=0.05,
        mutation_strength=0.2,
        input_size=12,
        hidden_size=16,
        output_size=3,
    ):
        self.population_size = population_size
        self.elite_size = max(2, int(population_size * elite_fraction))
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.population = [
            NeuralNetwork(input_size, hidden_size, output_size)
            for _ in range(population_size)
        ]

        self.generation = 1
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_overall_fitness = -np.inf
        self.best_overall_weights = None

    def evolve(self, fitness_scores):
        fitness_scores = np.array(fitness_scores, dtype=float)

        max_fit = float(np.max(fitness_scores))
        avg_fit = float(np.mean(fitness_scores))
        self.best_fitness_history.append(max_fit)
        self.avg_fitness_history.append(avg_fit)

        best_idx = int(np.argmax(fitness_scores))
        if max_fit > self.best_overall_fitness:
            self.best_overall_fitness = max_fit
            self.best_overall_weights = self.population[best_idx].get_weights().copy()

        sorted_indices = np.argsort(fitness_scores)[::-1]
        elites = [self.population[i] for i in sorted_indices[: self.elite_size]]
        elite_chromosomes = [ind.get_weights() for ind in elites]
        new_population = []

        for weights in elite_chromosomes:
            child_net = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            child_net.set_weights(weights.copy())
            new_population.append(child_net)

        while len(new_population) < self.population_size:
            idx1, idx2 = np.random.choice(len(elite_chromosomes), size=2, replace=False)
            parent1 = elite_chromosomes[idx1]
            parent2 = elite_chromosomes[idx2]

            child_weights = self._crossover(parent1, parent2)

            child_weights = self._mutate(child_weights)

            child_net = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            child_net.set_weights(child_weights)
            new_population.append(child_net)

        self.population = new_population
        self.generation += 1

        return max_fit, avg_fit

    def _crossover(self, parent1, parent2):
        mask = np.random.rand(len(parent1)) < 0.5
        return np.where(mask, parent1, parent2)

    def _mutate(self, weights):
        mutation_mask = np.random.rand(len(weights)) < self.mutation_rate
        noise = np.random.randn(len(weights)) * self.mutation_strength
        weights[mutation_mask] += noise[mutation_mask]
        return weights

    def save_best_model(self, filepath="models/best_brain.npy"):
        if self.best_overall_weights is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.save(filepath, self.best_overall_weights)

    def load_model(self, filepath="models/best_brain.npy"):
        weights = np.load(filepath)
        net = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        net.set_weights(weights)
        return net