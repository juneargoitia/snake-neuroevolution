import os
import matplotlib.pyplot as plt


def plot_evolution(best_history, avg_history, save_path="results/fitness_evolution.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    generations = range(1, len(best_history) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_history, label="Mejor Fitness (Máximo)", color="#2ecc71", linewidth=2)
    plt.plot(generations, avg_history, label="Fitness Medio de la Población", color="#3498db", linestyle="--", linewidth=1.8)

    plt.title("Evolución del Rendimiento por Generación (Neuroevolución Snake)", fontsize=13, fontweight="bold")
    plt.xlabel("Generación", fontsize=11)
    plt.ylabel("Fitness (Puntuación)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Gráfica guardada correctamente en: {save_path}")