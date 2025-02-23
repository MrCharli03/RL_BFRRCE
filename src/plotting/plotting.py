"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy, UCB1, UCB2, Softmax, GradientBandit


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__

    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, UCB1):
        label += f" (c={algo.c})"
    elif isinstance(algo, UCB2):
        label += f" (alpha={algo.alpha})"
    elif isinstance(algo, Softmax):
        label += f" (tau={algo.tau})"
    elif isinstance(algo, GradientBandit):
        label += f" (alpha={algo.alpha})"
    else:
        raise ValueError(f"El algoritmo debe ser de la clase Algorithm o una subclase. Recibido: {type(algo).__name__}")

    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo')
    plt.ylabel('Porcentaje de Selecciones Óptimas')
    plt.title('Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo')
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(arm_stats: List[dict], algorithms: List[Algorithm]):
    """
    arm_stats: lista de diccionarios donde cada uno tiene por ejemplo:
    arm_stats[i] = {
        "mean_rewards": np.array con la media de recompensa por brazo,
        "selection_counts": np.array con el número de selecciones por brazo,
        "optimal_arm": índice del brazo óptimo
    }
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    for idx, algo in enumerate(algorithms):
        stats = arm_stats[idx]
        arms = np.arange(len(stats["mean_rewards"]))

        plt.figure(figsize=(12, 6))
        bars = plt.bar(arms, stats["mean_rewards"], color='lightblue', alpha=0.7)
        plt.xlabel('Brazo')
        plt.ylabel('Promedio de Ganancias')
        plt.title(f'Estadísticas de los Brazos - {get_algorithm_label(algo)}')

        for i, (bar, count) in enumerate(zip(bars, stats["selection_counts"])):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height(), f'{count}', ha='center', va='bottom')

        # Marcar el brazo óptimo
        plt.bar(stats["optimal_arm"], stats["mean_rewards"][stats["optimal_arm"]], color='orange', alpha=0.9)

        plt.tight_layout()
        plt.show()


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm]):
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo')
    plt.ylabel('Rechazo Acumulado')
    plt.title('Evolución del Rechazo (Regret) Acumulado vs Pasos de Tiempo')
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()
