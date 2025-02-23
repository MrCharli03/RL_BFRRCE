"""
Module: algorithms/gradient_bandit.py
Description: Implementación del algoritmo gradient_bandit para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from algorithms.algorithm import Algorithm
import numpy as np


class GradientBandit(Algorithm):
    def __init__(self, k: int, alpha: float = 0.1):
        super().__init__(k)
        self.alpha = alpha
        self.preferences = np.zeros(k) # Inicializamos las preferencias en 0
        self.avg_reward = 0 # Recompensa promedio acumulada
        self.t = 0 

    def select_arm(self) -> int:
        exp_preferences = np.exp(self.preferences)
        probabilities = exp_preferences / np.sum(exp_preferences) # Aquí calculamos la distribución softmax, que convierte las preferencias H(a) en probabilidades
        return np.random.choice(self.k, p=probabilities) # Elegimos un brazo aleatorio en base a esas probabilidades

    def update(self, chosen_arm: int, reward: float):
        self.t += 1
        self.avg_reward += (reward - self.avg_reward) / self.t # Actualizamos la recompensa promedio acumulada
        probabilities = np.exp(self.preferences) / np.sum(np.exp(self.preferences)) # Calculamos otra vez las probabilidades actuales con softmax

        for i in range(self.k): # Actualizamos preferencias usando gradientes
            if i == chosen_arm:
                self.preferences[i] += self.alpha * (reward - self.avg_reward) * (1 - probabilities[i]) # El brazo elegido sube su preferencia si da una recompensa mayor que el promedio, si no, baja
            else:
                self.preferences[i] -= self.alpha * (reward - self.avg_reward) * probabilities[i] # Los otros brazos suben un poco si el elegido fue malo, si no, bajan
