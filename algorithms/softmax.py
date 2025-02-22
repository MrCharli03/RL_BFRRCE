"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo softmax para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from algorithms.algorithm import Algorithm
import numpy as np

# Aquí básicamente seleccionamos cada brazo con una probabilidad proporcional a su valor estimado. La temperatura (tau) nos indica: 
# A más alta -> más exploración (todas las probabilidades son casi iguales)
# A más baja -> más explotación (brazo con mejor valor estimado tiene probabilidad muy alta).
# Tau cercano a 0 -> Se comporta casi como greedy, selecciona casi siempre al mejor


class Softmax(Algorithm):
    def __init__(self, k: int, tau: float = 1.0):
        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int: 
        exp_values = np.exp(self.values / self.tau) # Elevamos a e los valores estimados de cada brazo (self.values) y dividimos por tau
        probabilities = exp_values / np.sum(exp_values) # Se normalizan las probabilidades
        return np.random.choice(self.k, p=probabilities) # Se elige un brazo al azar, pero con sesgo hacia los de mayor probabilidad