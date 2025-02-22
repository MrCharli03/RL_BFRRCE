"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo ucb1 para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from algorithms.algorithm import Algorithm
import numpy as np


class UCB1(Algorithm):
    def __init__(self, k: int, c: float = 1):
        super().__init__(k)
        self.t = 0  # Paso actual
        self.c = c  # Parámetro para ajustar exploración

    def select_arm(self) -> int:
        self.t += 1
        for i in range(self.k):
            if self.counts[i] == 0: # Explorar cada brazo al menos una vez
                return i  

        # Cálculo del valor UCB1 para cada brazo, según fórmula de las diapositivas
        ucb_values = self.values + self.c * np.sqrt(2 * np.log(self.t) / self.counts)

        # Seleccionamos brazo con mayor valor UCB
        return np.argmax(ucb_values)
