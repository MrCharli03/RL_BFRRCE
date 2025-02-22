from algorithms.algorithm import Algorithm
import numpy as np
import math


class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 0.5):
        super().__init__(k)
        self.alpha = alpha
        self.t = 0
        self.ka = np.zeros(k, dtype=int)  # Contador de épocas por brazo
        self.tau = np.ones(k, dtype=int)  # Tamaño de época por brazo (Empezar en 1 para evitar log(0) y divisiones por cero)

    def select_arm(self) -> int:
        self.t += 1

        for i in range(self.k):
            if self.counts[i] == 0:  # Explorar cada brazo al menos una vez
                self.tau[i] = 1
                return i

        # Cálculo del valor UCB2 para cada brazo, según la fórmula de las diapositivas
        ucb_values = self.values + np.sqrt(
            (1 + self.alpha) * np.log(math.e * self.t / np.maximum(self.tau, 1)) / (2 * np.maximum(self.tau, 1))
        )

        # Evitar NaNs por si acaso
        ucb_values = np.nan_to_num(ucb_values, nan=-np.inf)

        arm = np.argmax(ucb_values)  # Elegimos el brazo con mayor valor UCB

        # Actualizamos la época si ya hemos elegido este brazo `tau` veces
        if self.counts[arm] >= self.tau[arm]:
            self.ka[arm] += 1
            self.tau[arm] = min(math.ceil((1 + self.alpha) ** self.ka[arm]), 10**6)  # Limitar tau para evitar overflow

        return arm
