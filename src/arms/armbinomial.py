"""
Module: arms/armbinomial.py
Description: Contains the implementation of the ArmBinomial class for the normal distribution arm.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from arms import Arm
import numpy as np


class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        assert n > 0, "El parámetro n debe ser mayor que 0."
        assert 0 <= p <= 1, "La probabilidad p debe estar entre 0 y 1."
        self.n = n
        self.p = p

    def pull(self):
        return np.random.binomial(self.n, self.p)

    def get_expected_value(self) -> float:
        return self.n * self.p

    @classmethod
    def generate_arms(cls, k: int, n: int = 10):
        ps = np.random.uniform(0, 1, k)
        return [cls(n, p) for p in ps]

    def __str__(self):
        return f"ArmBinomial(n={self.n}, p={self.p:.2f})"
