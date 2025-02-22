"""
Module: arms/armbernouilli.py
Description: Contains the implementation of the ArmBernouilli class for the normal distribution arm.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


from arms import Arm
import numpy as np


class ArmBernoulli(Arm):
    def __init__(self, p: float):
        assert 0 <= p <= 1, "La probabilidad p debe estar entre 0 y 1."
        self.p = p

    def pull(self):
        return np.random.binomial(1, self.p)

    def get_expected_value(self) -> float:
        return self.p

    @classmethod
    def generate_arms(cls, k: int):
        ps = np.random.uniform(0, 1, k)
        return [cls(p) for p in ps]

    def __str__(self):
        return f"ArmBernoulli(p={self.p:.2f})"
