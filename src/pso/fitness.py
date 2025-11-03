# Fitness.py

# Minimal fitness wrapper so we can plug different objectives (e.g., Sphere, ANN).
from typing import Callable
import numpy as np

class Fitness:
    """
    Wraps an objective function: f(x) -> scalar. Lower is better.
    """
    def __init__(self, objective_func: Callable[[np.ndarray], float], minimize: bool = True):
        self.objective_func = objective_func
        self.minimize = minimize

    def evaluate(self, position: np.ndarray) -> float:
        value = float(self.objective_func(position))
        return value if self.minimize else -value
