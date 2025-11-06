# Fitness.py
# This small wrapper class lets us plug different objective functions into the PSO.
# In our coursework, the main objective is the ANN's MSE, but the wrapper keeps it general.

from typing import Callable
import numpy as np

class Fitness:
    """
    Simple container for any objective function of the form f(x) -> scalar.
    PSO will call this to evaluate each particle's position.
    Lower fitness values correspond to better solutions unless 'minimize' is set to False.
    """
    def __init__(self, objective_func: Callable[[np.ndarray], float], minimize: bool = True):
        # Store the target objective function (e.g., ANN forward pass + MSE calculation)
        self.objective_func = objective_func
        
        # Allows switching between minimisation and maximisation if needed
        self.minimize = minimize

    def evaluate(self, position: np.ndarray) -> float:
        """
        Compute the fitness of a particle at the given position.
        Position is a flattened vector of ANN parameters in our main experiments.
        """
        value = float(self.objective_func(position))
        
        # If we are maximising instead of minimising, simply negate the score.
        return value if self.minimize else -value
