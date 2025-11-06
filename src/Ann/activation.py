# Activation Functions 
# This file defines the activation functions used by the ANN. Keeping them here makes it easy
# to switch between functions or add new ones later.

import numpy as np
from typing import Callable

import sys
import os

# Add parent directory so the ANN can access the PSO package if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sigmoid activation with clipping to avoid overflow in exp().
# The clip doesn't change behaviour on normal inputs but prevents numerical issues.
def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clip))

# ReLU activation (simple and commonly used).
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

# Hyperbolic tangent activation.
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

# Linear activation – used for the output layer in regression.
def linear(x: np.ndarray) -> np.ndarray:
    return x


# Map activation names to their corresponding functions.
# Having a dictionary like this avoids long if/else chains.
ACTIVATIONS = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "linear": linear,
}

# Return the activation function based on its name.
# This also ensures invalid function names are caught early.
def get_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from {list(ACTIVATIONS.keys())}."
        )
    return ACTIVATIONS[name]