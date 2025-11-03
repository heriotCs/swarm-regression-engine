# Activation Functions 

import numpy as np
from typing import Callable

import sys
import os

# parent directory to Python path so we can import pso as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# define sigmoid activation function
def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clip))

#define relu
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

#define hyperbolic tangent activation function
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

#define linear identity activation function
def linear(x: np.ndarray) -> np.ndarray:
    return x


# store them under relu
ACTIVATIONS = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "linear": linear,
}

# return the activatin function callable
def get_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from {list(ACTIVATIONS.keys())}."
        )
    return ACTIVATIONS[name]