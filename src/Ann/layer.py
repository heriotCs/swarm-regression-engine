# Defining a fully connected layer for neural networks.
# This module holds the implementation of one dense layer, including
# weight initialisation, activation handling, and a numerically safe forward pass.

import numpy as np
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Ann.activation import get_activation

# Clip values to prevent PSO from generating extreme weights that
# would result in NaNs or overflow during matrix multiplications.
_SAFE_CLIP_VALUE = 1e6


class Layer:
    
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        activation: str, 
        rng: Optional[np.random.Generator] = None
    ):
        # Store basic layer dimensions
        self.input_size = input_size
        self.output_size = output_size

        # Normalise activation name and fetch function
        self.activation_name = activation.lower()
        self.activation_fn = get_activation(self.activation_name)

        # Use provided RNG or create our own (helps reproducibility)
        self.rng = rng if rng is not None else np.random.default_rng()

        # He-initialisation for ReLU, otherwise Xavier-like init
        if self.activation_name == "relu":
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)

        # Weight matrix initialised with normal distribution
        self.W = self.rng.normal(
            loc=0.0,
            scale=scale,
            size=(output_size, input_size)
        ).astype(np.float64)

        # Bias initialised to zeros
        self.b = np.zeros((output_size,), dtype=np.float64)

    # Forward pass
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute forward pass through this layer.

        Supports both 1D input vectors and batched 2D inputs.
        """

        # Convert 1D inputs into a batch of size 1
        original_1d = False
        if x.ndim == 1:
            x = x[np.newaxis, :]
            original_1d = True

        # Clip inputs and weights to avoid overflow (important when PSO
        # generates unusually large values during optimisation)
        safe_x = np.clip(
            np.nan_to_num(x, nan=0.0, posinf=_SAFE_CLIP_VALUE, neginf=-_SAFE_CLIP_VALUE),
            -_SAFE_CLIP_VALUE, _SAFE_CLIP_VALUE
        )
        safe_W = np.clip(
            np.nan_to_num(self.W, nan=0.0, posinf=_SAFE_CLIP_VALUE, neginf=-_SAFE_CLIP_VALUE),
            -_SAFE_CLIP_VALUE, _SAFE_CLIP_VALUE
        )
        safe_b = np.clip(
            np.nan_to_num(self.b, nan=0.0, posinf=_SAFE_CLIP_VALUE, neginf=-_SAFE_CLIP_VALUE),
            -_SAFE_CLIP_VALUE, _SAFE_CLIP_VALUE
        )

        # Perform the affine transformation z = xW^T + b
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            z = safe_x @ safe_W.T + safe_b

        # Apply the selected activation function
        a = self.activation_fn(z)

        # If original input was 1D, return a 1D output for convenience
        return a.squeeze(0) if original_1d else a

    # String representation for debugging/printing
    def __repr__(self) -> str:
        return (
            f"Layer(in={self.input_size}, out={self.output_size}, "
            f"act='{self.activation_name}')"
        )