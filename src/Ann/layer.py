#Defining a fully connected layer for nueral networks 

import numpy as np
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Ann.activation import get_activation

# Guard rail to keep PSO from pushing the network into NaN/Inf territory.
# Values far outside this band deteriorate training anyway, so we clip inputs
# and weights to a large but finite range before each matmul.
_SAFE_CLIP_VALUE = 1e6


class Layer:
    
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        activation: str, 
        rng: Optional[np.random.Generator] = None
    ):
        
        self.input_size = input_size # number of input neurons
        self.output_size = output_size # number of output neurons
        self.activation_name = activation.lower() # name of activation function
        self.activation_fn = get_activation(self.activation_name)
        self.rng = rng if rng is not None else np.random.default_rng()

        # weight initialization based on activation function
        if self.activation_name == "relu":
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)

        # Initialize weights and biases
        self.W = self.rng.normal(
            loc=0.0, 
            scale=scale, 
            size=(output_size, input_size)
        ).astype(np.float64)
        
        self.b = np.zeros((output_size,), dtype=np.float64)

        
        """ Forward pass through the layer.
        
        Args:
           x: Input with shape [in_dim] or [batch, in_dim]
            
        Returns:
           Activated output with shape [out_dim] or [batch, out_dim] """
        
    def forward(self, x: np.ndarray) -> np.ndarray:
    
        # Handle both 1D and 2D inputs
        original_1d = False
        if x.ndim == 1:
            x = x[np.newaxis, :]  # [1, in_dim]
            original_1d = True

        # Linear transformation: z = x @ W^T + b
        safe_x = np.clip(
            np.nan_to_num(x, nan=0.0, posinf=_SAFE_CLIP_VALUE, neginf=-_SAFE_CLIP_VALUE),
            -_SAFE_CLIP_VALUE,
            _SAFE_CLIP_VALUE,
        )
        safe_W = np.clip(
            np.nan_to_num(
                self.W, nan=0.0, posinf=_SAFE_CLIP_VALUE, neginf=-_SAFE_CLIP_VALUE
            ),
            -_SAFE_CLIP_VALUE,
            _SAFE_CLIP_VALUE,
        )
        safe_b = np.clip(
            np.nan_to_num(
                self.b, nan=0.0, posinf=_SAFE_CLIP_VALUE, neginf=-_SAFE_CLIP_VALUE
            ),
            -_SAFE_CLIP_VALUE,
            _SAFE_CLIP_VALUE,
        )

        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            z = safe_x @ safe_W.T + safe_b  # [batch, out_dim]

        # Apply activation function
        a = self.activation_fn(z)

        # Return original shape if input was 1D
        return a.squeeze(0) if original_1d else a

    def __repr__(self) -> str:  #String representation of the layer.
        return (
            f"Layer(in={self.input_size}, out={self.output_size}, "
            f"act='{self.activation_name}')"
        )
