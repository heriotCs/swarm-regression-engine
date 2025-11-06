# Neural Network: Defines a multi-layer feed-forward neural network.
# This class holds the overall ANN structure and handles layer creation,
# forward propagation, parameter counting, and simple printing utilities.

import numpy as np
from typing import List, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Ann.layer import Layer


class NeuralNetwork:

    def __init__(
        self, 
        architecture: List[int], 
        activations: List[str], 
        seed: Optional[int] = None
    ):
        """
        architecture: list of layer sizes, e.g. [8, 10, 1]
        activations: list of activation names for each layer except input
        """

        # Basic sanity checks to prevent misconfigured networks
        if len(architecture) < 2:
            raise ValueError(
                "Architecture must include at least input and output sizes "
                "(e.g. [8, 1])."
            )
        if len(activations) != len(architecture) - 1:
            raise ValueError(
                f"Expected {len(architecture)-1} activation functions, "
                f"but got {len(activations)}."
            )

        self.architecture = architecture
        self.activations = [a.lower() for a in activations]

        # RNG is stored so all layers share the same seed if provided
        self.rng = np.random.default_rng(seed)

        # -------------------------------------------------------------
        # Build each layer using the Layer class
        # -------------------------------------------------------------
        self.layers: List[Layer] = []
        for in_size, out_size, act in zip(
            architecture[:-1],
            architecture[1:],
            self.activations
        ):
            self.layers.append(
                Layer(in_size, out_size, act, rng=self.rng)
            )

    # -------------------------------------------------------------
    # Forward pass through the entire network
    # -------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply each layer sequentially. The Layer class handles
        activation functions and numerical clipping internally.
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # Allow the network to be called like a function: nn(x)
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    # -------------------------------------------------------------
    # Simple printed summary for debugging/inspection
    # -------------------------------------------------------------
    def summary(self) -> None:
        print("=" * 50)
        print("Neural Network Summary")
        print("=" * 50)
        for idx, layer in enumerate(self.layers, start=1):
            print(f"  Layer {idx}: {layer}")
        print("=" * 50)
        print(f"Total layers (excluding input): {len(self.layers)}")
        print("=" * 50)

    # -------------------------------------------------------------
    # Count total trainable parameters
    # -------------------------------------------------------------
    def get_num_parameters(self) -> int:
        """
        Returns the total number of parameters (weights + biases)
        which is what PSO optimises over.
        """
        total = 0
        for layer in self.layers:
            total += layer.W.size + layer.b.size
        return total