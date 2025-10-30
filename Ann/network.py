# Neural Network : Defines a multi-layer feed-forward neural network.
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
        
        # Validation
        if len(architecture) < 2: # architecture must have at least input and output sizes 
            raise ValueError(
                "Architecture must have at least input and output sizes "
                "e.g [8, 1]"
            )
        if len(activations) != len(architecture) - 1:
            raise ValueError(
                f"Need one activation per layer except input. "
                f"Got {len(activations)} activations for "
                f"{len(architecture)-1} layers."
            )

        self.architecture = architecture
        self.activations = [a.lower() for a in activations]
        self.rng = np.random.default_rng(seed)

        # Build layers
        self.layers: List[Layer] = []
        for in_size, out_size, act in zip(
            architecture[:-1], 
            architecture[1:], 
            self.activations
        ):
            self.layers.append(
                Layer(in_size, out_size, act, rng=self.rng)
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
    
        # Forward propagation through all layers.
        
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def summary(self) -> None: # prints a summary of the network architechture
        print("=" * 50)
        print("Neural Network Summary")
        print("=" * 50)
        for idx, layer in enumerate(self.layers, start=1):
            print(f"  Layer {idx}: {layer}")
        print("=" * 50)
        print(f"Total layers (excluding input): {len(self.layers)}")
        print("=" * 50)
    
    def get_num_parameters(self) -> int: # Calculate total number of trainable parameters.
        
        total = 0
        for layer in self.layers:
            # Weights + biases
            total += layer.W.size + layer.b.size
        return total