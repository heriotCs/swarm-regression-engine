from typing import List, Optional
import sys
import os

# Add parent directory to the path so the ANN modules can import PSO components if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .network import NeuralNetwork


class ANNBuilder:
    """
    Helper class to build different neural network architectures.
    Keeps all network construction logic in one place so the main scripts
    don’t need to manually assemble layer sizes or activation lists.
    """

    # -------------------------------------------------------------
    # Regression network (supports custom activations)
    # -------------------------------------------------------------
    @staticmethod
    def build_regression_network(
        input_size: int,
        hidden_layers: List[int],
        activations: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> NeuralNetwork:
        """
        Build a feedforward neural network for regression tasks.

        Parameters:
          - input_size: number of input features
          - hidden_layers: list describing the width of each hidden layer
          - activations: optional list of activation names for each hidden layer
                         If not provided, defaults to ReLU for hidden layers.
          - seed: optional random seed for reproducibility
        """

        # If no activation list is provided, use ReLU for hidden layers
        # and enforce linear output (standard for regression tasks).
        if activations is None:
            activations = ['relu'] * len(hidden_layers) + ['linear']
        else:
            # Regardless of what is provided, regression output should stay linear.
            activations = activations + ['linear']

        # Build architecture vector: [input, hidden..., output]
        architecture = [input_size] + hidden_layers + [1]
        return NeuralNetwork(architecture, activations, seed)

    # -------------------------------------------------------------
    # Binary classification network
    # -------------------------------------------------------------
    @staticmethod
    def build_binary_classification_network(
        input_size: int,
        hidden_layers: List[int],
        seed: Optional[int] = None
    ) -> NeuralNetwork:
        """
        Create a simple binary classifier.
        Assumes sigmoid output layer and ReLU hidden layers.
        """
        architecture = [input_size] + hidden_layers + [1]
        activations = ['relu'] * len(hidden_layers) + ['sigmoid']
        return NeuralNetwork(architecture, activations, seed)

    # -------------------------------------------------------------
    # Custom network (manual architecture + activations)
    # -------------------------------------------------------------
    @staticmethod
    def build_custom_network(
        architecture: List[int],
        activations: List[str],
        seed: Optional[int] = None
    ) -> NeuralNetwork:
        """
        Allow full manual control of both architecture and activations.
        Mainly used for experiments or alternative configurations.
        """
        return NeuralNetwork(architecture, activations, seed)

    # -------------------------------------------------------------
    # Deep network (uniform hidden size)
    # -------------------------------------------------------------
    @staticmethod
    def build_deep_network(
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_activation: str = 'linear',
        seed: Optional[int] = None
    ) -> NeuralNetwork:
        """
        Build a deeper network where every hidden layer has the same size.
        Useful when testing PSO performance on increasing depth.
        """
        architecture = [input_size] + [hidden_size] * num_hidden_layers + [1]
        activations = ['relu'] * num_hidden_layers + [output_activation]
        return NeuralNetwork(architecture, activations, seed)


# -------------------------------------------------------------
# Convenience functions for quick network creation
# -------------------------------------------------------------

def create_simple_regression_nn(
    input_size: int,
    hidden_size: int = 10,
    seed: Optional[int] = None
) -> NeuralNetwork:
    """
    Convenience helper for a one-hidden-layer regression model.
    Avoids repeating simple setups in experiments.
    """
    return ANNBuilder.build_regression_network(input_size, [hidden_size], seed=seed)


def create_deep_regression_nn(
    input_size: int,
    num_layers: int = 3,
    hidden_size: int = 20,
    seed: Optional[int] = None
) -> NeuralNetwork:
    """
    Create a deeper regression model with several hidden layers
    of the same width. Handy for testing how PSO scales with depth.
    """
    return ANNBuilder.build_deep_network(
        input_size,
        hidden_size,
        num_layers,
        'linear',
        seed
    )