from typing import List, Optional
import sys
import os

# Add parent directory to path so we can import from pso or Ann modules easily
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .network import NeuralNetwork


class ANNBuilder:
    """
    Builder class for creating neural network architectures easily.
    Now supports optional dynamic activation functions for bonus experiments.
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
        - hidden_layers: list specifying number of neurons per hidden layer
        - activations: optional list of activation functions for each hidden layer
                       (e.g., ['relu', 'tanh']); last layer is always linear
        - seed: random seed for reproducibility
        """
        # Default activations: ReLU for hidden layers, linear for output
        if activations is None:
            activations = ['relu'] * len(hidden_layers) + ['linear']
        else:
            # Always ensure output layer is linear for regression
            activations = activations + ['linear']

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
        Build a network for binary classification with sigmoid output.
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
        Build a fully custom network with arbitrary architecture and activations.
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
        Build a deep network with a uniform hidden size and specified depth.
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
    Quickly create a simple one-hidden-layer regression network.
    """
    return ANNBuilder.build_regression_network(input_size, [hidden_size], seed=seed)


def create_deep_regression_nn(
    input_size: int,
    num_layers: int = 3,
    hidden_size: int = 20,
    seed: Optional[int] = None
) -> NeuralNetwork:
    """
    Quickly create a deeper regression network with uniform layer sizes.
    """
    return ANNBuilder.build_deep_network(
        input_size,
        hidden_size,
        num_layers,
        'linear',
        seed
    )
