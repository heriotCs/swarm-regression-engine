from typing import List, Optional

import sys
import os

# parent directory to Python path so we can import pso as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network import NeuralNetwork


# Builder class for creating neural network architectures easily.
class ANNBuilder:
   
    @staticmethod
    def build_regression_network(
        input_size: int,
        hidden_layers: List[int],
        seed: Optional[int] = None
    ) -> NeuralNetwork:
        
        # maunally initialising an example for regression tasks
        architecture = [input_size] + hidden_layers + [1]
        activations = ['relu'] * len(hidden_layers) + ['linear']
        
        return NeuralNetwork(architecture, activations, seed)
    
    @staticmethod  # Build a network for binary classification.
    def build_binary_classification_network(
        input_size: int,
        hidden_layers: List[int],
        seed: Optional[int] = None
    ) -> NeuralNetwork:
        
        architecture = [input_size] + hidden_layers + [1]
        activations = ['relu'] * len(hidden_layers) + ['sigmoid']
        
        return NeuralNetwork(architecture, activations, seed)
    
    @staticmethod # custom network with specified architecture and activations.
    def build_custom_network(
        architecture: List[int],
        activations: List[str],
        seed: Optional[int] = None
    ) -> NeuralNetwork:
        
        return NeuralNetwork(architecture, activations, seed)
    
    @staticmethod # deep network with uniform hidden layer sizes.
    def build_deep_network(
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_activation: str = 'linear',
        seed: Optional[int] = None
    ) -> NeuralNetwork:
        
        architecture = [input_size] + [hidden_size] * num_hidden_layers + [1]
        activations = ['relu'] * num_hidden_layers + [output_activation]
        
        return NeuralNetwork(architecture, activations, seed)


# Convenience functions for quick network creation 

# Create a simple regression network
def create_simple_regression_nn(
    input_size: int, 
    hidden_size: int = 10, 
    seed: Optional[int] = None
) -> NeuralNetwork:
    
    return ANNBuilder.build_regression_network(input_size, [hidden_size], seed)

# create a deep regression network with multiple hidden layers. with default no of hidden layers set to 3
def create_deep_regression_nn(
    input_size: int,
    num_layers: int = 3,
    hidden_size: int = 20,
    seed: Optional[int] = None
) -> NeuralNetwork:
    
    return ANNBuilder.build_deep_network(
        input_size, 
        hidden_size, 
        num_layers, 
        'linear', 
        seed
    )