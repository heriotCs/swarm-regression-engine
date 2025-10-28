import numpy as np
from typing import Tuple
from Ann.network import NeuralNetwork


class ANNFitness: # Fitness function that evaluates an ANN by unpacking PSO particle positions into network weights and computing error on training data.
    
    # Initialize ANN fitness evaluator
    def __init__(
        self,
        network: NeuralNetwork, # network: NeuralNetwork instance , we already defined the architecture 
        X_train: np.ndarray, # X_train: Training features [n_samples, n_features]
        y_train: np.ndarray, # y_train: Training targets [n_samples] or [n_samples, 1]
        metric: str = "mse" # metric: Error metric to use ('mse', 'mae', 'rmse', etc), here we are using mean square error
    ):
        
        self.network = network
        self.X_train = X_train
        self.y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        self.metric = metric.lower()
        
        # Calculate total number of parameters 
        self.num_params = self.network.get_num_parameters()
        
        # Store layer shapes for unpacking weights
        self.layer_shapes = []
        for layer in network.layers:
            w_shape = layer.W.shape
            b_shape = layer.b.shape
            self.layer_shapes.append((w_shape, b_shape))


    # decode PSO position vector into network weights and biases
    def decode_weights(self, position: np.ndarray) -> None:
        
        idx = 0
        for layer, (w_shape, b_shape) in zip(self.network.layers, self.layer_shapes):
            # Extract and reshape weights
            w_size = np.prod(w_shape)
            layer.W = position[idx:idx + w_size].reshape(w_shape)
            idx += w_size
            
            # Extract biases
            b_size = np.prod(b_shape)
            layer.b = position[idx:idx + b_size].reshape(b_shape)
            idx += b_size
    
    # encode current network weights into a flat position vector.
    def encode_weights(self) -> np.ndarray:
    
        params = []
        for layer in self.network.layers:
            params.append(layer.W.flatten())
            params.append(layer.b.flatten())
        return np.concatenate(params)
    
    # evalute fitness for specific netwoek weights
    def evaluate(self, position: np.ndarray) -> float:

        # Set network weights from position
        self.decode_weights(position)
        
        # Forward pass on all training data
        predictions = self.network.forward(self.X_train)
        
        # Ensure predictions have correct shape
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Calculate error based on metric
        errors = predictions - self.y_train
        
        if self.metric == "mse":
            return float(np.mean(errors ** 2))
        elif self.metric == "rmse":
            return float(np.sqrt(np.mean(errors ** 2)))
        elif self.metric == "mae":
            return float(np.mean(np.abs(errors)))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    # get parameter bounds for PSO
    def get_bounds(self, weight_range: float = 5.0) -> list:
        
        return [(-weight_range, weight_range)] * self.num_params # List of (min, max) tuples for each parameter
    
    # direct calling of the fitness function
    def __call__(self, position: np.ndarray) -> float:
        return self.evaluate(position)

# extended fitness function that also tracks validation performance
class ANNFitnessWithValidation(ANNFitness):
    
    # Initialize with both training and validation data
    def __init__(
        self,
        network: NeuralNetwork,
        X_train: np.ndarray, # training features 
        y_train: np.ndarray, # training target s
        X_val: np.ndarray, # validation features 
        y_val: np.ndarray, # validation targets 
        metric: str = "mse"
    ):
        
        super().__init__(network, X_train, y_train, metric)
        self.X_val = X_val
        self.y_val = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
        
        # Track best validation score
        self.best_val_error = np.inf
        self.best_weights = None
    
    # Evaluate on training data but also track validation performance
    def evaluate(self, position: np.ndarray) -> float:
        # Get training error
        train_error = super().evaluate(position)
        
        # Also compute validation error (but don't use it for PSO)
        val_predictions = self.network.forward(self.X_val)
        if val_predictions.ndim == 1:
            val_predictions = val_predictions.reshape(-1, 1)
        
        val_errors = val_predictions - self.y_val
        
        if self.metric == "mse":
            val_error = float(np.mean(val_errors ** 2))
        elif self.metric == "rmse":
            val_error = float(np.sqrt(np.mean(val_errors ** 2)))
        elif self.metric == "mae":
            val_error = float(np.mean(np.abs(val_errors)))
        else:
            val_error = train_error
        
        # Track best validation weights
        if val_error < self.best_val_error:
            self.best_val_error = val_error
            self.best_weights = position.copy()
        
        return train_error
    
    # get the weights that achieved best validation error
    def get_best_validation_weights(self) -> Tuple[np.ndarray, float]:
        
        if self.best_weights is None:
            raise RuntimeError("No weights evaluated yet")
        return self.best_weights.copy(), self.best_val_error # Tuple of (best_weights, best_validation_error)