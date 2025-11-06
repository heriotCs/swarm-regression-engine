import numpy as np
from typing import Tuple
import sys
import os

# Add parent directory to path so we can import from Ann and pso folders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Ann.network import NeuralNetwork


class ANNFitness:
    """
    Fitness function used to evaluate how well a given ANN performs.
    PSO supplies a particle position (a long 1-D vector containing all
    weights and biases), and this class unpacks those values into the
    network before computing the error on the training set.
    """

    def __init__(
        self,
        network: NeuralNetwork,         # The ANN whose parameters are being optimised
        X_train: np.ndarray,            # Training features
        y_train: np.ndarray,            # Training targets
        metric: str = "mse"             # Error metric used for fitness
    ):
        self.network = network
        self.X_train = X_train

        # Keep targets as a column vector for consistency during subtraction
        self.y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train

        self.metric = metric.lower()

        # Total number of trainable parameters (used for PSO particle size)
        self.num_params = self.network.get_num_parameters()

        # Cache shapes of every layer's weight/bias matrices
        # This allows us to reconstruct them easily from the flat particle vector
        self.layer_shapes = []
        for layer in network.layers:
            w_shape = layer.W.shape
            b_shape = layer.b.shape
            self.layer_shapes.append((w_shape, b_shape))

    def decode_weights(self, position: np.ndarray) -> None:
        """
        Takes a flat PSO position vector and reshapes it back into the
        corresponding weight and bias matrices inside the ANN.
        """
        idx = 0
        for layer, (w_shape, b_shape) in zip(self.network.layers, self.layer_shapes):

            # Extract and reshape weights
            w_size = np.prod(w_shape)
            layer.W = position[idx:idx + w_size].reshape(w_shape)
            idx += w_size

            # Extract and reshape biases
            b_size = np.prod(b_shape)
            layer.b = position[idx:idx + b_size].reshape(b_shape)
            idx += b_size

    def encode_weights(self) -> np.ndarray:
        """
        Opposite of decode_weights(). Flattens the network’s current weights
        into a single vector so PSO can use it as a particle.
        """
        params = []
        for layer in self.network.layers:
            params.append(layer.W.flatten())
            params.append(layer.b.flatten())
        return np.concatenate(params)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Core fitness function used by PSO.
        - Applies the particle's weights to the ANN
        - Runs a forward pass on all training examples
        - Computes the chosen error metric
        """
        # Load PSO particle into network
        self.decode_weights(position)

        # Forward pass
        predictions = self.network.forward(self.X_train)

        # Keep consistent shape
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        # Compute residuals
        errors = predictions - self.y_train

        # Return selected metric
        if self.metric == "mse":
            return float(np.mean(errors ** 2))
        elif self.metric == "rmse":
            return float(np.sqrt(np.mean(errors ** 2)))
        elif self.metric == "mae":
            return float(np.mean(np.abs(errors)))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def get_bounds(self, weight_range: float = 5.0) -> list:
        """
        Provides PSO with parameter bounds. Every ANN weight/bias is
        allowed to vary between [-weight_range, +weight_range].
        """
        return [(-weight_range, weight_range)] * self.num_params

    def __call__(self, position: np.ndarray) -> float:
        """
        Allows the object to be called directly, e.g. fit(pos).
        """
        return self.evaluate(position)


class ANNFitnessWithValidation(ANNFitness):
    """
    Extension of the basic fitness class that also tracks validation error.
    PSO still optimises training error, but we record the best validation
    score to analyse generalisation.
    """

    def __init__(
        self,
        network: NeuralNetwork,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "mse"
    ):
        # Reuse parent initialisation for training data
        super().__init__(network, X_train, y_train, metric)

        # Store validation set
        self.X_val = X_val
        self.y_val = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val

        # Track best validation results
        self.best_val_error = np.inf
        self.best_weights = None

    def evaluate(self, position: np.ndarray) -> float:
        """
        Computes training error (used by PSO) and also validation error
        (only recorded for analysis).
        """
        # Compute training error using parent method
        train_error = super().evaluate(position)

        # Compute validation predictions
        val_preds = self.network.forward(self.X_val)
        if val_preds.ndim == 1:
            val_preds = val_preds.reshape(-1, 1)

        # Compute validation error
        val_errors = val_preds - self.y_val
        if self.metric == "mse":
            val_error = float(np.mean(val_errors ** 2))
        elif self.metric == "rmse":
            val_error = float(np.sqrt(np.mean(val_errors ** 2)))
        elif self.metric == "mae":
            val_error = float(np.mean(np.abs(val_errors)))
        else:
            val_error = train_error  # Fallback

        # Update record of best validation weights
        if val_error < self.best_val_error:
            self.best_val_error = val_error
            self.best_weights = position.copy()

        return train_error

    def get_best_validation_weights(self) -> Tuple[np.ndarray, float]:
        """
        Returns the parameter vector that achieved the lowest validation error.
        Useful for post-training inspection.
        """
        if self.best_weights is None:
            raise RuntimeError("No validation evaluations have been performed yet.")
        return self.best_weights.copy(), self.best_val_error