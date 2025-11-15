import numpy as np
from typing import Optional, Tuple, Dict
import sys
import os

# Add parent directory so imports work when running experiments
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Ann.network import NeuralNetwork
from AnnPso.annFitness import ANNFitness, ANNFitnessWithValidation
from pso.fitness import Fitness
from pso.pso import PSO
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Trainer class that connects the ANN and PSO components.
# Handles: creating the fitness wrapper, configuring PSO,
# running optimisation, and evaluating final performance.
class ANNPSOTrainer:

    def __init__(
        self,
        network: NeuralNetwork,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        metric: str = "mse"
    ):
        """
        Stores the dataset and decides whether to use the validation wrapper
        based on whether validation data is supplied.
        """

        self.network = network
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.metric = metric

        # Select correct fitness implementation (with or without validation tracking)
        if X_val is not None and y_val is not None:
            self.ann_fitness = ANNFitnessWithValidation(
                network, X_train, y_train, X_val, y_val, metric
            )
            self.use_validation = True
        else:
            self.ann_fitness = ANNFitness(
                network, X_train, y_train, metric
            )
            self.use_validation = False

        # Placeholder for PSO instance after construction
        self.pso: Optional[PSO] = None

        # Will store optimisation results for later inspection
        self.training_history: Dict = {}

    # Main PSO training loop for the ANN
    def train(
        self,
        swarm_size: int = 50,
        max_iters: int = 200,
        weight_range: float = 5.0,

        # Core PSO hyperparameters
        chi: float = 0.7298,
        c1: float = 1.5,
        c2: float = 1.5,
        c3: float = 0.0,

        topology: str = "random",
        num_informants: int = 3,
        refresh_rate: int = 7,

        # Additional extensions
        boundary_mode: str = "reflect",
        vmax: Optional[float] = None,
        stagnation_iters: Optional[int] = 60,

        # Optional linear schedules for parameters
        schedule_c1: Optional[Tuple[float, float]] = None,
        schedule_c2: Optional[Tuple[float, float]] = None,
        schedule_chi: Optional[Tuple[float, float]] = None,

        seed: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Runs PSO to optimise the ANN weights. Returns:
        - Best weight vector
        - Best training error
        """

        if verbose:
            print("=" * 60)
            print("Training Neural Network with PSO")
            print("=" * 60)
            self.network.summary()
            print(f"\nTotal parameters to optimise: {self.ann_fitness.num_params}")
            print(f"Training samples: {self.X_train.shape[0]}")
            if self.use_validation:
                print(f"Validation samples: {self.X_val.shape[0]}")
            print(f"Metric: {self.metric.upper()}")
            print("=" * 60)

        # Fitness wrapper that PSO expects
        fitness = Fitness(self.ann_fitness, minimize=True)

        # PSO parameter bounds (same range for every network parameter)
        bounds = self.ann_fitness.get_bounds(weight_range)

        # Create PSO optimiser
        self.pso = PSO(
            fitness=fitness,
            dim=self.ann_fitness.num_params,
            bounds=bounds,
            swarm_size=swarm_size,
            max_iters=max_iters,
            chi=chi,
            c1=c1,
            c2=c2,
            c3=c3,
            topology=topology,
            num_informants=num_informants,
            refresh_rate=refresh_rate,
            boundary_mode=boundary_mode,
            vmax=vmax,
            stagnation_iters=stagnation_iters,
            schedule_c1=schedule_c1,
            schedule_c2=schedule_c2,
            schedule_chi=schedule_chi,
            seed=seed,
            verbose=verbose
        )

        # Run PSO optimisation
        best_weights, best_fitness = self.pso.optimise()

        # Apply best weights to the network for final evaluation
        self.ann_fitness.decode_weights(best_weights)

        # Record results
        self.training_history = {
            'best_training_error': best_fitness,
            'final_weights': best_weights
        }

        # If validation is being tracked, store best validation metrics as well
        if self.use_validation:
            best_val_weights, best_val_error = \
                self.ann_fitness.get_best_validation_weights()

            self.training_history['best_validation_error'] = best_val_error
            self.training_history['best_val_weights'] = best_val_weights

            if verbose:
                print(f"\nBest validation error: {best_val_error:.6f}")
                print("Use .set_to_best_validation() to load these weights.")

        if verbose:
            print("\nTraining complete!")
            print(f"Final training error: {best_fitness:.6f}")
            print("=" * 60)

        return best_weights, best_fitness

    # Load the best validation weights back into the model
    def set_to_best_validation(self) -> None:
        if not self.use_validation:
            raise RuntimeError("No validation data was supplied during training.")

        best_val_weights = self.training_history['best_val_weights']
        self.ann_fitness.decode_weights(best_val_weights)
        print("Network weights restored to best validation performance.")


    # Compute evaluation metrics on any given dataset
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        return_predictions: bool = False
    ) -> Dict:
        """
        Computes several regression metrics (MSE, RMSE, MAE, R²)
        on the provided dataset.
        """

        preds = self.network.forward(X)

        # Ensure matching shapes
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        results = {
            'mse': float(mean_squared_error(y, preds)),
            'rmse': float(np.sqrt(mean_squared_error(y, preds))),
            'mae': float(mean_absolute_error(y, preds)),
            'r2': float(r2_score(y, preds))
        }

        if return_predictions:
            results['predictions'] = preds

        return results

    # Print metrics in a clean format
    def print_evaluation(self, X: np.ndarray, y: np.ndarray, dataset_name: str = "Test"):
        results = self.evaluate(X, y)

        print(f"\n{dataset_name} Set Evaluation:")
        print("-" * 40)
        print(f"  MSE:  {results['mse']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  MAE:  {results['mae']:.4f}")
        print(f"  R²:   {results['r2']:.4f}")
        print("-" * 40)


# Minimal example to ensure the trainer works independently
if __name__ == "__main__":
    # Dummy network for quick testing
    network = NeuralNetwork([8, 10, 1], activations=["relu", "linear"])

    # Generate random training + validation sets
    X_train = np.random.rand(100, 8)
    y_train = np.random.rand(100, 1)
    X_val = np.random.rand(20, 8)
    y_val = np.random.rand(20, 1)

    # Create trainer
    trainer = ANNPSOTrainer(
        network=network,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        metric="mse"
    )

    # Run atraining loop
    best_weights, best_fitness = trainer.train(swarm_size=10, max_iters=100)

    # Print evaluation on validation set
    trainer.print_evaluation(X_val, y_val, dataset_name="Validation")