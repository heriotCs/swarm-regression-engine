import numpy as np
from typing import Optional, Tuple, Dict
from network import NeuralNetwork
from annFitness import ANNFitness, ANNFitnessWithValidation
from pso.fitness import Fitness
from pso.pso import PSO
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Training neural netwroks using PSO
class ANNPSOTrainer:
    
    # initialise trainer with data
    def __init__(
        self,
        network: NeuralNetwork,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        metric: str = "mse"
    ):
        
        self.network = network
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.metric = metric
        
        # Create appropriate fitness function
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
        
        # Will store PSO optimizer
        self.pso: Optional[PSO] = None
        self.training_history: Dict = {}
    
    # train the neural network using PSO
    def train(
        
        self,
        swarm_size: int = 50, # number of particles in swarm
        max_iters: int = 200, # maximum PSO iterations
        weight_range: float = 5.0, # range for weight initialization
        
        # PSO hyperparameters
        chi: float = 0.7298,
        c1: float = 1.5,
        c2: float = 1.5,
        c3: float = 0.0,
        topology: str = "random",
        num_informants: int = 3, # number of informants per particle
        refresh_rate: int = 7, # ow often to refresh topology
        
        # Extensions
        boundary_mode: str = "reflect",
        vmax: Optional[float] = None,
        stagnation_iters: Optional[int] = 60,

        # Schedules
        schedule_c1: Optional[Tuple[float, float]] = None,
        schedule_c2: Optional[Tuple[float, float]] = None,
        schedule_chi: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None, # Random seed
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]: # return Tuple of (best_weights, best_fitness)
        
        if verbose:
            print("="*60)
            print("Training Neural Network with PSO")
            print("="*60)
            self.network.summary()
            print(f"\nTotal parameters to optimize: {self.ann_fitness.num_params}")
            print(f"Training samples: {self.X_train.shape[0]}")
            if self.use_validation:
                print(f"Validation samples: {self.X_val.shape[0]}")
            print(f"Metric: {self.metric.upper()}")
            print("="*60)
        
        # Create PSO fitness wrapper
        fitness = Fitness(self.ann_fitness, minimize=True)
        
        # Get parameter bounds
        bounds = self.ann_fitness.get_bounds(weight_range)
        
        # Create PSO optimizer
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
        
        # Run optimization
        best_weights, best_fitness = self.pso.optimise()
        
        # Set network to best weights
        self.ann_fitness.decode_weights(best_weights)
        
        # Store training history
        self.training_history = {
            'best_training_error': best_fitness,
            'final_weights': best_weights
        }
        
        if self.use_validation:
            best_val_weights, best_val_error = \
                self.ann_fitness.get_best_validation_weights()
            self.training_history['best_validation_error'] = best_val_error
            self.training_history['best_val_weights'] = best_val_weights
            
            if verbose:
                print(f"\nBest validation error: {best_val_error:.6f}")
                print("Use .set_to_best_validation() to load best validation weights")
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"Final training error: {best_fitness:.6f}")
            print("="*60)
        
        return best_weights, best_fitness
    
    # Load the weights that achieved best validation performance
    def set_to_best_validation(self) -> None:
        
        if not self.use_validation:
            raise RuntimeError("No validation data provided during initialization")
        
        best_val_weights = self.training_history['best_val_weights']
        self.ann_fitness.decode_weights(best_val_weights)
        print("Network weights set to best validation weights")
    
    # Evaluate network on given data
    def evaluate(
        self, 
        X: np.ndarray, # Features
        y: np.ndarray, # True targets
        return_predictions: bool = False # whether to return predictions
    ) -> Dict: # retuens dictionary of metrics
        
        predictions = self.network.forward(X)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        results = {
            'mse': float(mean_squared_error(y, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
            'mae': float(mean_absolute_error(y, predictions)),
            'r2': float(r2_score(y, predictions))
        }
        
        if return_predictions:
            results['predictions'] = predictions
        
        return results
    
    def print_evaluation(self, X: np.ndarray, y: np.ndarray, dataset_name: str = "Test"):
        """Print evaluation metrics for a dataset."""
        results = self.evaluate(X, y)
        
        print(f"\n{dataset_name} Set Evaluation:")
        print("-" * 40)
        print(f"  MSE:  {results['mse']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  MAE:  {results['mae']:.4f}")
        print(f"  R²:   {results['r2']:.4f}")
        print("-" * 40)