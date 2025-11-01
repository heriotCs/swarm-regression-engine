"""
Task 4: Train ANN to predict concrete compressive strength using PSO

This script loads the concrete dataset, trains an ANN using PSO,
and evaluates its performance on the regression task.

Dataset: Concrete Compressive Strength
Source: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
Download: https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set
"""

import numpy as np
import csv
import os
from Ann.network import NeuralNetwork
from pso.pso import PSO


class ConcreteStrengthPredictor:
    """
    Complete system for training and evaluating ANN on concrete strength prediction.
    """

    def __init__(self):
        self.train_inputs = None
        self.train_outputs = None
        self.test_inputs = None
        self.test_outputs = None
        self.input_scaler = None
        self.output_scaler = None
        self.trained_ann = None
        self.training_history = None

    def load_and_preprocess_data(self, folder_path="Dataset", train_split=0.7, random_seed=42):
        """
        Load and normalize the concrete dataset from the Dataset folder.
        Looks for either:
        - Dataset/Concrete_Data.csv
        - Dataset/concrete_train.csv and Dataset/concrete_test.csv
        """

        print("=" * 70)
        print("LOADING CONCRETE COMPRESSIVE STRENGTH DATASET")
        print("=" * 70)

        dataset_path_csv = os.path.join(folder_path, "Concrete_Data.csv")
        dataset_train = os.path.join(folder_path, "concrete_train.csv")
        dataset_test = os.path.join(folder_path, "concrete_test.csv")

        data = []

        # Load from main CSV if available
        if os.path.exists(dataset_path_csv):
            with open(dataset_path_csv, 'r') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)
                for row in csv_reader:
                    try:
                        data.append([float(value) for value in row])
                    except ValueError:
                        continue
        # Otherwise try combining train + test CSVs
        elif os.path.exists(dataset_train) and os.path.exists(dataset_test):
            for path in [dataset_train, dataset_test]:
                with open(path, 'r') as file:
                    csv_reader = csv.reader(file)
                    header = next(csv_reader)
                    for row in csv_reader:
                        try:
                            data.append([float(value) for value in row])
                        except ValueError:
                            continue
        else:
            raise FileNotFoundError(
                f"No dataset found in {folder_path}\n"
                f"Expected one of:\n - Concrete_Data.csv\n - concrete_train.csv and concrete_test.csv"
            )

        data = np.array(data)
        inputs = data[:, :-1]
        outputs = data[:, -1]

        np.random.seed(random_seed)
        shuffle_indices = np.random.permutation(len(data))
        inputs = inputs[shuffle_indices]
        outputs = outputs[shuffle_indices]

        split_idx = int(len(data) * train_split)
        train_inputs_raw = inputs[:split_idx]
        train_outputs_raw = outputs[:split_idx]
        test_inputs_raw = inputs[split_idx:]
        test_outputs_raw = outputs[split_idx:]

        # Normalize (Z-score)
        input_mean = np.mean(train_inputs_raw, axis=0)
        input_std = np.std(train_inputs_raw, axis=0)
        output_mean = np.mean(train_outputs_raw)
        output_std = np.std(train_outputs_raw)

        self.input_scaler = {'mean': input_mean, 'std': input_std}
        self.output_scaler = {'mean': output_mean, 'std': output_std}

        self.train_inputs = (train_inputs_raw - input_mean) / (input_std + 1e-8)
        self.train_outputs = (train_outputs_raw - output_mean) / (output_std + 1e-8)
        self.test_inputs = (test_inputs_raw - input_mean) / (input_std + 1e-8)
        self.test_outputs = (test_outputs_raw - output_mean) / (output_std + 1e-8)

        stats = {
            'total_samples': len(data),
            'train_samples': len(self.train_inputs),
            'test_samples': len(self.test_inputs),
            'num_features': inputs.shape[1],
            'output_range': (np.min(outputs), np.max(outputs)),
            'output_mean': output_mean,
            'output_std': output_std
        }

        return stats

    def denormalize_predictions(self, normalized_predictions):
        return (normalized_predictions * self.output_scaler['std']) + self.output_scaler['mean']

    def create_fitness_function(self):
        def fitness_function(parameter_vector):
            ann = NeuralNetwork(self.ann_architecture, self.activation_functions)
            ann.set_parameters(parameter_vector)
            predictions = ann.predict(self.train_inputs)
            mae = np.mean(np.abs(predictions - self.train_outputs))
            return mae
        return fitness_function

    def train_with_pso(self, ann_architecture=[8, 10, 5, 1],
                       activation_functions=['relu', 'relu', 'linear'],
                       swarm_size=30, num_iterations=100, num_informants=3,
                       phi1=2.05, phi2=2.05, boundary_handling='clip',
                       weight_bound=2.0, verbose=True):
        """Train ANN using PSO."""
        self.ann_architecture = ann_architecture
        self.activation_functions = activation_functions

        template_ann = NeuralNetwork(ann_architecture, activation_functions)
        num_parameters = template_ann.get_num_parameters()

        fitness_func = self.create_fitness_function()

        pso = PSO(
            fitness_function=fitness_func,
            num_dimensions=num_parameters,
            swarm_size=swarm_size,
            num_iterations=num_iterations,
            num_informants=num_informants,
            phi1=phi1,
            phi2=phi2,
            min_bound=-weight_bound,
            max_bound=weight_bound,
            boundary_handling=boundary_handling,
            verbose=verbose
        )

        best_parameters, best_fitness = pso.optimize()
        self.trained_ann = NeuralNetwork(ann_architecture, activation_functions)
        self.trained_ann.set_parameters(best_parameters)
        self.training_history = pso.get_fitness_history()

        print("\nTRAINING COMPLETE")
        print(f"Best training MAE (normalized): {best_fitness:.6f}")

        return self.trained_ann

    def evaluate_model(self, ann=None, show_examples=5):
        if ann is None:
            ann = self.trained_ann
        if ann is None:
            raise ValueError("No trained model available. Train model first.")

        train_pred = ann.predict(self.train_inputs)
        test_pred = ann.predict(self.test_inputs)

        train_mae = np.mean(np.abs(train_pred - self.train_outputs))
        train_rmse = np.sqrt(np.mean((train_pred - self.train_outputs) ** 2))
        train_r2 = 1 - np.sum((self.train_outputs - train_pred) ** 2) / np.sum(
            (self.train_outputs - np.mean(self.train_outputs)) ** 2)

        test_mae = np.mean(np.abs(test_pred - self.test_outputs))
        test_rmse = np.sqrt(np.mean((test_pred - self.test_outputs) ** 2))
        test_r2 = 1 - np.sum((self.test_outputs - test_pred) ** 2) / np.sum(
            (self.test_outputs - np.mean(self.test_outputs)) ** 2)

        train_pred_mpa = self.denormalize_predictions(train_pred)
        test_pred_mpa = self.denormalize_predictions(test_pred)
        train_actual_mpa = self.denormalize_predictions(self.train_outputs)
        test_actual_mpa = self.denormalize_predictions(self.test_outputs)

        train_mae_mpa = np.mean(np.abs(train_pred_mpa - train_actual_mpa))
        test_mae_mpa = np.mean(np.abs(test_pred_mpa - test_actual_mpa))
        train_rmse_mpa = np.sqrt(np.mean((train_pred_mpa - train_actual_mpa) ** 2))
        test_rmse_mpa = np.sqrt(np.mean((test_pred_mpa - test_actual_mpa) ** 2))

        if show_examples > 0:
            print(f"\nExample Predictions (first {show_examples} test samples):")
            print("-" * 70)
            print(f"{'Actual (MPa)':<15} {'Predicted (MPa)':<18} {'Error (MPa)':<15} {'Error %':<10}")
            print("-" * 70)
            for i in range(min(show_examples, len(test_actual_mpa))):
                actual = test_actual_mpa[i].item() if isinstance(test_actual_mpa[i], np.ndarray) else test_actual_mpa[i]
                predicted = test_pred_mpa[i].item() if isinstance(test_pred_mpa[i], np.ndarray) else test_pred_mpa[i]
                error = abs(actual - predicted)
                error_pct = (error / actual * 100) if actual != 0 else 0
                print(f"{actual:<15.2f} {predicted:<18.2f} {error:<15.2f} {error_pct:<10.2f}")

        results = {
            'train_mae': train_mae, 'train_rmse': train_rmse, 'train_r2': train_r2,
            'test_mae': test_mae, 'test_rmse': test_rmse, 'test_r2': test_r2,
            'train_mae_mpa': train_mae_mpa, 'train_rmse_mpa': train_rmse_mpa,
            'test_mae_mpa': test_mae_mpa, 'test_rmse_mpa': test_rmse_mpa,
            'train_predictions': train_pred, 'test_predictions': test_pred
        }

        return results


if __name__ == "__main__":
    predictor = ConcreteStrengthPredictor()

    try:
        stats = predictor.load_and_preprocess_data(folder_path="Dataset")
        trained_ann = predictor.train_with_pso(
            ann_architecture=[8, 10, 5, 1],
            activation_functions=['relu', 'relu', 'linear'],
            swarm_size=20,
            num_iterations=50,
            num_informants=3,
            verbose=True
        )
        results = predictor.evaluate_model(show_examples=10)

        print("\nSummary:")
        print(f"  Dataset loaded: {stats['total_samples']} samples")
        print(f"  Train/test split: {stats['train_samples']}/{stats['test_samples']}")
        print(f"  Test MAE: {results['test_mae_mpa']:.4f} MPa")
        print(f"  Test R²: {results['test_r2']:.4f}")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please ensure the dataset is in the 'Dataset' folder.")
