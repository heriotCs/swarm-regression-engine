import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import sys
import os

# ✅ Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Ann.builder import ANNBuilder
from AnnPso.annPsoTrainer import ANNPSOTrainer
from pso.pso import PSO
from pso.fitness import Fitness


def load_data(filepath: str = "Concrete_Data.xls"):
    print("=" * 70)
    print("LOADING CONCRETE COMPRESSIVE STRENGTH DATASET")
    print("=" * 70)

    df = pd.read_excel(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Dataset loaded: {len(df)} samples")
    print(f"Train/Test split: {len(X_train)}/{len(X_test)}\n")
    return X_train, X_test, y_train, y_test


def run_experiment(X_train, X_test, y_train, y_test):
    # Build a simple 8-10-1 network
    network = ANNBuilder.build_regression_network(
        input_size=8,
        hidden_layers=[10],
        seed=42
    )

    # Initialize trainer
    trainer = ANNPSOTrainer(
        network=network,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        metric="mae"  # Using MAE as fitness
    )

    print("Initial best fitness: inf")

    # PSO hyperparameters
    swarm_size = 30
    max_iters = 100
    weight_range = 3.0
    np.random.seed(42)

    # ✅ Create Fitness wrapper & bounds (as in ANNPSOTrainer.train)
    fitness = Fitness(trainer.ann_fitness, minimize=True)
    bounds = trainer.ann_fitness.get_bounds(weight_range)

    # ✅ Initialize PSO
    trainer.pso = PSO(
        fitness=fitness,
        dim=trainer.ann_fitness.num_params,
        bounds=bounds,
        swarm_size=swarm_size,
        max_iters=max_iters,
        chi=0.7298,
        c1=2.0,
        c2=2.0,
        topology="random",
        num_informants=3,
        seed=42,
        verbose=True
    )

    # ✅ Run optimization
    best_weights, best_fitness = trainer.pso.optimise()

    print("\nOptimization complete!")
    print(f"Final best fitness: {best_fitness:.6f}\n")
    print("TRAINING COMPLETE")
    print(f"Best training MAE (normalized): {best_fitness:.6f}\n")

    # ✅ Apply the best weights properly using ann_fitness decoder
    trainer.ann_fitness.decode_weights(best_weights)

    # Evaluate model on test data
    preds = trainer.network.forward(X_test).flatten()
    y_true = y_test.flatten()
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    # Example Predictions
    print("Example Predictions (first 10 test samples):")
    print("-" * 70)
    print(f"{'Actual (MPa)':<15s}{'Predicted (MPa)':<18s}{'Error (MPa)':<15s}{'Error %':<10s}")
    print("-" * 70)
    for i in range(min(10, len(y_true))):
        err = abs(y_true[i] - preds[i])
        err_pct = (err / y_true[i] * 100) if y_true[i] != 0 else 0
        print(f"{y_true[i]:<15.2f}{preds[i]:<18.2f}{err:<15.2f}{err_pct:<10.2f}")

    print("\nSummary:")
    print(f"  Dataset loaded: {len(X_train) + len(X_test)} samples")
    print(f"  Train/test split: {len(X_train)}/{len(X_test)}")
    print(f"  Test MAE: {mae:.4f} MPa")
    print(f"  Test R²: {r2:.4f}")
    print("=" * 70)


def main():
    try:
        X_train, X_test, y_train, y_test = load_data("Concrete_Data.xls")
    except FileNotFoundError:
        print("Error: Concrete_Data.xls not found!")
        return
    run_experiment(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()