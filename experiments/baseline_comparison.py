import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import os, sys, warnings
from sklearn.exceptions import ConvergenceWarning

# ✅ Ignore unnecessary warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ✅ Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Ann.builder import ANNBuilder
from src.AnnPso.annPsoTrainer import ANNPSOTrainer
from src.pso.pso import PSO
from src.pso.fitness import Fitness


# ==========================================================
# CREATE RESULTS DIRECTORY
# ==========================================================
def ensure_dirs():
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)


# ==========================================================
# LOAD DATASET
# ==========================================================
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


# ==========================================================
# PLOT UTILITY
# ==========================================================
def save_plot(fig, filename):
    path = f"results/plots/{filename}"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✅ Saved plot -> {path}")


# ==========================================================
# RUN SINGLE PSO ANN EXPERIMENT (DETAILED OUTPUT)
# ==========================================================
def run_experiment(X_train, X_test, y_train, y_test):
    print("\nRunning baseline architecture 8-10-1...\n")

    network = ANNBuilder.build_regression_network(input_size=8, hidden_layers=[10], seed=42)
    trainer = ANNPSOTrainer(
        network=network,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        metric="mae"
    )

    print("Initial best fitness: inf")

    fitness = Fitness(trainer.ann_fitness, minimize=True)
    bounds = trainer.ann_fitness.get_bounds(weight_range=3.0)
    trainer.pso = PSO(
        fitness=fitness,
        dim=trainer.ann_fitness.num_params,
        bounds=bounds,
        swarm_size=20,
        max_iters=300,
        chi=0.7298,
        c1=2.0,
        c2=2.0,
        topology="random",
        num_informants=3,
        seed=42,
        verbose=True
    )

    # ===== Run PSO =====
    best_weights, best_fitness = trainer.pso.optimise()

    print("\nOptimization complete!")
    print(f"Final best fitness: {best_fitness:.6f}\n")
    print("TRAINING COMPLETE")
    print(f"Best training MAE (normalized): {best_fitness:.6f}\n")

    # ===== Decode best weights =====
    trainer.ann_fitness.decode_weights(best_weights)

    # ===== Evaluate model =====
    preds = trainer.network.forward(X_test).flatten()
    y_true = y_test.flatten()
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    # ===== Example Predictions =====
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

    # ===== Plot Actual vs Predicted =====
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, alpha=0.7, label="Predicted vs Actual")
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Perfect Fit")
    ax.set_xlabel("Actual Strength (MPa)")
    ax.set_ylabel("Predicted Strength (MPa)")
    ax.set_title("PSO ANN (8-10-1) Actual vs Predicted")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "pso_ann_actual_vs_predicted.png")

    return ("8-10-1", mae, r2)


# ==========================================================
# MULTIPLE ARCHITECTURE COMPARISON
# ==========================================================
def run_multiple_experiments(X_train, X_test, y_train, y_test):
    architectures = [[10], [15], [10, 5], [20, 10]]
    results = []

    for layers in architectures:
        print(f"\n{'='*70}")
        print(f"Running architecture: 8-{('-'.join(map(str, layers)))}-1")
        print(f"{'='*70}")

        network = ANNBuilder.build_regression_network(input_size=8, hidden_layers=layers, seed=42)
        trainer = ANNPSOTrainer(
            network=network,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            metric="mae"
        )

        fitness = Fitness(trainer.ann_fitness, minimize=True)
        bounds = trainer.ann_fitness.get_bounds(weight_range=3.0)
        trainer.pso = PSO(
            fitness=fitness,
            dim=trainer.ann_fitness.num_params,
            bounds=bounds,
            swarm_size=20,
            max_iters=200,
            chi=0.7298,
            c1=2.0,
            c2=2.0,
            topology="random",
            num_informants=3,
            seed=42,
            verbose=False
        )

        best_weights, best_fitness = trainer.pso.optimise()
        trainer.ann_fitness.decode_weights(best_weights)

        preds = trainer.network.forward(X_test).flatten()
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results.append((f"8-{'-'.join(map(str, layers))}-1", mae, r2))
        print(f"MAE: {mae:.4f} | R²: {r2:.4f}")

    # ===== Plot Architecture Comparison =====
    labels = [r[0] for r in results]
    maes = [r[1] for r in results]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(labels, maes, color='skyblue', label="MAE (lower is better)")
    ax1.set_ylabel("Mean Absolute Error (MPa)")
    ax1.set_title("PSO ANN - Architecture Performance Comparison")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(fig, "pso_architecture_performance.png")

    return results


# ==========================================================
# BACKPROPAGATION BASELINE
# ==========================================================
def compare_with_backprop(X_train, X_test, y_train, y_test):
    print("\nTraining baseline MLP (Adam Backpropagation)...")
    model = MLPRegressor(
        hidden_layer_sizes=(10,),
        activation='relu',
        solver='adam',
        max_iter=2000,
        learning_rate_init=0.0005,
        alpha=0.001,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Backpropagation (Adam) Results:")
    print(f"  Test MAE: {mae:.4f} MPa")
    print(f"  Test R²: {r2:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, alpha=0.7, color='orange', label="MLP Predictions")
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Perfect Fit")
    ax.set_xlabel("Actual Strength (MPa)")
    ax.set_ylabel("Predicted Strength (MPa)")
    ax.set_title("Backpropagation (Adam) - Actual vs Predicted")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "adam_actual_vs_predicted.png")

    return ("Adam (8-10-1)", mae, r2)


# ==========================================================
# MAIN
# ==========================================================
def main():
    ensure_dirs()
    try:
        X_train, X_test, y_train, y_test = load_data("Concrete_Data.xls")
    except FileNotFoundError:
        print("❌ Error: Concrete_Data.xls not found!")
        return

    all_results = []

    # PSO single baseline
    baseline = run_experiment(X_train, X_test, y_train, y_test)
    all_results.append(baseline)

    # Multiple PSO architectures
    arch_results = run_multiple_experiments(X_train, X_test, y_train, y_test)
    all_results.extend(arch_results)

    # Backpropagation comparison
    bp_result = compare_with_backprop(X_train, X_test, y_train, y_test)
    all_results.append(bp_result)

    # Save summary CSV
    df_results = pd.DataFrame(all_results, columns=["Architecture", "MAE", "R2"])
    df_results.to_csv("results/metrics.csv", index=False)
    print("\n✅ All results saved to results/metrics.csv\n")
    print(df_results)


if __name__ == "__main__":
    main()
