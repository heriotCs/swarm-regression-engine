import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Ann.builder import ANNBuilder
from AnnPso.annPsoTrainer import ANNPSOTrainer


def load_and_preprocess_data(filepath: str, test_size: float = 0.3, seed: int = 42):
    
    print("Loading concrete dataset...")
    df = pd.read_excel(filepath)
    
    # Split features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"✓ Dataset loaded:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, scaler


def train_simple_network(X_train, X_test, y_train, y_test):
    """Train a simple 8-10-1 network."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Simple Network (8-10-1)")
    print("="*60)
    
    # Create network
    network = ANNBuilder.build_regression_network(
        input_size=8,
        hidden_layers=[10],
        seed=42
    )
    
    # Create trainer
    trainer = ANNPSOTrainer(
        network=network,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        metric="mse"
    )
    
    # Train with PSO
    best_weights, best_fitness = trainer.train(
        swarm_size=30,
        max_iters=100,
        weight_range=3.0,
        chi=0.7298,
        c1=2.0,
        c2=2.0,
        topology="random",
        num_informants=3,
        stagnation_iters=40,
        schedule_c1=(2.5, 1.0),  # Decay cognitive
        schedule_c2=(1.0, 2.5),  # Increase social
        seed=42,
        verbose=True
    )
    
    # Evaluate
    trainer.print_evaluation(X_train, y_train, "Training")
    trainer.print_evaluation(X_test, y_test, "Test")
    
    return trainer


def train_deep_network(X_train, X_test, y_train, y_test):
    """Train a deeper network."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Deep Network (8-20-10-1)")
    print("="*60)
    
    # Create deeper network
    network = ANNBuilder.build_regression_network(
        input_size=8,
        hidden_layers=[20, 10],
        seed=42
    )
    
    # Create trainer
    trainer = ANNPSOTrainer(
        network=network,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        metric="mse"
    )
    
    # Train with PSO (more particles for more complex network)
    best_weights, best_fitness = trainer.train(
        swarm_size=50,
        max_iters=150,
        weight_range=3.0,
        chi=0.7298,
        c1=1.8,
        c2=1.8,
        topology="random",
        num_informants=5,
        stagnation_iters=50,
        schedule_c1=(2.5, 1.2),
        schedule_c2=(1.2, 2.5),
        seed=42,
        verbose=True
    )
    
    # Evaluate
    trainer.print_evaluation(X_train, y_train, "Training")
    trainer.print_evaluation(X_test, y_test, "Test")
    
    return trainer


def visualize_results(trainer, X_test, y_test):
    """Create visualization plots."""
    print("\nGenerating visualizations...")
    
    results = trainer.evaluate(X_test, y_test, return_predictions=True)
    predictions = results['predictions'].flatten()
    y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot: Predicted vs Actual
    axes[0].scatter(y_test_flat, predictions, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[0].plot([y_test_flat.min(), y_test_flat.max()], 
                 [y_test_flat.min(), y_test_flat.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Strength (MPa)', fontsize=11)
    axes[0].set_ylabel('Predicted Strength (MPa)', fontsize=11)
    axes[0].set_title('Predicted vs Actual Concrete Strength', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram: Prediction errors
    residuals = y_test_flat - predictions
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_xlabel('Prediction Error (MPa)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ann_pso_results.png', dpi=150, bbox_inches='tight')
    print("✓ Plots saved as 'ann_pso_results.png'")
    plt.show()


def compare_topologies(X_train, X_test, y_train, y_test):
    """Compare different PSO topologies."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Topology Comparison")
    print("="*60)
    
    topologies = ["random", "ring", "fully_informed"]
    results_summary = []
    
    for topology in topologies:
        print(f"\n--- Testing topology: {topology} ---")
        
        network = ANNBuilder.build_regression_network(
            input_size=8,
            hidden_layers=[10],
            seed=42
        )
        
        trainer = ANNPSOTrainer(
            network=network,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            metric="mse"
        )
        
        best_weights, best_fitness = trainer.train(
            swarm_size=30,
            max_iters=80,
            weight_range=3.0,
            topology=topology,
            num_informants=3 if topology == "random" else 3,
            seed=42,
            verbose=False
        )
        
        test_results = trainer.evaluate(X_test, y_test)
        results_summary.append({
            'topology': topology,
            'train_mse': best_fitness,
            'test_mse': test_results['mse'],
            'test_r2': test_results['r2']
        })
        
        print(f"  Training MSE: {best_fitness:.4f}")
        print(f"  Test MSE: {test_results['mse']:.4f}")
        print(f"  Test R²: {test_results['r2']:.4f}")
    
    print("\n" + "="*60)
    print("TOPOLOGY COMPARISON SUMMARY")
    print("="*60)
    for r in results_summary:
        print(f"{r['topology']:20s} | Train MSE: {r['train_mse']:8.4f} | "
              f"Test MSE: {r['test_mse']:8.4f} | R²: {r['test_r2']:6.4f}")

# best way of allocating solution evaluations
def investigate_swarm_size_vs_iterations(X_train, X_test, y_train, y_test):
    
    print("\n" + "="*60)
    print("EXPERIMENT 4: Swarm Size vs Iterations (Fixed Budget)")
    print("="*60)
    
    budget = 1000  # Fixed evaluation budget
    configs = [
        (10, 100),   # Small swarm, many iterations
        (20, 50),    # Medium swarm, medium iterations
        (50, 20),    # Large swarm, few iterations
        (100, 10),   # Very large swarm, very few iterations
    ]
    
    results = []
    
    for swarm_size, max_iters in configs:
        print(f"\n--- Testing: Swarm={swarm_size}, Iters={max_iters} ---")
        print(f"    Total evaluations: {swarm_size * max_iters}")
        
        network = ANNBuilder.build_regression_network(
            input_size=8,
            hidden_layers=[10],
            seed=42
        )
        
        trainer = ANNPSOTrainer(
            network=network,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            metric="mse"
        )
        
        best_weights, best_fitness = trainer.train(
            swarm_size=swarm_size,
            max_iters=max_iters,
            weight_range=3.0,
            seed=42,
            verbose=False
        )
        
        test_metrics = trainer.evaluate(X_test, y_test)
        
        results.append({
            'config': f"S={swarm_size}, I={max_iters}",
            'swarm': swarm_size,
            'iters': max_iters,
            'train_mse': best_fitness,
            'test_mse': test_metrics['mse'],
            'test_r2': test_metrics['r2']
        })
        
        print(f"  Training MSE: {best_fitness:.4f}")
        print(f"  Test MSE: {test_metrics['mse']:.4f}")
    
    print("\n" + "="*60)
    print("ALLOCATION COMPARISON (Budget = 1000 evaluations)")
    print("="*60)
    print(f"{'Config':15s} | {'Train MSE':>10s} | {'Test MSE':>10s} | {'R²':>8s}")
    print("-" * 60)
    for r in results:
        print(f"{r['config']:15s} | {r['train_mse']:10.4f} | "
              f"{r['test_mse']:10.4f} | {r['test_r2']:8.4f}")
    
    # Find best configuration
    best_config = min(results, key=lambda x: x['test_mse'])
    print("\n" + "="*60)
    print(f"BEST CONFIGURATION: Swarm={best_config['swarm']}, Iterations={best_config['iters']}")
    print(f"Test MSE: {best_config['test_mse']:.4f}")
    print("="*60)

# effect of varying the acceleration coefficients
def investigate_acceleration_coefficients(X_train, X_test, y_train, y_test):
    
    print("\n" + "="*60)
    print("EXPERIMENT 5: Acceleration Coefficients (c1 and c2)")
    print("Coursework Question: Effect of acceleration coefficients")
    print("="*60)
    
    configs = [
        (2.5, 0.5, "High Cognitive, Low Social (Exploration)"),
        (2.0, 1.0, "Cognitive Dominant"),
        (1.5, 1.5, "Balanced (Equal weights)"),
        (1.0, 2.0, "Social Dominant"),
        (0.5, 2.5, "Low Cognitive, High Social (Exploitation)"),
    ]
    
    results = []
    
    for c1, c2, description in configs:
        print(f"\n--- Testing: c1={c1}, c2={c2} ---")
        print(f"    Strategy: {description}")
        
        network = ANNBuilder.build_regression_network(
            input_size=8,
            hidden_layers=[10],
            seed=42
        )
        
        trainer = ANNPSOTrainer(
            network=network,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            metric="mse"
        )
        
        best_weights, best_fitness = trainer.train(
            swarm_size=30,
            max_iters=100,
            weight_range=3.0,
            c1=c1,
            c2=c2,
            seed=42,
            verbose=False
        )
        
        test_metrics = trainer.evaluate(X_test, y_test)
        
        results.append({
            'c1': c1,
            'c2': c2,
            'description': description,
            'train_mse': best_fitness,
            'test_mse': test_metrics['mse'],
            'test_r2': test_metrics['r2']
        })
        
        print(f"  Training MSE: {best_fitness:.4f}")
        print(f"  Test MSE: {test_metrics['mse']:.4f}")
    
    print("\n" + "="*60)
    print("ACCELERATION COEFFICIENT COMPARISON")
    print("="*60)
    print(f"{'c1':>4s} | {'c2':>4s} | {'Strategy':35s} | {'Train MSE':>10s} | {'Test MSE':>10s} | {'R²':>8s}")
    print("-" * 85)
    for r in results:
        print(f"{r['c1']:4.1f} | {r['c2']:4.1f} | {r['description']:35s} | "
              f"{r['train_mse']:10.4f} | {r['test_mse']:10.4f} | {r['test_r2']:8.4f}")
    
    # Find best configuration
    best_config = min(results, key=lambda x: x['test_mse'])
    print("\n" + "="*60)
    print(f"BEST CONFIGURATION: c1={best_config['c1']}, c2={best_config['c2']}")
    print(f"Strategy: {best_config['description']}")
    print(f"Test MSE: {best_config['test_mse']:.4f}")
    print("="*60)

# Investigating the effect ANN architecture have on its ability to solve the problem
def investigate_architecture(X_train, X_test, y_train, y_test):
    
    print("\n" + "="*60)
    print("EXPERIMENT 6: ANN Architecture Comparison")
    print("="*60)
    
    architectures = [
        ([5], "8-5-1 (Small)"),
        ([10], "8-10-1 (Medium)"),
        ([20], "8-20-1 (Large)"),
        ([10, 5], "8-10-5-1 (Two layers)"),
        ([20, 10], "8-20-10-1 (Deep)"),
    ]
    
    results = []
    
    for hidden_layers, description in architectures:
        print(f"\n--- Testing architecture: {description} ---")
        
        network = ANNBuilder.build_regression_network(
            input_size=8,
            hidden_layers=hidden_layers,
            seed=42
        )
        
        trainer = ANNPSOTrainer(
            network=network,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            metric="mse"
        )
        
        # Adjust iterations based on network size
        num_params = trainer.ann_fitness.num_params
        max_iters = 100 if num_params < 100 else 150
        
        best_weights, best_fitness = trainer.train(
            swarm_size=30,
            max_iters=max_iters,
            weight_range=3.0,
            seed=42,
            verbose=False
        )
        
        test_metrics = trainer.evaluate(X_test, y_test)
        
        results.append({
            'architecture': description,
            'params': num_params,
            'train_mse': best_fitness,
            'test_mse': test_metrics['mse'],
            'test_r2': test_metrics['r2']
        })
        
        print(f"  Parameters: {num_params}")
        print(f"  Training MSE: {best_fitness:.4f}")
        print(f"  Test MSE: {test_metrics['mse']:.4f}")
    
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON")
    print("="*60)
    print(f"{'Architecture':20s} | {'Params':>7s} | {'Train MSE':>10s} | {'Test MSE':>10s} | {'R²':>8s}")
    print("-" * 70)
    for r in results:
        print(f"{r['architecture']:20s} | {r['params']:7d} | "
              f"{r['train_mse']:10.4f} | {r['test_mse']:10.4f} | {r['test_r2']:8.4f}")
    
    # Find best architecture
    best_arch = min(results, key=lambda x: x['test_mse'])
    print("\n" + "="*60)
    print(f"BEST ARCHITECTURE: {best_arch['architecture']}")
    print(f"Parameters: {best_arch['params']}")
    print(f"Test MSE: {best_arch['test_mse']:.4f}, R²: {best_arch['test_r2']:.4f}")
    print("="*60)

# main that runs all cw experiments 
def main():

    print("\n" + "="*60)
    print("ANN Training with PSO - Concrete Strength Prediction")
    print("="*60)
    
    # Load data
    try:
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
            "Concrete_Data.xls",
            test_size=0.3,
            seed=42
        )
    except FileNotFoundError:
        print("\nError: Concrete_Data.xls not found!")
        print("Please ensure the dataset is in the current directory.")
        return
    
    # Run all experiments
    print("\nRunning coursework experiments...")
    print("This will take several minutes...\n")
    
    # Baseline experiments
    print("\n" + "#"*60)
    print("# BASELINE EXPERIMENTS")
    print("#"*60)
    
    trainer1 = train_simple_network(X_train, X_test, y_train, y_test)
    trainer2 = train_deep_network(X_train, X_test, y_train, y_test)
    
    # Required coursework experiments
    print("\n" + "#"*60)
    print("Cw experiments")
    print("#"*60)
    
    # Question 1: Architecture effect
    investigate_architecture(X_train, X_test, y_train, y_test)
    
    # Question 2: Solution evaluation allocation
    investigate_swarm_size_vs_iterations(X_train, X_test, y_train, y_test)
    
    # Question 3: Acceleration coefficients
    investigate_acceleration_coefficients(X_train, X_test, y_train, y_test)
    
    # Additional experiments
    print("\n" + "#"*60)
    print("# ADDITIONAL EXPERIMENTS")
    print("#"*60)
    
    compare_topologies(X_train, X_test, y_train, y_test)
    
    # Generate visualizations
    visualize_results(trainer1, X_test, y_test)
    
    print("\n" + "="*60)
    print("All experiments completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()