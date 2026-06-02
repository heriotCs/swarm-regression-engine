# PSO-Optimized Neural Network for Compressive Strength Prediction

## Overview

This project implements a **Particle Swarm Optimization (PSO) engine coupled with a custom Artificial Neural Network (ANN)** to predict concrete compressive strength. Rather than relying on conventional gradient-based training (backpropagation), the system uses a biologically-inspired swarm intelligence algorithm to optimize network weights, demonstrating a viable alternative approach for regression tasks where gradient information may be unavailable or unreliable.

Both the ANN and PSO are built from scratch with no ML framework dependencies, giving full transparency and control over the optimization process.

---

## Setup

**Requirements:** Python 3.x, plus five packages:

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

Place `Concrete_Data.xls` in the project root before running.

---

## Running the System

**Run unit tests:**
```bash
python3 experiments/test_ann.py   # ANN forward propagation
python3 experiments/test_pso.py   # PSO on benchmark functions
```

**Run the full experimental pipeline:**
```bash
cd experiments
python3 trainOnConcrete.py
```

This executes 5 hyperparameter investigations (10-run averages each), saves result plots to `results/plots/`, and writes a metrics summary to `results/metrics.csv`.

**Run baseline comparison (PSO vs Adam optimizer):**
```bash
cd experiments
python3 baseline_comparison.py
```

**Custom usage:**
```python
from Ann.builder import ANNBuilder
from AnnPso.annPsoTrainer import ANNPSOTrainer

network = ANNBuilder.build_regression_network(
    input_size=8,
    hidden_layers=[10],
    activation=['relu']
)

trainer = ANNPSOTrainer(network, X_train, y_train, metric='mse')
trainer.train(
    swarm_size=30,
    max_iters=100,
    topology='fully_informed',
    boundary_mode='reflect'
)
```

---

## Architecture

### ANN (`Ann/`)
- Configurable layers with forward propagation
- Activation functions: Sigmoid, ReLU, Tanh, Linear
- Builder pattern for clean network construction

### PSO (`pso/`)
- Full implementation of Algorithm 39 from *Essentials of Metaheuristics* (Luke, 2013)
- Informant-based topology with constriction factor (x = 0.7298)
- Boundary handling: reflect, clamp, wrap
- Parameter scheduling (c1/c2 linear decay)
- Stagnation detection with automatic diversity recovery

### PSO-ANN Coupling (`AnnPso/`)
- Weight/bias vectors encoded as PSO particle positions
- Fitness evaluated as MSE on training data
- Separate validation tracking to prevent overfitting

---

## Dataset

**UCI Concrete Compressive Strength** -- 1,030 samples, 8 input features (cement, water, coarse/fine aggregates, fly ash, blast furnace slag, superplasticizer, age), 1 continuous output (MPa).

Split: 70% training (721 samples) / 30% test (309 samples).

---

## Experimental Results

Five hyperparameter investigations, each averaged over 10 independent runs (seed=42):

| Investigation | Best Configuration | Best MSE |
|---|---|---|
| Architecture | [8-10-1] | 113.2 +/- 11.5 |
| Swarm budget | 10 particles x 100 iterations | 139.8 +/- 26.5 |
| Acceleration (c1, c2) | Balanced (1.5, 1.5) | -- |
| Topology | Fully Informed | 97.2 +/- 9.8 |
| Activation x Boundary | ReLU + Reflect/Clamp | -- |

### Optimal Configuration

```
Architecture:  [8-10-1]
Activation:    ReLU -> Linear
Topology:      Fully Informed
Swarm size:    30 particles
Iterations:    100
Boundary:      Reflect
c1, c2:        1.5, 1.5
```

### PSO vs Gradient-Based Baseline

| Method | Test MAE | Test MSE | Test RMSE | Training Time |
|---|---|---|---|---|
| PSO-ANN | ~7.8 | ~97.2 | ~9.86 | ~30s |
| Adam (MLP) | ~7.2 | ~89.5 | ~9.46 | ~10s |

PSO achieves comparable predictive accuracy to Adam while offering advantages in gradient-free optimization, useful where loss surfaces are non-differentiable or highly non-convex.

---

## Notable Engineering Decisions

- **No ML framework dependency** -- ANN and PSO fully custom-built for transparency
- **Multiple boundary strategies** -- reflect, clamp, wrap modes selectable at runtime
- **Dynamic topology refresh** -- informants reassigned periodically to prevent premature convergence
- **Reproducible experiments** -- fixed seeds, 10-run averaging, full CSV output

---

## References

1. Luke, S. (2013). *Essentials of Metaheuristics*, Section 3.5 -- PSO Algorithm 39
2. Kennedy & Eberhart (1995). *Particle Swarm Optimization*
3. Clerc & Kennedy (2002). *The Particle Swarm -- Explosion, Stability, and Convergence*
4. UCI ML Repository -- Concrete Compressive Strength Dataset
