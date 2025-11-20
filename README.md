# BicAnnImplementation-

# Particle Swarm Optimization for Neural Network Training
## F20BC/F21BC Biologically-Inspired Computation Coursework

### Team Members
Student 1: Muhammad Hassan (H00390896)
Student 2: Saad Noman (H00399124)


Project Overview

This project implements Artificial Neural Networks (ANNs) and Particle Swarm Optimization (PSO)  from scratch to solve a regression problem which is predicting concrete compressive strength. This implementation sees how we can use PSO to effectively optimize ANN weights rather than using default gradient-based methods.

Key Achievement: Our PSO-ANN system achieves comparable performance to gradient-based methods (Adam optimizer) while demonstrating the effectiveness of biologically-inspired optimization.

---

Quick Start Guide

Need to install 5 required packages:
. numpy 
. pandas
. matplotlib
. scikit-learn
. openpyx

# 1. Install packages
pip install numpy pandas matplotlib scikit-learn openpyxl

make sure dataset is present
# Place Concrete_Data.xls in the project root directory


### Running the Code

1. Test individual components :

Test ANN implementation (Task 1)
"python3 experiments/test_ann.py"

Test PSO on benchmark functions (Task 2)
"python3 experiments/test_pso.py"


2. Run Main Experiment (Tasks 3-5)

Train the PSO–ANN on the concrete dataset with the full experimental setup.

Important: To run the main experiment, first change into the experiments directory.

"cd experiments"

"python3 trainOnConcrete.py" # run the main experiments


This will:
- Load and preprocess the concrete dataset (70/30 train/test split)
- Run all 5 experimental investigations with 10-run averages
- Generate result plots in "experiments/results/plots/"
- Save summary CSV in "experiments/results/"

3. Run Baseline Comaprison (PSO vs Adam Backwardpropagation)

This script trains multiple ANN architectures using PSO and also trains a baseline MLP using the Adam optimizer.
It prints full training logs, saves plots, and writes all metrics to results/metrics.csv.

To run:
"cd experiments"

"python3 basline_comparison.py"

#### 3. Train with Custom Parameters
```python
from Ann.builder import ANNBuilder
from AnnPso.annPsoTrainer import ANNPSOTrainer

# Create network
network = ANNBuilder.build_regression_network(
    input_size=8,
    hidden_layers=[10],  # Best architecture found
    activation=['relu']   # Best activation found
)

# Train with PSO
trainer = ANNPSOTrainer(network, X_train, y_train, metric='mse')
trainer.train(
    swarm_size=30,
    max_iters=100,
    topology='fully_informed',  # Best topology found
    boundary_mode='reflect'     # Best boundary handling
)
```

---

## Implementation Details

### Task 1: ANN Implementation (`Ann/`)
- **network.py**: Core neural network with configurable layers and forward propagation
- **layer.py**: Layer operations (linear transformation + activation)
- **activation.py**: Implemented activation functions:
  - Logistic (Sigmoid)
  - ReLU
  - Tanh
  - Linear (for output layer)
- **builder.py**: Factory patterns for easy network construction

### Task 2: PSO Implementation (`pso/`)
- **pso.py**: Full Algorithm 39 from "Essentials of Metaheuristics" with:
  - Informant-based topology (lines clearly commented with algorithm line numbers)
  - Constriction factor (χ = 0.7298)
  - Multiple boundary handling modes (reflect, clamp, wrap)
  - Parameter scheduling (c1, c2 linear decay)
  - Stagnation detection and recovery
- **particle.py**: Individual particle with position/velocity updates
- **informants.py**: Topology implementations (random, ring, fully_informed)
- **fitness.py**: Fitness function wrapper for minimization/maximization

### Task 3: PSO-ANN Coupling (`AnnPso/`)
- **annFitness.py**: 
  - Encodes ANN weights/biases as PSO particle position vector
  - Decodes particle position back to network parameters
  - Evaluates fitness as MSE on training data
- **annPsoTrainer.py**: 
  - High-level training interface
  - Handles train/validation splits
  - Provides comprehensive evaluation metrics

### Task 4: Problem Application
- **Dataset**: UCI Concrete Compressive Strength (1030 samples)
- **Input**: 8 features (cement, water, aggregates, etc.)
- **Output**: Compressive strength (MPa)
- **Split**: 70% training (721), 30% testing (309)
- **Metric**: Mean Absolute Error (MAE) and MSE

### Task 5: Experimental Investigation (`Ann/trainOnConcrete.py`)
Comprehensive hyperparameter study with 10-run averages investigating:

1. **ANN Architecture Effect**
   - Tested: [8-5-1], [8-10-1], [8-20-1], [8-10-5-1], [8-20-10-1]
   - **Best**: [8-10-1] (MSE: 113.2±11.5)

2. **Swarm Size vs Iterations Trade-off** (Budget: 1000 evaluations)
   - Tested: 10×100, 20×50, 50×20, 100×10
   - **Best**: 10×100 (MSE: 139.8±26.5)

3. **Acceleration Coefficients (c1, c2)**
   - Tested: Exploration→Exploitation strategies
   - **Best**: Balanced (c1=1.5, c2=1.5) or Exploitation (c1=0.5, c2=2.5)

4. **PSO Topology**
   - Tested: random, ring, fully_informed
   - **Best**: fully_informed (MSE: 97.2±9.8)

5. **Design Decisions** (Activation × Boundary)
   - **Best**: ReLU with clamp or reflect boundary handling

---

## Results Summary

### Best Configuration Found
```
Architecture:     [8-10-1]
Activation:       ReLU → Linear
Topology:         Fully Informed
Swarm Size:       30
Iterations:       100
Boundary:         Reflect
c1, c2:          1.5, 1.5 (balanced)
```

### Performance Comparison
| Method | Test MAE | Test MSE | Test RMSE | Training Time |
|--------|----------|----------|-----------|---------------|
| PSO-ANN | ~7.8 | ~97.2 | ~9.86 | ~30s |
| Adam (Baseline) | ~7.2 | ~89.5 | ~9.46 | ~10s |

**Key Findings**: PSO achieves comparable performance to gradient-based methods while offering advantages in avoiding local minima and not requiring differentiable activation functions.

---

## Extensions Beyond Core Specification

### Implemented Extensions ("Going Further")
1. **Multiple Boundary Handling**: Reflect, clamp, and wrap modes
2. **Parameter Scheduling**: Linear decay for c1, c2
3. **Stagnation Recovery**: Automatic diversity injection
4. **Topology Refresh**: Dynamic informant reassignment
5. **Validation Tracking**: Separate best validation weights
6. **Comprehensive Metrics**: MSE, RMSE, MAE, R²

---

##  Use of Generative AI

### Disclosure
This project was developed with assistance from GitHub Copilot for:
- Boilerplate code generation (data loading, plotting functions)
- Documentation and comment generation
- Bug fixing suggestions

### Human-Written Components
- Core algorithm implementations (PSO Algorithm 39, ANN forward propagation)
- Experimental design and analysis
- PSO-ANN coupling strategy
- Hyperparameter investigations

### AI-Assisted Components
- Utility functions in `trainOnConcrete.py` (plotting, data preprocessing)
- Standard evaluation metrics calculations
- File I/O operations

**All code includes comments mapping to algorithm specifications as required.**

---

## Key References

1. Luke, S. (2013). *Essentials of Metaheuristics* (Section 3.5: PSO Algorithm 39)
2. Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization"
3. Clerc, M., & Kennedy, J. (2002). "The particle swarm - explosion, stability, and convergence"
4. UCI Machine Learning Repository - Concrete Compressive Strength Dataset

---

## Team Contributions

### Student 1: Muhammad Hassan
- ANN implementation (network.py, layer.py, activation.py)
- PSO-ANN coupling (annFitness.py, annPsoTrainer.py)
- Experimental investigations 4 and 5
- Report sections: Results & Discussion

### Student 2: Saad Noman
- PSO implementation (pso.py, particle.py, informants.py)
- Experimental investigations 1-3
- Report sections: Implementation


---

## Notes for Marking

1. **Line-by-line mapping**: PSO implementation in `pso/pso.py` includes comments mapping to Algorithm 39 line numbers
2. **Reproducibility**: All experiments use fixed seeds (seed=42 default)
3. **10-run averages**: All results show mean±std over 10 independent runs
4. **Extensions implemented**: "Going Further" section above
5. **To run full experiment suite**: run `python3 experiment/trainOnConcrete.py`

---

## Testing Guide


1. Verify ANN implementation
python3 experiments/test_ann.py
Expected: Network summary, forward propagation tests pass

2. Verify PSO implementation  
python3 experiments/test_pso.py
Expected: Sphere function optimized to near 0

3. Run complete experimental study
python3 experiments/trainOnConcrete.py
Expected: 5 experiments complete, plots saved, CSV generated

# 4. Check results
All experiment outputs are stored in the `results` directory:

- **Plots** are saved in:
  results/plots

- **Performance metrics (MAE, R², architectures, Adam baseline)** are saved in:
  results/metrics.csv

---

*Last Updated: 20th November 2025*
*Coursework for F20BC - Biologically-Inspired Computation*
