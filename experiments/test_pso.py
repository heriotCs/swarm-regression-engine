import sys
import os

# Add the parent directory to the Python path so this file can import the PSO package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.pso.fitness import Fitness
from src.pso.pso import PSO


# Basic Sphere test function – common benchmark for optimisation.
# Has a global minimum at x = 0 (ideal for checking if PSO behaves correctly).
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


if __name__ == "__main__":
    # Optional global seed to make the run reproducible
    np.random.seed(42)

    dim = 10
    # Same bounds for all dimensions (standard Sphere test range)
    bounds = [(-5.12, 5.12)] * dim

    # Wrap the sphere function so it fits the Fitness interface used by PSO
    fit = Fitness(sphere)

    # Create PSO instance with chosen hyperparameters.
    # Some "Going Further" features (like scheduling c1/c2) are enabled here.
    pso = PSO(
        fitness=fit,
        dim=dim,
        bounds=bounds,
        swarm_size=40,
        max_iters=150,
        chi=0.7298,
        c1=1.5,
        c2=1.5,
        c3=0.0,  # δ=0 is the usual constriction variant recommended by Luke
        topology="random",
        num_informants=3,
        refresh_rate=7,
        boundary_mode="reflect",
        vmax=None,
        schedule_c1=(2.0, 1.2),  # Cognitive weight slowly decreases
        schedule_c2=(1.2, 2.0),  # Social weight slowly increases
        schedule_chi=None,
        stagnation_iters=60,
        seed=42,
        verbose=True,
    )

    # Run the full optimisation loop
    best_pos, best_fit = pso.optimise()

    # Print summary at the end of the run
    print("\n============================================================")
    print("PSO Run Complete")
    print("============================================================")
    print(f"Best fitness: {best_fit:.10f}")
    print(f"Best pos (first 5 dims): {best_pos[:5]}")