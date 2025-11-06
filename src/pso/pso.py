# pso.py
# Main PSO implementation following Algorithm 39 from Luke.
# This file controls the whole optimisation loop and ties together
# particles, informants, fitness evaluation, parameter scheduling, and resets.

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Literal, Optional

import os
import sys

# Add parent directory so imports work regardless of where the script is run
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pso.particle import Particle
from pso.informants import Informants
from pso.fitness import Fitness

BoundaryMode = Literal["clamp", "reflect", "wrap"]
Topology = Literal["random", "ring", "fully_informed"]


class PSO:
    def __init__(
        self,
        fitness: Fitness,
        dim: int,
        bounds: List[Tuple[float, float]],
        swarm_size: int = 30,
        chi: float = 0.7298,
        c1: float = 1.49618,
        c2: float = 1.49618,
        c3: float = 0.0,
        step_scale: float = 1.0,
        topology: Topology = "random",
        num_informants: int = 3,
        refresh_rate: int = 5,
        boundary_mode: BoundaryMode = "reflect",
        vmax: float | None = None,
        stagnation_iters: Optional[int] = 50,
        schedule_c1: Optional[tuple[float, float]] = None,
        schedule_c2: Optional[tuple[float, float]] = None,
        schedule_chi: Optional[tuple[float, float]] = None,
        max_iters: int = 200,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        # Store objective function wrapper
        self.fitness = fitness

        # Dimensionality of ANN parameter vector
        self.dim = dim
        self.bounds = bounds

        # Standard PSO parameters
        self.swarm_size = swarm_size
        self.max_iters = max_iters
        self.verbose = verbose

        # Base coefficients (chi, c1, c2) – possibly overridden by schedules
        self.base_chi = chi
        self.base_c1 = c1
        self.base_c2 = c2
        self.c3 = c3       # optional global attractor term (not always used)
        self.step_scale = step_scale

        # Parameter schedules (linear interpolation over time)
        self.schedule_c1 = schedule_c1
        self.schedule_c2 = schedule_c2
        self.schedule_chi = schedule_chi

        # Boundary and velocity configuration
        self.vmax = vmax
        self.boundary_mode = boundary_mode
        self.stagnation_iters = stagnation_iters

        # RNG for reproducibility
        self.rng = np.random.default_rng(seed)

        # ------------------ Particle Initialisation ------------------
        # A39 L7–L9: create swarm with randomly initialised positions
        self.particles: List[Particle] = [
            Particle(dim, bounds, rng=self.rng, vmax=vmax, boundary_mode=boundary_mode)
            for _ in range(swarm_size)
        ]

        # Build initial informant network for chosen topology
        self.informants = Informants(
            num_particles=swarm_size,
            topology=topology,
            num_informants=num_informants,
            refresh_rate=refresh_rate,
            rng=self.rng,
        )

        # Global best solution across the entire swarm
        self.global_best_pos: Optional[np.ndarray] = None
        self.global_best_fit: float = np.inf
        self._last_improvement_iter: int = 0

    # ---------------------------------------------------------------------

    def _lin_schedule(self, start: float, end: float, t: int) -> float:
        """Return linearly interpolated parameter value at iteration t."""
        if self.max_iters <= 1:
            return end
        frac = t / (self.max_iters - 1)
        return start + frac * (end - start)

    def _params_at(self, t: int) -> tuple[float, float, float]:
        """Return chi, c1, c2 at iteration t (use schedules if provided)."""
        chi = self._lin_schedule(*self.schedule_chi, t) if self.schedule_chi else self.base_chi
        c1 = self._lin_schedule(*self.schedule_c1, t) if self.schedule_c1 else self.base_c1
        c2 = self._lin_schedule(*self.schedule_c2, t) if self.schedule_c2 else self.base_c2
        return chi, c1, c2

    # ---------------------------------------------------------------------

    def _maybe_stagnation_reset(self, t: int) -> None:
        """
        Reset some of the worst particles if the swarm hasn't improved for a while.
        This is a "going further" extension – not part of the strict A39 pseudocode.
        """
        if self.stagnation_iters is None or self.global_best_pos is None:
            return

        # If no improvement for 'stagnation_iters' iterations, reinitialise part of the swarm
        if (t - self._last_improvement_iter) >= self.stagnation_iters:
            fits = np.array([p.best_fitness for p in self.particles])
            order = np.argsort(-fits)  # worst first
            
            lows, highs = np.array(self.bounds)[:, 0], np.array(self.bounds)[:, 1]

            # Reset top third of worst-performing particles
            for idx in order[: max(1, self.swarm_size // 3)]:
                p = self.particles[idx]
                p.position = self.rng.uniform(lows, highs)  # new random pos
                span = (highs - lows)
                p.velocity = self.rng.uniform(-0.5 * span, 0.5 * span) * 0.1
                p.best_fitness = np.inf
                p.best_position = p.position.copy()

            if self.verbose:
                print(f"[iter {t}] stagnation reset applied.")
            self._last_improvement_iter = t

    # ---------------------------------------------------------------------
    #                           MAIN OPTIMISER
    # ---------------------------------------------------------------------

    def optimise(self) -> tuple[np.ndarray, float]:
        """Run the full PSO loop and return the best-found position and fitness."""

        for t in range(self.max_iters):

            # Optionally refresh informants (only applies for random topology)
            self.informants.maybe_refresh(t)

            # Get appropriate coefficients for this iteration (scheduling)
            chi, c1, c2 = self._params_at(t)

            # A39 L12–L15: Evaluate each particle and update personal/global bests
            for p in self.particles:
                f = self.fitness.evaluate(p.position)

                # Update particle's personal best
                if f < p.best_fitness:
                    p.best_fitness = f
                    p.best_position = p.position.copy()

                # Update global best if necessary
                if f < self.global_best_fit:
                    self.global_best_fit = f
                    self.global_best_pos = p.position.copy()
                    self._last_improvement_iter = t

            # A39 L16–L25: Velocity update using personal and informant bests
            for i, p in enumerate(self.particles):
                pbest = p.best_position

                # Retrieve this particle's informants
                inf_ids = self.informants.informant_list[i]

                # Find best informant among the group
                lbest = pbest
                lfit = p.best_fitness
                for j in inf_ids:
                    if self.particles[j].best_fitness < lfit:
                        lfit = self.particles[j].best_fitness
                        lbest = self.particles[j].best_position

                # Random coefficients for stochastic update components
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)

                # Update velocity using constriction-based formula
                p.update_velocity_constriction(
                    chi=chi, c1=c1, c2=c2, r1=r1, r2=r2,
                    pbest=pbest, lbest=lbest,
                    gbest=self.global_best_pos, c3=self.c3
                )

            # A39 L26: Update particle positions
            for p in self.particles:
                p.update_position(step_scale=self.step_scale)

            # Apply stagnation reset extension
            self._maybe_stagnation_reset(t)

            # Verbose logging every 10 iterations
            if self.verbose and (t % 10 == 0 or t == self.max_iters - 1):
                fitness_values = [p.best_fitness for p in self.particles]
                mean_fitness = np.mean(fitness_values)
                print(f"iter {t+1:4d}/{self.max_iters} | best = {self.global_best_fit:.6f} | mean = {mean_fitness:.6f}")

        # A39 L28: Return the best position found
        assert self.global_best_pos is not None
        return self.global_best_pos.copy(), float(self.global_best_fit)