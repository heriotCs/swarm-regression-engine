# particle.py
# This file defines the Particle class, which stores everything related to a single PSO particle.
# Each particle keeps track of its position, velocity, and its personal-best solution.

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Literal

BoundaryMode = Literal["clamp", "reflect", "wrap"]


class Particle:
    def __init__(
        self,
        dim: int,
        bounds: List[Tuple[float, float]],
        rng: np.random.Generator,
        vmax: float | None = None,
        boundary_mode: BoundaryMode = "reflect",
    ):
        # Dimension of the search vector (i.e., number of ANN parameters)
        self.dim = dim
        
        # Bounds for each parameter value
        self.bounds = np.array(bounds, dtype=float)
        
        # Random number generator for reproducibility
        self.rng = rng
        
        # How the particle behaves when it goes out of bounds
        self.boundary_mode = boundary_mode
        
        # Optional max velocity
        self.vmax = vmax

        lows, highs = self.bounds[:, 0], self.bounds[:, 1]

        # A39 L9: randomly initialise position inside the search space
        self.position = self.rng.uniform(lows, highs)

        # Initial velocity is small relative to the parameter range
        # This stops particles from jumping around too aggressively early on
        self.velocity = self.rng.uniform(-np.abs(highs - lows), np.abs(highs - lows)) * 0.1

        # Record of the best position this particle has discovered so far
        self.best_position = self.position.copy()
        self.best_fitness = np.inf

    # ------------------------------------------------------------------

    def _apply_velocity_limits(self):
        """Limit velocity magnitude so particles don't explode numerically."""
        if self.vmax is not None:
            # If user specifies vmax, enforce that cap
            np.clip(self.velocity, -self.vmax, self.vmax, out=self.velocity)
        else:
            # Otherwise use a soft limit: 10% of the search range for each dimension
            span = self.bounds[:, 1] - self.bounds[:, 0]
            v_max = 0.1 * span
            np.clip(self.velocity, -v_max, v_max, out=self.velocity)

    # ------------------------------------------------------------------

    def _apply_bounds(self):
        """Apply the selected boundary-handling strategy to keep the particle inside valid ranges."""
        lows, highs = self.bounds[:, 0], self.bounds[:, 1]

        if self.boundary_mode == "clamp":
            # If out of bounds, clamp to boundary and zero velocity to stop it bouncing
            out_low = self.position < lows
            out_high = self.position > highs
            self.position = np.minimum(np.maximum(self.position, lows), highs)
            self.velocity[out_low | out_high] = 0.0

        elif self.boundary_mode == "reflect":
            # Reflective boundary: particle bounces back like a mirror
            for i in range(self.dim):
                if self.position[i] < lows[i]:
                    # Reflect below lower bound
                    self.position[i] = lows[i] + (lows[i] - self.position[i])
                    self.velocity[i] *= -1
                if self.position[i] > highs[i]:
                    # Reflect above upper bound
                    self.position[i] = highs[i] - (self.position[i] - highs[i])
                    self.velocity[i] *= -1
                # Safety clamp just in case floating-point issues push it out again
                self.position[i] = min(max(self.position[i], lows[i]), highs[i])

        elif self.boundary_mode == "wrap":
            # Wrap-around: values exceeding the bounds re-enter on the opposite side
            span = highs - lows
            self.position = lows + np.mod(self.position - lows, span)

        else:
            raise ValueError(f"Unknown boundary mode: {self.boundary_mode}")

    # ------------------------------------------------------------------

    def update_velocity_constriction(
        self,
        chi: float,  # constriction factor (Luke A39)
        c1: float,
        c2: float,
        r1: np.ndarray,
        r2: np.ndarray,
        pbest: np.ndarray,
        lbest: np.ndarray,
        gbest: np.ndarray | None,
        c3: float = 0.0,
    ):
        """Apply the constricted PSO velocity update (A39)."""
        
        # Cognitive component: how strongly particle moves towards its own best solution
        cognitive = c1 * r1 * (pbest - self.position)
        
        # Social component: attraction towards the best informant in the neighbourhood
        social = c2 * r2 * (lbest - self.position)
        
        # Optional global term if gbest is provided (not always used)
        global_t = 0.0 if (gbest is None or c3 == 0.0) else c3 * self.rng.random(self.dim) * (gbest - self.position)
        
        # A39 L24: overall constricted update rule
        self.velocity = chi * (self.velocity + cognitive + social + global_t)
        
        # Ensure updated velocity doesn't exceed allowed limits
        self._apply_velocity_limits()

    # ------------------------------------------------------------------

    def update_position(self, step_scale: float = 1.0):
        """A39 L26: move particle according to current velocity."""
        self.position = self.position + step_scale * self.velocity
        
        # After moving, enforce boundary behaviour
        self._apply_bounds()