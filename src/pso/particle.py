# particle.py
# Encapsulates per-particle state and updates (position/velocity + boundary handling).
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
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)
        self.rng = rng
        self.boundary_mode = boundary_mode
        self.vmax = vmax

        lows, highs = self.bounds[:, 0], self.bounds[:, 1]

        # A39 L9: create random particle (position)
        self.position = self.rng.uniform(lows, highs)

        # small random initial velocity (10% of range)
        self.velocity = self.rng.uniform(-np.abs(highs - lows), np.abs(highs - lows)) * 0.1

        # Personal best (A39 L13–L15 updates happen in PSO loop)
        self.best_position = self.position.copy()
        self.best_fitness = np.inf

    # ------------------------------------------------------------------

    def _apply_velocity_limits(self):
        """Clamp velocity if vmax is set, otherwise skip."""
        if self.vmax is not None:
            np.clip(self.velocity, -self.vmax, self.vmax, out=self.velocity)
        else:
            # optional dynamic clamp (~10% of span) for stability
            span = self.bounds[:, 1] - self.bounds[:, 0]
            v_max = 0.1 * span
            np.clip(self.velocity, -v_max, v_max, out=self.velocity)

    # ------------------------------------------------------------------

    def _apply_bounds(self):
        """Handle boundary conditions for particle position."""
        lows, highs = self.bounds[:, 0], self.bounds[:, 1]

        if self.boundary_mode == "clamp":
            out_low = self.position < lows
            out_high = self.position > highs
            self.position = np.minimum(np.maximum(self.position, lows), highs)
            self.velocity[out_low | out_high] = 0.0

        elif self.boundary_mode == "reflect":
            for i in range(self.dim):
                if self.position[i] < lows[i]:
                    self.position[i] = lows[i] + (lows[i] - self.position[i])
                    self.velocity[i] *= -1
                if self.position[i] > highs[i]:
                    self.position[i] = highs[i] - (self.position[i] - highs[i])
                    self.velocity[i] *= -1
                # second pass safety
                self.position[i] = min(max(self.position[i], lows[i]), highs[i])

        elif self.boundary_mode == "wrap":
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
        c3: float = 0.0,  # optional global term (A39 δ)
    ):
        """Update particle velocity according to constricted PSO formula."""
        # A39 L24: velocity update
        cognitive = c1 * r1 * (pbest - self.position)
        social = c2 * r2 * (lbest - self.position)
        global_t = 0.0 if (gbest is None or c3 == 0.0) else c3 * self.rng.random(self.dim) * (gbest - self.position)
        self.velocity = chi * (self.velocity + cognitive + social + global_t)
        self._apply_velocity_limits()

    # ------------------------------------------------------------------

    def update_position(self, step_scale: float = 1.0):
        """A39 L26: update position x <- x + e * v."""
        self.position = self.position + step_scale * self.velocity
        self._apply_bounds()