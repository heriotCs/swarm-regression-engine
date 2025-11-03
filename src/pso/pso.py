# pso.py
# Full Algorithm 39 (with informants) + "Going Further" extensions.
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Literal, Optional

import os
import sys

# Add parent directory to path
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
        self.fitness = fitness
        self.dim = dim
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.max_iters = max_iters
        self.verbose = verbose

        self.base_chi = chi
        self.base_c1 = c1
        self.base_c2 = c2
        self.c3 = c3
        self.step_scale = step_scale

        self.schedule_c1 = schedule_c1
        self.schedule_c2 = schedule_c2
        self.schedule_chi = schedule_chi

        self.vmax = vmax
        self.boundary_mode = boundary_mode
        self.stagnation_iters = stagnation_iters

        self.rng = np.random.default_rng(seed)

        # Create swarm (A39 L7–L9)
        self.particles: List[Particle] = [
            Particle(dim, bounds, rng=self.rng, vmax=vmax, boundary_mode=boundary_mode)
            for _ in range(swarm_size)
        ]

        # Informants (Algorithm 39 notion)
        self.informants = Informants(
            num_particles=swarm_size,
            topology=topology,
            num_informants=num_informants,
            refresh_rate=refresh_rate,
            rng=self.rng,
        )

        # Best-so-far (A39 L10 Best ← ☐)
        self.global_best_pos: Optional[np.ndarray] = None
        self.global_best_fit: float = np.inf
        self._last_improvement_iter: int = 0

    # ---------------------------------------------------------------------

    def _lin_schedule(self, start: float, end: float, t: int) -> float:
        if self.max_iters <= 1:
            return end
        frac = t / (self.max_iters - 1)
        return start + frac * (end - start)

    def _params_at(self, t: int) -> tuple[float, float, float]:
        chi = self._lin_schedule(*self.schedule_chi, t) if self.schedule_chi else self.base_chi
        c1 = self._lin_schedule(*self.schedule_c1, t) if self.schedule_c1 else self.base_c1
        c2 = self._lin_schedule(*self.schedule_c2, t) if self.schedule_c2 else self.base_c2
        return chi, c1, c2

    def _maybe_stagnation_reset(self, t: int) -> None:
        if self.stagnation_iters is None or self.global_best_pos is None:
            return
        if (t - self._last_improvement_iter) >= self.stagnation_iters:
            fits = np.array([p.best_fitness for p in self.particles])
            order = np.argsort(-fits)  # worst first
            lows, highs = np.array(self.bounds)[:, 0], np.array(self.bounds)[:, 1]
            for idx in order[: max(1, self.swarm_size // 3)]:
                p = self.particles[idx]
                p.position = self.rng.uniform(lows, highs)
                span = (highs - lows)
                p.velocity = self.rng.uniform(-0.5 * span, 0.5 * span) * 0.1
                p.best_fitness = np.inf
                p.best_position = p.position.copy()
            if self.verbose:
                print(f"[iter {t}] stagnation reset applied.")
            self._last_improvement_iter = t

    # ---------------------------------------------------------------------
    #                           MAIN OPTIMIZER
    # ---------------------------------------------------------------------

    def optimise(self) -> tuple[np.ndarray, float]:
        for t in range(self.max_iters):
            # Optional: refresh informants
            self.informants.maybe_refresh(t)

            chi, c1, c2 = self._params_at(t)

            # A39 L12–L15: Evaluate fitness and update personal/global bests
            for p in self.particles:
                f = self.fitness.evaluate(p.position)
                if f < p.best_fitness:
                    p.best_fitness = f
                    p.best_position = p.position.copy()

                if f < self.global_best_fit:
                    self.global_best_fit = f
                    self.global_best_pos = p.position.copy()
                    self._last_improvement_iter = t

            # A39 L16–L25: Velocity update
            for i, p in enumerate(self.particles):
                pbest = p.best_position
                inf_ids = self.informants.informant_list[i]
                lbest = pbest
                lfit = p.best_fitness
                for j in inf_ids:
                    if self.particles[j].best_fitness < lfit:
                        lfit = self.particles[j].best_fitness
                        lbest = self.particles[j].best_position

                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                p.update_velocity_constriction(
                    chi=chi, c1=c1, c2=c2, r1=r1, r2=r2,
                    pbest=pbest, lbest=lbest, gbest=self.global_best_pos, c3=self.c3
                )

            # A39 L26: Update positions
            for p in self.particles:
                p.update_position(step_scale=self.step_scale)

            # Optional: stagnation reset
            self._maybe_stagnation_reset(t)

            # ------------------ 💡 Mean Fitness Printing ------------------
            if self.verbose and (t % 10 == 0 or t == self.max_iters - 1):
                fitness_values = [p.best_fitness for p in self.particles]
                mean_fitness = np.mean(fitness_values)
                print(f"iter {t+1:4d}/{self.max_iters} | best = {self.global_best_fit:.6f} | mean = {mean_fitness:.6f}")

        # A39 L28: return global best
        assert self.global_best_pos is not None
        return self.global_best_pos.copy(), float(self.global_best_fit)