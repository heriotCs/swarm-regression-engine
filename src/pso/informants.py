# Informants.py
# This module manages the informant network used in PSO. Each particle has a set of
# “informants” whose best-known positions it is allowed to consider during updates.

from __future__ import annotations
import numpy as np
from typing import List, Literal

Topology = Literal["random", "ring", "fully_informed"]

class Informants:
    """
    This class builds and stores the informant structure for the swarm.
    Depending on the topology, a particle may be influenced by:
      - a random subset of neighbours,
      - its two neighbours in a ring layout,
      - or the entire swarm (fully informed).
    The network can also refresh periodically for extra exploration.
    """
    def __init__(
        self,
        num_particles: int,
        topology: Topology = "random",
        num_informants: int = 3,
        refresh_rate: int = 5,
        rng: np.random.Generator | None = None,
    ):
        self.n = num_particles
        self.topology = topology
        self.k = num_informants
        # How often the informants are regenerated (mainly for random topology)
        self.refresh_rate = max(1, refresh_rate)
        # Use provided RNG or create a new one (helps with reproducibility)
        self.rng = rng if rng is not None else np.random.default_rng()

        # This will store a list of arrays, where each array is the informants of that particle
        self.informant_list: List[np.ndarray] = []
        self._generate()   # build the initial informant network

    # ------------------------------------------------------------------

    def _generate(self) -> None:
        """Create the informant lists according to the chosen topology."""
        if self.topology == "random":
            # For random topology, each particle samples k distinct informants.
            self.informant_list = [
                np.unique(
                    self.rng.choice(self.n, size=min(self.k, self.n), replace=False)
                )
                for _ in range(self.n)
            ]
            # Ensure the particle is always included as one of its own informants,
            # which aligns with the description in Luke's algorithm.
            for i in range(self.n):
                if i not in self.informant_list[i]:
                    # Replace a random index with the particle itself
                    idx = self.rng.integers(len(self.informant_list[i]))
                    self.informant_list[i][idx] = i

        elif self.topology == "ring":
            # In ring topology, each particle is informed by itself and its two immediate neighbours.
            self.informant_list = [
                np.array([(i - 1) % self.n, i, (i + 1) % self.n], dtype=int)
                for i in range(self.n)
            ]

        elif self.topology == "fully_informed":
            # Fully informed means every particle receives information from the whole swarm.
            self.informant_list = [np.arange(self.n, dtype=int) for _ in range(self.n)]

        else:
            raise ValueError(f"Unknown topology: {self.topology}")

    # ------------------------------------------------------------------

    def maybe_refresh(self, iteration: int) -> None:
        """
        Refresh informants for random topology every 'refresh_rate' iterations.
        This isn't in Algorithm 39 directly, but it's a common extension to help
        avoid stagnation in the swarm.
        """
        if self.topology == "random" and (iteration % self.refresh_rate == 0):
            self._generate()