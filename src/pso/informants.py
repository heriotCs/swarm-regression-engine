# Informants.py
# Informant networks for Algorithm 39 (Luke). Supports several topologies + periodic refresh.
from __future__ import annotations
import numpy as np
from typing import List, Literal

Topology = Literal["random", "ring", "fully_informed"]

class Informants:
    """
    Maintains, and periodically refreshes, the informant graph.
    - random: for each particle, sample k distinct informants (including itself).
    - ring: each particle informed by itself and its two neighbors (k is ignored).
    - fully_informed: everyone is informed by everyone (k ignored).
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
        self.refresh_rate = max(1, refresh_rate)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.informant_list: List[np.ndarray] = []
        self._generate()

    def _generate(self) -> None:
        if self.topology == "random":
            self.informant_list = [
                np.unique(
                    self.rng.choice(self.n, size=min(self.k, self.n), replace=False)
                )
                for _ in range(self.n)
            ]
            # Ensure self is an informant (as per Luke’s description, the particle is one of its own informants)
            for i in range(self.n):
                if i not in self.informant_list[i]:
                    # replace a random index with self
                    idx = self.rng.integers(len(self.informant_list[i]))
                    self.informant_list[i][idx] = i

        elif self.topology == "ring":
            self.informant_list = [
                np.array([(i - 1) % self.n, i, (i + 1) % self.n], dtype=int)
                for i in range(self.n)
            ]

        elif self.topology == "fully_informed":
            self.informant_list = [np.arange(self.n, dtype=int) for _ in range(self.n)]

        else:
            raise ValueError(f"Unknown topology: {self.topology}")

    def maybe_refresh(self, iteration: int) -> None:
        # Alg39 has a notion of informants; the “refresh” is a modern practice we include as an extension.
        if self.topology == "random" and (iteration % self.refresh_rate == 0):
            self._generate()
