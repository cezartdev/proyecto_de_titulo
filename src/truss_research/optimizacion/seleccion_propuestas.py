"""Selección de propuestas usando solo información disponible antes de la consulta."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from truss_research.optimizacion.busqueda import EvolutionSearch
from truss_research.optimizacion.entrenamiento_propuestas import (
    ProposalEncoder,
    choose_proposal,
)
from truss_research.optimizacion.estrategias_busqueda import SearchStrategy
from truss_research.optimizacion.politica import CandidateQNetwork


class ProposalSelector(Protocol):
    def select(self, search: EvolutionSearch, rng: np.random.Generator) -> int: ...


@dataclass(frozen=True)
class RandomProposalSelector:
    def select(self, search: EvolutionSearch, rng: np.random.Generator) -> int:
        assert search.batch is not None
        return int(rng.integers(len(search.batch.features)))


@dataclass(frozen=True)
class GreedyProposalSelector:
    def select(self, search: EvolutionSearch, rng: np.random.Generator) -> int:
        assert search.batch is not None and search.batch.cheap_values is not None
        gains = [
            search.context.space.hypervolume(np.vstack((search.responses, value)))
            for value in search.batch.cheap_values
        ]
        return int(np.argmax(gains))


@dataclass(frozen=True)
class FrozenProposalSelector:
    components: tuple[CandidateQNetwork, ProposalEncoder]

    def select(self, search: EvolutionSearch, rng: np.random.Generator) -> int:
        network, encoder = self.components
        return choose_proposal(network, encoder.encode(search))


def proposal_selector(
    strategy: SearchStrategy, policy: tuple[CandidateQNetwork, ProposalEncoder] | None
) -> ProposalSelector:
    if strategy.learned != (policy is not None):
        raise ValueError("La estrategia aprendida requiere una política congelada")
    if policy is not None:
        return FrozenProposalSelector(policy)
    selectors: dict[str, ProposalSelector] = {
        "random": RandomProposalSelector(),
        "greedy": GreedyProposalSelector(),
    }
    return selectors[strategy.selector]
