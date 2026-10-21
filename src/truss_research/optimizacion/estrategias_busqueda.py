"""Matriz explícita de generadores y selectores del experimento evolutivo."""

from dataclasses import dataclass, fields
from math import isfinite
from typing import Literal

from truss_research.datos.procesados import JsonValue

type GeneratorName = Literal["genetic", "differential"]
type SelectorName = Literal["random", "greedy", "double_dqn", "contextual_bandit"]
MIN_DIFFERENTIAL_POPULATION = 4


@dataclass(frozen=True)
class SearchStrategy:
    name: str
    generator: GeneratorName = "genetic"
    selector: SelectorName = "random"
    assisted: bool = False
    single_proposal: bool = False

    @property
    def learned(self) -> bool:
        return self.selector in ("double_dqn", "contextual_bandit")


STRATEGY_DEFINITIONS = {
    row.name: row
    for row in (
        SearchStrategy("evolution", single_proposal=True),
        SearchStrategy("random"),
        SearchStrategy("greedy", selector="greedy", assisted=True),
        SearchStrategy("rl", selector="double_dqn"),
        SearchStrategy("rl_assisted", selector="double_dqn", assisted=True),
        SearchStrategy("contextual_bandit", selector="contextual_bandit"),
        SearchStrategy(
            "contextual_bandit_assisted", selector="contextual_bandit", assisted=True
        ),
        SearchStrategy("de", generator="differential", single_proposal=True),
        SearchStrategy(
            "de_assisted", generator="differential", selector="greedy", assisted=True
        ),
    )
}


@dataclass(frozen=True)
class SearchParameters:
    strategies: tuple[str, ...] = tuple(STRATEGY_DEFINITIONS)
    differential_weight: float = 0.5
    differential_crossover: float = 0.9
    target_fraction: float = 0.9

    def __post_init__(self) -> None:
        if (
            not self.strategies
            or len(set(self.strategies)) != len(self.strategies)
            or any(name not in STRATEGY_DEFINITIONS for name in self.strategies)
        ):
            raise ValueError("Estrategias de búsqueda desconocidas, vacías o repetidas")
        if (
            not all(
                type(value) in (int, float) and isfinite(value)
                for value in (
                    self.differential_weight,
                    self.differential_crossover,
                    self.target_fraction,
                )
            )
            or not 0 < self.differential_weight <= 2
            or not 0 <= self.differential_crossover <= 1
            or not 0 < self.target_fraction <= 1
        ):
            raise ValueError("Parámetros diferenciales o umbral objetivo inválidos")

    @property
    def definitions(self) -> tuple[SearchStrategy, ...]:
        return tuple(STRATEGY_DEFINITIONS[name] for name in self.strategies)

    @property
    def policies(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.definitions if row.learned)


def read_search_parameters(payload: JsonValue) -> SearchParameters:
    if not isinstance(payload, dict) or set(payload) - {
        part.name for part in fields(SearchParameters)
    }:
        raise ValueError("Parámetros de búsqueda desconocidos")
    values = dict(payload)
    names = values.pop("strategies", tuple(STRATEGY_DEFINITIONS))
    if not isinstance(names, (list, tuple)) or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError("search.strategies debe ser una lista de nombres")
    return SearchParameters(strategies=tuple(names), **values)
