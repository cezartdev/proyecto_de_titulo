"""Generadores sin acceso al evaluador privado; supervivencia Pareto compartida."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2

from truss_research.cerchas.schemas import (
    DESIGN_BOUNDS,
    INPUT_COLUMNS,
    INTEGER_COLUMNS,
    WEB_CONFIG_VALUES,
)
from truss_research.optimizacion.estrategias_busqueda import (
    MIN_DIFFERENTIAL_POPULATION,
    GeneratorName,
    SearchParameters,
)
from truss_research.surrogates.configuracion import FloatArray

DIFFERENTIAL_DONORS = 3


@dataclass(frozen=True)
class GeneratedBatch:
    features: FloatArray
    lineage: tuple[tuple[int, ...], ...]
    origin: str


class ProposalGenerator(Protocol):
    def generate(
        self, algorithm: NSGA2, request: tuple[int, np.random.Generator]
    ) -> GeneratedBatch: ...


@dataclass(frozen=True)
class GeneticGenerator:
    def generate(
        self, algorithm: NSGA2, request: tuple[int, np.random.Generator]
    ) -> GeneratedBatch:
        count, rng = request
        offspring = algorithm.mating.do(
            algorithm.problem,
            algorithm.pop,
            count,
            random_state=rng,
            algorithm=algorithm,
        )
        values = np.array(
            [[item.X[name] for name in INPUT_COLUMNS] for item in offspring],
            dtype=float,
        ).reshape(-1, len(INPUT_COLUMNS))
        return GeneratedBatch(
            values, tuple(() for _ in values), "mixed_crossover_mutation"
        )


@dataclass(frozen=True)
class DifferentialGenerator:
    parameters: SearchParameters

    def generate(
        self, algorithm: NSGA2, request: tuple[int, np.random.Generator]
    ) -> GeneratedBatch:
        count, rng = request
        population = np.array(
            [[item.X[name] for name in INPUT_COLUMNS] for item in algorithm.pop],
            dtype=float,
        )
        if len(population) < MIN_DIFFERENTIAL_POPULATION:
            raise ValueError(
                "La variación diferencial requiere al menos cuatro individuos"
            )
        rows: list[FloatArray] = []
        lineage: list[tuple[int, ...]] = []
        lower = np.array([DESIGN_BOUNDS[name][0] for name in INPUT_COLUMNS])
        upper = np.array([DESIGN_BOUNDS[name][1] for name in INPUT_COLUMNS])
        numeric = np.arange(1, len(INPUT_COLUMNS))
        for _ in range(count):
            target = int(rng.integers(len(population)))
            donors = rng.choice(
                np.delete(np.arange(len(population)), target),
                DIFFERENTIAL_DONORS,
                replace=False,
            )
            first, second, third = population[donors]
            mutant = first.copy()
            mutant[numeric] += self.parameters.differential_weight * (
                second[numeric] - third[numeric]
            )
            mutant[0] = population[int(rng.choice(donors)), 0]
            if rng.random() < 1 / len(INPUT_COLUMNS):
                mutant[0] = rng.choice(WEB_CONFIG_VALUES)
            mask = (
                rng.random(len(INPUT_COLUMNS)) < self.parameters.differential_crossover
            )
            mask[int(rng.choice(numeric))] = True
            trial = np.clip(np.where(mask, mutant, population[target]), lower, upper)
            for index, name in enumerate(INPUT_COLUMNS):
                if name in INTEGER_COLUMNS:
                    trial[index] = np.rint(trial[index])
            rows.append(trial)
            lineage.append((target, *(int(index) for index in donors)))
        return GeneratedBatch(
            np.array(rows), tuple(lineage), "mixed_differential_rand_1_bin"
        )


def proposal_generator(
    name: GeneratorName, parameters: SearchParameters
) -> ProposalGenerator:
    generators: dict[GeneratorName, ProposalGenerator] = {
        "genetic": GeneticGenerator(),
        "differential": DifferentialGenerator(parameters),
    }
    return generators[name]
