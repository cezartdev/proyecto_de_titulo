"""NSGA-II de pymoo adaptado a consultas de un catálogo de diseños registrados."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.core.individual import Individual
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableMating,
    MixedVariableSampling,
)
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.variable import Choice, Integer, Real
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.selection.tournament import TournamentSelection

from truss_research.cerchas.schemas import (
    DESIGN_BOUNDS,
    INPUT_COLUMNS,
    INTEGER_COLUMNS,
    WEB_CONFIG_VALUES,
)
from truss_research.datos.distancias import design_distances
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.pareto import OBJECTIVE_COUNT, ObjectiveSpace
from truss_research.optimizacion.contratos import (
    EpisodeObservation,
    EvolutionParameters,
)
from truss_research.optimizacion.estrategias import available_actions
from truss_research.surrogates.configuracion import FloatArray

type DesignGenes = dict[str, int | float]


@dataclass(frozen=True)
class EvolutionContext:
    space: ObjectiveSpace
    feature_ranges: tuple[float, ...]
    seed: int
    parameters: EvolutionParameters
    assisted: bool = False


def mixed_problem() -> Problem:
    variables: dict[str, Choice | Integer | Real] = {
        name: Integer(bounds=DESIGN_BOUNDS[name])
        if name in INTEGER_COLUMNS
        else Real(bounds=DESIGN_BOUNDS[name])
        for name in INPUT_COLUMNS
    }
    variables[INPUT_COLUMNS[0]] = Choice(options=WEB_CONFIG_VALUES)
    return Problem(vars=variables, n_obj=OBJECTIVE_COUNT, n_ieq_constr=2)


def mixed_mating(parameters: EvolutionParameters) -> MixedVariableMating:
    probability = 1 / len(INPUT_COLUMNS)
    return MixedVariableMating(
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover={
            Choice: UX(prob=parameters.crossover_probability),
            Real: SBX(
                prob=parameters.crossover_probability, eta=parameters.crossover_eta
            ),
            Integer: SBX(
                prob=parameters.crossover_probability,
                eta=parameters.crossover_eta,
                vtype=float,
                repair=RoundingRepair(),
            ),
        },
        mutation={
            Choice: ChoiceRandomMutation(prob=1.0, prob_var=probability),
            Real: PM(prob=1.0, prob_var=probability, eta=parameters.mutation_eta),
            Integer: PM(
                prob=1.0,
                prob_var=probability,
                eta=parameters.mutation_eta,
                vtype=float,
                repair=RoundingRepair(),
            ),
        },
        eliminate_duplicates=MixedVariableDuplicateElimination(),
        n_max_iterations=parameters.max_mating_attempts,
    )


def design_genes(values: FloatArray) -> DesignGenes:
    return {
        name: int(value) if name in INTEGER_COLUMNS else float(value)
        for name, value in zip(INPUT_COLUMNS, values, strict=True)
    }


def evaluated_population(
    observation: EpisodeObservation, context: EvolutionContext
) -> Population:
    known = np.flatnonzero(~observation.available)
    responses = observation.observed[known]
    constraints = np.column_stack(
        (
            responses[:, 1] - context.space.limits.utilization_limit,
            np.zeros(len(responses))
            if context.space.limits.displacement_limit_m is None
            else responses[:, 2] / context.space.limits.displacement_limit_m - 1,
        )
    )
    return Population.new(
        "X",
        np.array(
            [design_genes(row) for row in observation.features[known]], dtype=object
        ),
        "F",
        context.space.transform(responses),
        "G",
        constraints,
    )


class EvolutionStrategy:
    def __init__(self, context: EvolutionContext) -> None:
        if (
            len(context.feature_ranges) != len(INPUT_COLUMNS)
            or not np.isfinite(context.feature_ranges).all()
            or min(context.feature_ranges) < 0
        ):
            raise ValueError(
                "Se requieren siete rangos de entradas ajustados en entrenamiento"
            )
        self.context = context
        self._rng = np.random.default_rng(context.seed)
        self._algorithm: NSGA2 | None = None
        self._pending: tuple[int, Individual] | None = None
        self.diagnostics: dict[str, JsonValue] = {}

    def initialize(self, observation: EpisodeObservation) -> int:
        available = available_actions(observation)
        if not self.context.parameters.guided_initialization:
            return int(self._rng.choice(available))
        known = observation.observed[~observation.available]
        diversity = design_distances(
            observation.features[available],
            observation.features[~observation.available],
            self.context.feature_ranges,
        ).min(axis=1)
        if not self.context.assisted:
            return int(available[np.argmax(diversity)])
        gains = np.array(
            [
                self.context.space.hypervolume(
                    np.vstack((known, observation.predictions[i]))
                )
                for i in available
            ]
        )
        return int(available[np.lexsort((available, -diversity, -gains))[0]])

    def _synchronize(self, observation: EpisodeObservation) -> None:
        if self._algorithm is None:
            self._algorithm = NSGA2(
                pop_size=min(
                    self.context.parameters.population_size, observation.budget
                ),
                n_offsprings=1,
                sampling=MixedVariableSampling(),
                mating=mixed_mating(self.context.parameters),
                eliminate_duplicates=MixedVariableDuplicateElimination(),
            )
            self._algorithm.setup(mixed_problem(), seed=self.context.seed)
            self._algorithm.tell(
                infills=evaluated_population(observation, self.context)
            )
            return
        if self._pending is None:
            raise ValueError("Falta la consulta evolutiva pendiente")
        index, individual = self._pending
        if observation.available[index]:
            raise ValueError(
                "Revelar la respuesta seleccionada antes de avanzar NSGA-II"
            )
        response = observation.observed[index]
        individual.F = self.context.space.transform(response[None, :])[0]
        individual.G = np.array(
            [
                response[1] - self.context.space.limits.utilization_limit,
                0.0
                if self.context.space.limits.displacement_limit_m is None
                else response[2] / self.context.space.limits.displacement_limit_m - 1,
            ]
        )
        self._algorithm.tell(infills=Population.create(individual))

    def project(
        self,
        proposal: DesignGenes,
        context: tuple[EpisodeObservation, npt.NDArray[np.int64]],
    ) -> tuple[int, float]:
        observation, candidates = context
        values = np.array([proposal[name] for name in INPUT_COLUMNS], dtype=float)
        distances = design_distances(
            observation.features[candidates],
            values[None, :],
            self.context.feature_ranges,
        )[:, 0]
        best = int(np.argmin(distances))
        return int(candidates[best]), float(distances[best])

    def choose(self, observation: EpisodeObservation) -> int:
        available = available_actions(observation)
        if observation.queries < min(
            self.context.parameters.population_size, observation.budget
        ):
            self.diagnostics = {
                "evolution_stage": "initial",
                "proposal_count": 0,
                "projection_distance": 0.0,
                "projection_collision_count": 0,
                "random_fallback": False,
            }
            return self.initialize(observation)
        self._synchronize(observation)
        algorithm = self._algorithm
        assert algorithm is not None
        count = min(
            len(available),
            self.context.parameters.proposal_pool if self.context.assisted else 1,
        )
        offspring = algorithm.mating.do(
            algorithm.problem,
            algorithm.pop,
            count,
            random_state=self._rng,
            algorithm=algorithm,
        )
        proposals = [cast(DesignGenes, item.X) for item in offspring]
        fallback = not proposals
        if fallback:
            proposals = [
                design_genes(observation.features[int(self._rng.choice(available))])
            ]
        projected: list[tuple[int, float]] = []
        original_available = available.copy()
        collisions = 0
        for proposal in proposals:
            nearest, _ = self.project(proposal, (observation, original_available))
            collisions += int(nearest not in available)
            action, distance = self.project(proposal, (observation, available))
            projected.append((action, distance))
            available = available[available != action]
        scores = (
            [
                self.context.space.hypervolume(
                    np.vstack(
                        (
                            observation.observed[~observation.available],
                            observation.predictions[index],
                        )
                    )
                )
                for index, _ in projected
            ]
            if self.context.assisted
            else [0.0]
        )
        best = int(np.argmax(scores))
        action, distance = projected[best]
        self._pending = (
            action,
            Individual(X=design_genes(observation.features[action])),
        )
        self.diagnostics = {
            "evolution_stage": "offspring",
            "proposal_count": len(projected),
            "projection_distance": distance,
            "projection_collision_count": collisions,
            "random_fallback": fallback,
            **{f"proposal_{name}": value for name, value in proposals[best].items()},
        }
        return action
