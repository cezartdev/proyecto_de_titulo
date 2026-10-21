"""Búsqueda evolutiva con población persistente y evaluación posterior a la acción."""

from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import cast

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.optimizacion.contratos import EvolutionParameters
from truss_research.optimizacion.entorno import marginal_reward
from truss_research.optimizacion.estrategias_busqueda import (
    MIN_DIFFERENTIAL_POPULATION,
    GeneratorName,
    SearchParameters,
)
from truss_research.optimizacion.evaluadores import (
    Evaluation,
    SearchEvaluator,
    UnsupportedProposal,
)
from truss_research.optimizacion.evolucion import (
    design_genes,
    mixed_mating,
    mixed_problem,
)
from truss_research.optimizacion.generacion import proposal_generator
from truss_research.optimizacion.propuestas import (
    ProposalParameters,
    TrainingSupport,
    design_identity,
)
from truss_research.surrogates.configuracion import FloatArray

PROPOSAL_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class SearchContext:
    training: FloatArray
    initial_responses: FloatArray
    worlds: tuple[SearchEvaluator, ...]
    support: TrainingSupport
    space: ObjectiveSpace
    reward_scale: float
    parameters: ProposalParameters
    evolution: EvolutionParameters
    archive_size: int
    cheap: SearchEvaluator | None = None
    assisted: bool = False
    validation_world: SearchEvaluator | None = None
    generator: GeneratorName = "genetic"
    search_parameters: SearchParameters = field(default_factory=SearchParameters)
    training_identities: frozenset[tuple[float, ...]] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "training_identities",
            frozenset(design_identity(row) for row in self.training),
        )


@dataclass(frozen=True)
class PendingBatch:
    features: FloatArray
    distances: FloatArray
    cheap_values: FloatArray | None = None
    lineage: tuple[tuple[int, ...], ...] = ()
    origin: str = "mixed_crossover_mutation"


@dataclass(frozen=True)
class SearchSnapshot:
    budget: int
    rng_state: dict[str, JsonValue]
    evaluations: int
    recommendations: int
    duplicates: int
    rejected: int
    reason: str
    seen: tuple[tuple[float, ...], ...]
    archive: FloatArray
    responses: FloatArray
    volumes: tuple[float, ...]
    batch: PendingBatch | None
    generated: int
    regeneration_rounds: int
    world_index: int
    population_features: FloatArray
    population_responses: FloatArray
    cheap_evaluations: int
    unique_evaluations: int
    last_evaluation: Evaluation | None
    population: Population
    member_predictions: int
    cheap_member_predictions: int
    generation_seconds: float
    evaluation_seconds: float
    cheap_seconds: float


class EvolutionSearch:
    def __init__(self, context: SearchContext, episode: tuple[int, int]) -> None:
        budget, seed = episode
        size = context.evolution.population_size
        if not size < budget <= context.parameters.max_evaluations:
            raise ValueError("El presupuesto total debe superar la población inicial")
        if (
            len(context.training) < size
            or not context.worlds
            or context.reward_scale <= 0
        ):
            raise ValueError("Faltan semillas, mundos o referencia de entrenamiento")
        self.context, self.budget = context, budget
        self.generator = proposal_generator(
            context.generator, context.search_parameters
        )
        if context.generator == "differential" and size < MIN_DIFFERENTIAL_POPULATION:
            raise ValueError(
                "La variación diferencial requiere población mínima de cuatro"
            )
        if context.assisted and context.cheap is None:
            raise ValueError("La selección asistida requiere un predictor auxiliar")
        self.generation_seconds = self.evaluation_seconds = self.cheap_seconds = 0.0
        self.rng = np.random.default_rng(seed)
        self.world_index = int(self.rng.integers(len(context.worlds)))
        indices = self.rng.choice(len(context.training), size, replace=False)
        self.archive = context.training[indices].copy()
        self.responses = context.initial_responses[indices].copy()
        self.evaluations = size
        self.recommendations = 0
        self.duplicates = self.rejected = self.generated = self.regeneration_rounds = 0
        self.cheap_evaluations = 0
        self.member_predictions = self.cheap_member_predictions = 0
        self.unique_evaluations = size
        self.seen = {design_identity(row) for row in self.archive}
        self.known = set(self.seen)
        self.volumes = [0.0] + [
            context.space.hypervolume(self.responses[:number])
            for number in range(1, size + 1)
        ]
        self.batch: PendingBatch | None = None
        self.last_evaluation: Evaluation | None = None
        self.reason = "budget_exhausted"
        self._algorithm = NSGA2(pop_size=size, mating=mixed_mating(context.evolution))
        problem = mixed_problem()
        problem.n_obj = len(context.space.active_dimensions)
        self._algorithm.setup(problem, seed=seed)
        self._algorithm.random_state = self.rng
        self._population(self.archive, self.responses)

    def _population(self, features: FloatArray, responses: FloatArray) -> None:
        limits = self.context.space.limits
        constraints = np.column_stack(
            (
                responses[:, 1] - limits.utilization_limit,
                np.zeros(len(responses))
                if limits.displacement_limit_m is None
                else responses[:, 2] / limits.displacement_limit_m - 1,
            )
        )
        invalid = (responses[:, 0] <= 0) | (responses[:, 1] < 0) | (responses[:, 2] < 0)
        constraints[:, 0] = np.where(invalid, 1.0, constraints[:, 0])
        population = Population.new(
            "X",
            np.array([design_genes(row) for row in features], dtype=object),
            "F",
            self.context.space.transform(responses)[
                :, self.context.space.active_dimensions
            ],
            "G",
            constraints,
        )
        self._algorithm.pop = self._algorithm.survival.do(
            self._algorithm.problem,
            population,
            n_survive=self.context.evolution.population_size,
            random_state=self.rng,
        )
        self.population_features = np.array(
            [[item.X[name] for name in INPUT_COLUMNS] for item in self._algorithm.pop],
            dtype=float,
        )
        self.population_responses = np.array(
            [
                responses[
                    next(
                        i
                        for i, row in enumerate(features)
                        if design_identity(row) == design_identity(value)
                    )
                ]
                for value in self.population_features
            ]
        )

    @property
    def remaining_evaluations(self) -> int:
        return self.budget - self.evaluations

    def prepare(self) -> bool:
        self.batch = None
        if not self.remaining_evaluations:
            return False
        limit = self.context.parameters.max_generated
        if self.generated >= limit:
            self.reason = "generation_budget_exhausted"
            return False
        for _ in range(self.context.parameters.max_regeneration):
            count = min(self.context.parameters.pool_size, limit - self.generated)
            if count <= 0:
                break
            started = perf_counter()
            generated = self.generator.generate(self._algorithm, (count, self.rng))
            proposals = generated.features
            self.generated += len(proposals)
            self.regeneration_rounds += 1
            valid = self.context.support.accepts(proposals)
            self.rejected += int((~valid).sum())
            rows: list[FloatArray] = []
            lineage: list[tuple[int, ...]] = []
            pending: set[tuple[float, ...]] = set()
            forbidden = self.context.training_identities
            for index in np.flatnonzero(valid):
                row = proposals[index]
                identity = design_identity(row)
                if (
                    identity in self.seen
                    or identity in forbidden
                    or identity in pending
                ):
                    self.duplicates += 1
                    continue
                pending.add(identity)
                rows.append(row)
                lineage.append(generated.lineage[index])
            self.generation_seconds += perf_counter() - started
            if not rows:
                continue
            values = np.array(rows)
            cheap = None
            if self.context.assisted and self.context.cheap is not None:
                started = perf_counter()
                estimate = self.context.cheap.evaluate(values)
                self.cheap_seconds += perf_counter() - started
                cheap = estimate.responses
                self.cheap_member_predictions += estimate.member_predictions
                self.cheap_evaluations += len(values)
            self.batch = PendingBatch(
                values,
                self.context.support.distances(values),
                cheap,
                tuple(lineage),
                generated.origin,
            )
            return True
        self.reason = "no_supported_proposals"
        return False

    def choose(self, action: int) -> float:
        if (
            self.batch is None
            or type(action) is not int
            or not 0 <= action < len(self.batch.features)
        ):
            raise ValueError("Acción fuera del lote pendiente")
        proposed = self.batch.features[[action]]
        self.evaluations += 1
        self.recommendations += 1
        self.seen.add(design_identity(proposed[0]))
        started = perf_counter()
        try:
            result = self.context.worlds[self.world_index].evaluate(proposed)
        except UnsupportedProposal as error:
            import logging

            logging.getLogger(__name__).warning(
                "Consulta sin respuesta histórica: %s", error
            )
            self.last_evaluation = None
            self.volumes.append(self.volumes[-1])
            self.batch = None
            return 0.0
        finally:
            self.evaluation_seconds += perf_counter() - started
        self.member_predictions += result.member_predictions
        self.last_evaluation = result
        identity = design_identity(result.features[0])
        if identity not in self.known:
            self.known.add(identity)
            self.unique_evaluations += 1
            self.archive = np.vstack((self.archive, result.features))
            self.responses = np.vstack((self.responses, result.responses))
            self._population(
                np.vstack((self.population_features, result.features)),
                np.vstack((self.population_responses, result.responses)),
            )
        else:
            self.duplicates += 1
        self.seen.add(identity)
        self.volumes.append(self.context.space.hypervolume(self.responses))
        self.batch = None
        return marginal_reward(
            self.volumes[-2],
            self.volumes[-1],
            (self.evaluations, self.budget, self.context.reward_scale),
        )

    def snapshot(self) -> SearchSnapshot:
        return SearchSnapshot(
            self.budget,
            cast(dict[str, JsonValue], self.rng.bit_generator.state),
            self.evaluations,
            self.recommendations,
            self.duplicates,
            self.rejected,
            self.reason,
            tuple(sorted(self.seen)),
            self.archive.copy(),
            self.responses.copy(),
            tuple(self.volumes),
            self.batch,
            self.generated,
            self.regeneration_rounds,
            self.world_index,
            self.population_features.copy(),
            self.population_responses.copy(),
            self.cheap_evaluations,
            self.unique_evaluations,
            self.last_evaluation,
            deepcopy(self._algorithm.pop),
            self.member_predictions,
            self.cheap_member_predictions,
            self.generation_seconds,
            self.evaluation_seconds,
            self.cheap_seconds,
        )

    @classmethod
    def restore(
        cls, context: SearchContext, state: SearchSnapshot
    ) -> "EvolutionSearch":
        search = cls(context, (state.budget, 0))
        for name in state.__dataclass_fields__:
            if name not in ("rng_state", "seen", "volumes", "population"):
                setattr(search, name, getattr(state, name))
        search.seen = set(state.seen)
        search.known = {design_identity(row) for row in state.archive}
        search.volumes = list(state.volumes)
        search.rng.bit_generator.state = state.rng_state
        search._algorithm.pop = deepcopy(state.population)
        search.rng.bit_generator.state = state.rng_state
        return search
