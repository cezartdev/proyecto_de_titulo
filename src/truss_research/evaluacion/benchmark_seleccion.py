"""Repetición temporal de estrategias congeladas, separada del evaluador retrospectivo."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.precision_cpu import cpu_batches, probe_cpu_clock
from truss_research.evaluacion.recursos import ResourceMonitor, timed
from truss_research.evaluacion.seleccion import EpisodeReport, selection_metrics
from truss_research.optimizacion.catalogos import SelectionBank, episode_data, load_bank
from truss_research.optimizacion.comparacion import (
    PolicyReference,
    load_references,
    strategy_episode_data,
)
from truss_research.optimizacion.contratos import (
    EpisodeSpec,
    SelectionConfig,
    SelectionStrategy,
)
from truss_research.optimizacion.entorno import create_episode
from truss_research.optimizacion.estrategias import (
    GreedyStrategy,
    RandomStrategy,
    RankingStrategy,
    SafeGreedyStrategy,
)
from truss_research.optimizacion.estudio_estado import StudyConfig
from truss_research.optimizacion.evolucion import EvolutionContext, EvolutionStrategy
from truss_research.optimizacion.politica import PolicyStrategy
from truss_research.optimizacion.variantes_rl import VARIANT_REGISTRY, policy_variant
from truss_research.surrogates.experimentos import load_predictor

ASSISTED_STRATEGIES = frozenset(
    (
        "ranking",
        "greedy",
        "nsga2_surrogate",
        "nsga2_surrogate_random_init",
        "greedy_safe",
        *(part.name for part in VARIANT_REGISTRY.values() if part.use_predictions),
    )
)
PROGRESS_EPISODES = 10
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionCase:
    name: str
    variant: str
    seed: int | None = None
    policy: PolicyStrategy | None = None

    @property
    def assisted(self) -> bool:
        return self.variant in ASSISTED_STRATEGIES


def strategy_factory(
    case: SelectionCase,
    bank: SelectionBank,
    context: tuple[EpisodeSpec, SelectionConfig],
) -> Callable[[], SelectionStrategy]:
    spec, config = context
    if case.policy is not None:
        return lambda: PolicyStrategy(case.policy.components)
    if case.name == "random":
        return lambda: RandomStrategy(spec.seed)
    if case.name == "ranking":
        observation = create_episode(episode_data(bank, spec)).observe()
        return lambda: RankingStrategy(observation, bank.space)
    if case.name == "greedy":
        return lambda: GreedyStrategy(bank.space)
    if case.name == "greedy_safe":
        if bank.safe_bounds is None:
            raise ValueError("Faltan cotas calibradas para greedy_safe")
        return lambda: SafeGreedyStrategy(bank.space, bank.safe_bounds)
    return lambda: EvolutionStrategy(
        EvolutionContext(
            bank.space,
            bank.feature_ranges,
            spec.seed,
            replace(config.evolution, guided_initialization=False)
            if case.name.endswith("random_init")
            else config.evolution,
            case.assisted,
        )
    )


def replay_episode(
    data: tuple[SelectionBank, EpisodeSpec], factory: Callable[[], SelectionStrategy]
) -> tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]:
    bank, spec = data
    strategy, preparation = timed(factory)
    private = strategy_episode_data(episode_data(bank, spec), strategy)
    environment, initial = timed(lambda: create_episode(private))
    selected = [spec.initial_index]
    decision_wall, decision_cpu = 0.0, 0.0
    environment_wall, environment_cpu = initial["wall_seconds"], initial["cpu_seconds"]
    trace: list[dict[str, JsonValue]] = [
        {
            "query": 1,
            "unique_queries": 1,
            "repeat_attempts": 0,
            "iter_id": spec.candidate_ids[spec.initial_index],
            "decision_cumulative_seconds": preparation["wall_seconds"],
            "environment_cumulative_seconds": environment_wall,
        }
    ]
    while not environment.observe().terminal:
        action, choice = timed(lambda: strategy.choose(environment.observe()))
        _, feedback = timed(lambda: environment.step(action))
        selected.append(action)
        decision_wall += choice["wall_seconds"]
        decision_cpu += choice["cpu_seconds"]
        environment_wall += feedback["wall_seconds"]
        environment_cpu += feedback["cpu_seconds"]
        trace.append(
            {
                "query": len(selected),
                "unique_queries": environment.observe().queries,
                "repeat_attempts": environment.observe().repeat_attempts,
                "iter_id": spec.candidate_ids[action],
                "decision_cumulative_seconds": preparation["wall_seconds"]
                + decision_wall,
                "environment_cumulative_seconds": environment_wall,
            }
        )
    volumes = tuple(
        bank.space.hypervolume(private.responses[selected[:index]])
        for index in range(1, len(selected) + 1)
    )
    metrics = selection_metrics(
        EpisodeReport(private.responses, tuple(selected), volumes, bank.space)
    )
    reference_volume = float(metrics["reference_hypervolume"])
    for row, volume in zip(trace, volumes, strict=True):
        row["relative_hypervolume"] = (
            volume / reference_volume if reference_volume > 0 else None
        )
    return {
        **metrics,
        "queries": environment.observe().queries,
        "attempts": environment.observe().attempt_count,
        "unique_queries": environment.observe().queries,
        "repeat_attempts": environment.observe().repeat_attempts,
        "repeat_rate": environment.observe().repeat_attempts
        / environment.observe().attempt_count,
        "initialization_seconds": preparation["wall_seconds"],
        "initialization_cpu_seconds": preparation["cpu_seconds"],
        "decision_seconds": decision_wall,
        "decision_cpu_seconds": decision_cpu,
        "environment_seconds": environment_wall,
        "environment_cpu_seconds": environment_cpu,
        "selection_seconds": preparation["wall_seconds"] + decision_wall,
        "selection_cpu_seconds": preparation["cpu_seconds"] + decision_cpu,
    }, trace


def replay_simulation(
    data: tuple[SelectionBank, EpisodeSpec], factory: Callable[[], SelectionStrategy]
) -> list[dict[str, JsonValue]]:
    bank, spec = data
    strategy = factory()
    environment = create_episode(
        strategy_episode_data(episode_data(bank, spec), strategy)
    )
    trajectory: list[dict[str, JsonValue]] = [
        {"iter_id": spec.candidate_ids[spec.initial_index]}
    ]
    while not environment.observe().terminal:
        action = strategy.choose(environment.observe())
        environment.step(action)
        trajectory.append({"iter_id": spec.candidate_ids[action]})
    return trajectory


def assert_trajectory(
    trace: list[dict[str, JsonValue]], expected: tuple[str, ...]
) -> None:
    if tuple(str(row["iter_id"]) for row in trace) != expected:
        raise ValueError("El benchmark cambió la trayectoria de consultas")


def selection_benchmark(
    config: StudyConfig, context: tuple[Path, dict[str, Path]], destination: Path
) -> None:
    winner, tasks = context
    policies = tuple(
        PolicyReference(tasks[f"policy/{variant}/{seed}"], seed, variant)
        for variant in config.rl.variants
        for seed in config.seeds
    )
    loaded: dict[str, SelectionStrategy] = {}
    loading_rows: list[dict[str, JsonValue]] = []
    for reference in policies:
        part, loading = timed(lambda: load_references((reference,), winner))
        loaded.update(part)
        loading_rows.append(
            {
                "operation": "load_policy",
                "comparison_id": f"{reference.variant}_{reference.seed}",
                **loading,
            }
        )
    model, predictor_load = timed(lambda: load_predictor(winner))
    base_cases = [SelectionCase(name, name) for name in ("random", "ranking", "greedy")]
    if (winner / "calibration.json").is_file():
        base_cases.append(SelectionCase("greedy_safe", "greedy_safe"))
    if config.evolution.enabled:
        base_cases.extend(
            SelectionCase(name, name) for name in ("nsga2", "nsga2_surrogate")
        )
        if config.evolution.initialization_ablation:
            base_cases.extend(
                SelectionCase(name, name)
                for name in ("nsga2_random_init", "nsga2_surrogate_random_init")
            )
    cases = base_cases + [
        SelectionCase(
            f"{ref.variant}_{ref.seed}",
            ref.variant,
            ref.seed,
            loaded[f"{ref.variant}_{ref.seed}"],
        )
        for ref in policies
    ]
    rows: list[dict[str, JsonValue]] = []
    traces: list[dict[str, JsonValue]] = []
    memories: list[dict[str, JsonValue]] = []
    cpu_samples: list[dict[str, JsonValue]] = []
    clock = probe_cpu_clock(config.measurement.cpu_probe_seconds)
    rng = np.random.default_rng(config.measurement.seed)
    for split, key in (
        ("validation", "decision/validation"),
        ("test", "test/decision"),
    ):
        primary = pd.read_csv(tasks[key] / "traces.csv")
        for size in config.analysis.catalog_sizes:
            selection = SelectionConfig(
                config.data_dir,
                winner,
                destination,
                split=split,
                catalog_size=size,
                budgets=config.budgets,
                seeds=config.seeds,
                episodes_per_seed=config.episodes_per_seed,
                evolution=config.evolution,
                base_catalog_size=config.catalog_size,
            )
            bank = load_bank(selection)
            logger.info(
                "Selección: %s, %s candidatos, %s episodios",
                split,
                size,
                len(bank.episodes),
            )
            for episode_number, spec in enumerate(bank.episodes, start=1):
                if episode_number % PROGRESS_EPISODES == 0:
                    logger.info(
                        "Selección %s/%s: episodio %s/%s",
                        split,
                        size,
                        episode_number,
                        len(bank.episodes),
                    )
                features = bank.table.loc[list(spec.candidate_ids), list(INPUT_COLUMNS)]
                for case_index in rng.permutation(len(cases)):
                    case = cases[int(case_index)]
                    identity: dict[str, JsonValue] = {
                        "split": split,
                        "catalog_size": size,
                        "episode_id": spec.episode_id,
                        "catalog_seed": spec.seed,
                        "budget": spec.budget,
                        "strategy": case.variant,
                        "algorithm": policy_variant(case.variant).algorithm
                        if case.policy is not None
                        else None,
                        "comparison_id": case.name,
                        "policy_training_seed": case.seed,
                        "uses_predictions": case.assisted,
                    }
                    factory = strategy_factory(case, bank, (spec, selection))
                    expected: tuple[str, ...] | None = None
                    if size == config.catalog_size:
                        subset = primary.loc[
                            primary.episode_id.eq(spec.episode_id)
                            & primary.comparison_id.eq(case.name)
                        ].sort_values("query")
                        expected = tuple(subset.iter_id)
                    warmup = (
                        config.measurement.warmup if spec == bank.episodes[0] else 0
                    )
                    for _ in range(warmup):
                        if case.assisted:
                            model.predict(features)
                        replay_episode((bank, spec), factory)
                    for repetition in range(config.measurement.selection_repeats):
                        prediction = {"wall_seconds": 0.0, "cpu_seconds": 0.0}
                        if case.assisted:
                            predicted, prediction = timed(
                                lambda: model.predict(features)
                            )
                            np.testing.assert_allclose(
                                predicted,
                                bank.predictions.loc[
                                    list(spec.candidate_ids)
                                ].to_numpy(),
                                rtol=1e-5,
                                atol=1e-7,
                            )
                        metrics, trace = replay_episode((bank, spec), factory)
                        expected = (
                            tuple(str(row["iter_id"]) for row in trace)
                            if expected is None
                            else expected
                        )
                        assert_trajectory(trace, expected)
                        rows.append(
                            {
                                **identity,
                                "repetition": repetition,
                                **metrics,
                                "prediction_seconds": prediction["wall_seconds"],
                                "prediction_cpu_seconds": prediction["cpu_seconds"],
                                "independent_seconds": float(
                                    metrics["selection_seconds"]
                                )
                                + prediction["wall_seconds"],
                                "independent_cpu_seconds": float(
                                    metrics["selection_cpu_seconds"]
                                )
                                + prediction["cpu_seconds"],
                                "simulation_seconds": float(
                                    metrics["selection_seconds"]
                                )
                                + prediction["wall_seconds"]
                                + float(metrics["environment_seconds"]),
                                "cached_seconds": metrics["selection_seconds"],
                            }
                        )
                        traces.extend(
                            {
                                **identity,
                                "repetition": repetition,
                                "independent_cumulative_seconds": float(
                                    row["decision_cumulative_seconds"]
                                )
                                + prediction["wall_seconds"],
                                "simulation_cumulative_seconds": float(
                                    row["decision_cumulative_seconds"]
                                )
                                + prediction["wall_seconds"]
                                + float(row["environment_cumulative_seconds"]),
                                **row,
                            }
                            for row in trace
                        )
                    representative = next(
                        episode
                        for episode in bank.episodes
                        if episode.budget == spec.budget
                    )
                    if spec == representative:

                        def cpu_episode() -> None:
                            trajectory = replay_simulation((bank, spec), factory)
                            assert_trajectory(trajectory, expected)

                        cpu_samples.extend(
                            {
                                **identity,
                                "operation": "episode",
                                "scope": "full_simulated_episode_including_setup_without_retrospective_metrics",
                                **sample,
                            }
                            for sample in cpu_batches(
                                cpu_episode, config.measurement, clock
                            )
                        )
                    with ResourceMonitor(
                        config.measurement.memory_interval_ms
                    ) as monitor:
                        if case.assisted:
                            model.predict(features)
                        _, memory_trace = replay_episode((bank, spec), factory)
                    assert_trajectory(memory_trace, expected)
                    memories.append({**identity, **monitor.snapshot()})
    pd.DataFrame(rows).to_csv(destination / "selection_samples.csv", index=False)
    pd.DataFrame(traces).to_csv(destination / "selection_time_traces.csv", index=False)
    pd.DataFrame(memories).to_csv(destination / "selection_memory.csv", index=False)
    pd.DataFrame(cpu_samples).to_csv(
        destination / "selection_cpu_batches.csv", index=False
    )
    loading_rows.append(
        {
            "operation": "load_predictor",
            "comparison_id": "shared_predictor",
            **predictor_load,
        }
    )
    pd.DataFrame(loading_rows).to_csv(
        destination / "selection_loading.csv", index=False
    )
