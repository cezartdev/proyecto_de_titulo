"""Evaluación conjunta de políticas sin repetir las estrategias de referencia."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter, process_time

import pandas as pd

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.pareto import compliance_report
from truss_research.evaluacion.seleccion import (
    EpisodeReport,
    selection_metrics,
)
from truss_research.optimizacion.catalogos import (
    SelectionBank,
    bank_manifest,
    comparison_signature,
    episode_data,
    load_bank,
)
from truss_research.optimizacion.contratos import (
    EpisodeSpec,
    SelectionConfig,
    SelectionStrategy,
)
from truss_research.optimizacion.entorno import EpisodeData, area_reward, create_episode
from truss_research.optimizacion.estrategias import (
    GreedyStrategy,
    RandomStrategy,
    RankingStrategy,
    SafeGreedyStrategy,
)
from truss_research.optimizacion.evolucion import EvolutionContext, EvolutionStrategy
from truss_research.optimizacion.seleccion_incierta import EnsembleStrategy
from truss_research.optimizacion.variantes_rl import PolicyVariant, policy_variant
from truss_research.surrogates.experimentos import read_run, write_json


@dataclass(frozen=True)
class PolicyReference:
    directory: Path
    seed: int
    variant: PolicyVariant


@dataclass(frozen=True)
class LearnedPolicies:
    strategies: dict[str, SelectionStrategy]
    references: tuple[PolicyReference, ...]


def comparison_episode(
    bank: SelectionBank,
    context: tuple[EpisodeSpec, SelectionConfig],
    policies: LearnedPolicies,
) -> tuple[list[dict[str, JsonValue]], list[dict[str, JsonValue]]]:
    spec, config = context
    identities = {
        f"{reference.variant}_{reference.seed}": reference
        for reference in policies.references
    }
    strategies = {
        **strategies_for(bank, (spec, config)),
        **{name: (policy, 0.0) for name, policy in policies.strategies.items()},
    }
    summaries: list[dict[str, JsonValue]] = []
    traces: list[dict[str, JsonValue]] = []
    for key, (strategy, preparation_seconds) in strategies.items():
        summary, trace = evaluate_episode((bank, spec), strategy)
        summary["preparation_seconds"] = preparation_seconds
        summary["strategy_seconds"] = preparation_seconds + float(
            summary["decision_seconds"]
        )
        reference = identities.get(key)
        identity: dict[str, JsonValue] = {
            "episode_id": spec.episode_id,
            "catalog_seed": spec.seed,
            "budget": spec.budget,
            "strategy": reference.variant if reference else key,
            "policy_training_seed": reference.seed if reference else None,
            "comparison_id": key,
            "algorithm": policy_variant(reference.variant).algorithm
            if reference
            else None,
            "use_predictions": policy_variant(reference.variant).use_predictions
            if reference
            else key
            in (
                "ranking",
                "greedy",
                "greedy_safe",
                "nsga2_surrogate",
                "nsga2_surrogate_random_init",
            ),
        }
        summaries.append({**identity, **summary})
        traces.extend({**identity, **row} for row in trace)
    return summaries, traces


def load_references(
    policies: tuple[PolicyReference, ...], predictor: Path
) -> dict[str, SelectionStrategy]:
    from truss_research.optimizacion.politica import load_policy

    strategies: dict[str, SelectionStrategy] = {}
    for reference in policies:
        metadata = read_run(reference.directory)
        settings = metadata.get("config")
        if not isinstance(settings, dict) or metadata.get("phase") != "complete":
            raise ValueError("La política debe estar completa y contener configuración")
        variant = policy_variant(reference.variant)
        penalty = settings.get("repeat_penalty")
        if (
            settings.get("seed") != reference.seed
            or settings.get("use_predictions") != variant.use_predictions
            or settings.get("algorithm") != variant.algorithm
            or settings.get("reward_mode") != variant.reward_mode
            or settings.get("enhanced_state") != variant.enhanced_state
            or settings.get("allow_repeats") != variant.allow_repeats
            or (not variant.repeat_penalty and settings.get("repeat_penalty") != 0.0)
            or (
                variant.repeat_penalty
                and (not isinstance(penalty, (int, float)) or penalty <= 0)
            )
        ):
            raise ValueError("La identidad o variante de la política no coincide")
        key = f"{reference.variant}_{reference.seed}"
        if key in strategies:
            raise ValueError("Hay políticas repetidas en la comparación")
        strategies[key] = load_policy(reference.directory, predictor)
    return strategies


def evaluate_selection_suite(
    config: SelectionConfig, policies: tuple[PolicyReference, ...]
) -> pd.DataFrame:
    if config.output_dir.exists():
        raise ValueError("El destino de comparación ya existe")
    start = perf_counter()
    bank = load_bank(config)
    bank_seconds = perf_counter() - start
    start = perf_counter()
    learned = load_references(policies, config.predictor_dir) if policies else {}
    policy_loading_seconds = perf_counter() - start
    summaries: list[dict[str, JsonValue]] = []
    traces: list[dict[str, JsonValue]] = []
    for spec in bank.episodes:
        rows, trajectory = comparison_episode(
            bank, (spec, config), LearnedPolicies(learned, policies)
        )
        summaries.extend(rows)
        traces.extend(trajectory)
    result = pd.DataFrame(summaries)
    timing: dict[str, JsonValue] = {
        "bank_preparation_seconds": bank_seconds - bank.prediction_seconds,
        "prediction_seconds": bank.prediction_seconds,
        "predicted_designs": len(bank.table),
        "policy_loading_seconds": policy_loading_seconds,
        "scope": "one_cached_batch_per_split; strategy_seconds includes initialization and choices per episode",
    }
    return write_comparison(
        bank, (config, policies), (result, pd.DataFrame(traces), timing)
    )


def write_comparison(
    bank: SelectionBank,
    context: tuple[SelectionConfig, tuple[PolicyReference, ...]],
    results: tuple[pd.DataFrame, pd.DataFrame, dict[str, JsonValue]],
) -> pd.DataFrame:
    config, policies = context
    result, traces, timing = results
    config.output_dir.mkdir(parents=True, exist_ok=False)
    result.to_csv(config.output_dir / "summary.csv", index=False)
    traces.to_csv(config.output_dir / "traces.csv", index=False)
    baseline = result[result["strategy"] == "random"].set_index("episode_id")
    paired = result[result["strategy"] != "random"].copy()
    paired["auc_difference_vs_random"] = paired["normalized_auc"] - paired[
        "episode_id"
    ].map(baseline["normalized_auc"])
    paired.to_csv(config.output_dir / "paired_differences.csv", index=False)
    manifest = bank_manifest(bank, config)
    manifest["comparison_signature"] = comparison_signature(bank, config)
    manifest["policies"] = [
        {
            **asdict(reference),
            "policy_signature": source_digest(reference.directory / "policy.pt"),
            "encoder_signature": source_digest(reference.directory / "encoder.joblib"),
        }
        for reference in policies
    ]
    manifest["timing"] = timing
    write_json(config.output_dir / "run.json", manifest)
    write_json(
        config.output_dir / "compliance.json",
        compliance_report(
            bank.table.loc[:, list(OUTPUT_COLUMNS)].to_numpy(),
            bank.predictions.to_numpy(),
            bank.space.limits,
        ),
    )
    return result


def evaluate_ranking(config: SelectionConfig) -> None:
    """Mantiene la estrategia fija para comparar la utilidad de predictores."""
    from truss_research.optimizacion.catalogos import episode_data
    from truss_research.optimizacion.estrategias import RankingStrategy

    if config.output_dir.exists():
        raise ValueError("El destino de ranking ya existe")
    bank = load_bank(config)
    rows: list[dict[str, JsonValue]] = []
    traces: list[dict[str, JsonValue]] = []
    for spec in bank.episodes:
        observation = create_episode(episode_data(bank, spec)).observe()
        strategy, preparation_seconds = timed_strategy(
            lambda: RankingStrategy(observation, bank.space)
        )
        summary, trace = evaluate_episode((bank, spec), strategy)
        summary["preparation_seconds"] = preparation_seconds
        summary["strategy_seconds"] = preparation_seconds + float(
            summary["decision_seconds"]
        )
        identity = {
            "episode_id": spec.episode_id,
            "catalog_seed": spec.seed,
            "budget": spec.budget,
            "strategy": "ranking",
        }
        rows.append({**identity, **summary})
        traces.extend({**identity, **row} for row in trace)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(config.output_dir / "summary.csv", index=False)
    pd.DataFrame(traces).to_csv(config.output_dir / "traces.csv", index=False)
    manifest = bank_manifest(bank, config)
    manifest["comparison_signature"] = comparison_signature(bank, config)
    manifest["prediction_seconds"] = bank.prediction_seconds
    manifest["predicted_designs"] = len(bank.table)
    write_json(config.output_dir / "run.json", manifest)


def strategy_episode_data(
    data: EpisodeData, strategy: SelectionStrategy
) -> EpisodeData:
    """Comparación y benchmark comparten el tratamiento de intentos repetidos."""
    from truss_research.optimizacion.politica import PolicyStrategy

    encoder = (
        strategy.components.encoder if isinstance(strategy, PolicyStrategy) else None
    )
    return replace(
        data,
        reward_mode="weighted_marginal",
        allow_repeats=encoder.allow_repeats if encoder else False,
        repeat_penalty=encoder.repeat_penalty if encoder else 0.0,
    )


def evaluate_episode(
    data: tuple[SelectionBank, EpisodeSpec], strategy: SelectionStrategy
) -> tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]:
    bank, spec = data
    private = strategy_episode_data(episode_data(bank, spec), strategy)
    scoring = bank.evaluation_space or bank.space
    reward_scale = scoring.hypervolume(private.responses)
    environment = create_episode(
        replace(
            private,
            space=scoring,
            reward_scale=reward_scale,
        )
    )
    selected = [spec.initial_index]
    volume = scoring.hypervolume(private.responses[selected])
    volumes = [volume]
    trace = [
        {
            "query": 1,
            "iter_id": spec.candidate_ids[spec.initial_index],
            "hypervolume": volume,
            "reward": environment.initial_reward,
            "hypervolume_increment": volume,
            "auc_contribution": area_reward(0.0, volume, (spec.budget, reward_scale)),
            "is_initial": True,
            "attempts": 1,
            "unique_queries": 1,
            "repeat_attempts": 0,
            "is_repeat": False,
            **dict(
                zip(OUTPUT_COLUMNS, private.responses[spec.initial_index], strict=True)
            ),
        }
    ]
    decision_seconds = 0.0
    decision_cpu_seconds = 0.0
    while not environment.observe().terminal:
        start = perf_counter()
        cpu_start = process_time()
        action = strategy.choose(environment.observe())
        decision_seconds += perf_counter() - start
        decision_cpu_seconds += process_time() - cpu_start
        repeated = not environment.observe().available[action]
        _, reward, _, _, _ = environment.step(action)
        response = environment.observe().observed[action]
        selected.append(action)
        volumes.append(scoring.hypervolume(private.responses[selected]))
        trace.append(
            {
                "query": len(selected),
                "iter_id": spec.candidate_ids[action],
                "hypervolume": volumes[-1],
                "reward": reward,
                "hypervolume_increment": volumes[-1] - volumes[-2],
                "auc_contribution": area_reward(
                    volumes[-2], volumes[-1], (spec.budget, reward_scale)
                ),
                "is_initial": False,
                "attempts": environment.observe().attempt_count,
                "unique_queries": environment.observe().queries,
                "repeat_attempts": environment.observe().repeat_attempts,
                "is_repeat": repeated,
                **dict(zip(OUTPUT_COLUMNS, response, strict=True)),
                **(
                    strategy.diagnostics
                    if isinstance(
                        strategy,
                        (EvolutionStrategy, EnsembleStrategy, SafeGreedyStrategy),
                    )
                    else {}
                ),
            }
        )
    report = EpisodeReport(private.responses, tuple(selected), tuple(volumes), scoring)
    summary = {
        **selection_metrics(report),
        "queries": environment.observe().queries,
        "attempts": environment.observe().attempt_count,
        "unique_queries": environment.observe().queries,
        "repeat_attempts": environment.observe().repeat_attempts,
        "repeat_rate": environment.observe().repeat_attempts
        / environment.observe().attempt_count,
        "penalized_return": sum(float(row["reward"]) for row in trace),
        "repetition_penalty": private.repeat_penalty,
        "decision_seconds": decision_seconds,
        "decision_cpu_seconds": decision_cpu_seconds,
    }
    return summary, trace


def timed_strategy(
    factory: Callable[[], SelectionStrategy],
) -> tuple[SelectionStrategy, float]:
    start = perf_counter()
    strategy = factory()
    return strategy, perf_counter() - start


def strategies_for(
    bank: SelectionBank, context: tuple[EpisodeSpec, SelectionConfig]
) -> dict[str, tuple[SelectionStrategy, float]]:
    spec, config = context
    observation = create_episode(episode_data(bank, spec)).observe()
    strategies: dict[str, tuple[SelectionStrategy, float]] = {
        "random": timed_strategy(lambda: RandomStrategy(spec.seed)),
        "ranking": timed_strategy(lambda: RankingStrategy(observation, bank.space)),
        "greedy": timed_strategy(lambda: GreedyStrategy(bank.space)),
    }
    if config.evolution.enabled:
        strategies.update(
            {
                name: timed_strategy(
                    lambda assisted=assisted: EvolutionStrategy(
                        EvolutionContext(
                            bank.space,
                            bank.feature_ranges,
                            spec.seed,
                            config.evolution,
                            assisted,
                        )
                    )
                )
                for name, assisted in (("nsga2", False), ("nsga2_surrogate", True))
            }
        )
        if config.evolution.initialization_ablation:
            strategies.update(
                {
                    name: timed_strategy(
                        lambda assisted=assisted: EvolutionStrategy(
                            EvolutionContext(
                                bank.space,
                                bank.feature_ranges,
                                spec.seed,
                                replace(config.evolution, guided_initialization=False),
                                assisted,
                            )
                        )
                    )
                    for name, assisted in (
                        ("nsga2_random_init", False),
                        ("nsga2_surrogate_random_init", True),
                    )
                }
            )
    if bank.safe_bounds is not None:
        strategies["greedy_safe"] = timed_strategy(
            lambda: SafeGreedyStrategy(bank.space, bank.safe_bounds)
        )
    return strategies


def load_selection_comparison(directories: tuple[Path, ...]) -> pd.DataFrame:
    from truss_research.surrogates.experimentos import read_run

    signatures = [
        read_run(directory).get("comparison_signature") for directory in directories
    ]
    if not signatures or None in signatures or len(set(signatures)) != 1:
        raise ValueError(
            "Las comparaciones no comparten dataset, episodios y referencia"
        )
    return pd.concat(
        [
            pd.read_csv(directory / "summary.csv").assign(run=str(directory))
            for directory in directories
        ],
        ignore_index=True,
    )
