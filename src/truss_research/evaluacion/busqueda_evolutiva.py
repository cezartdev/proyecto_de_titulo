"""Campaña temporal de búsqueda: desarrollo, congelación y cierre A/B separados."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    WEB_CONFIG_VALUES,
)
from truss_research.datos.artefactos import read_artifact, replace_artifact
from truss_research.datos.distancias import design_distances
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import (
    JsonValue,
    load_design_split,
    manifest_protocol,
    read_manifest,
)
from truss_research.evaluacion.benchmark_multiobjetivo import (
    benchmark_protocol,
    evaluate_predictor_utility,
)
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.optimizacion.busqueda import EvolutionSearch, SearchContext
from truss_research.optimizacion.entrenamiento_propuestas import (
    ProposalEncoder,
    ProposalTrainingConfig,
    load_proposal_policy,
    predicted_auc,
)
from truss_research.optimizacion.estrategias_busqueda import STRATEGY_DEFINITIONS
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    StudyPhase,
    TaskRecord,
    artifact_signatures,
    environment_versions,
    read_study,
    source_signature,
    study_lock,
    verify_task,
)
from truss_research.optimizacion.evaluadores import (
    HistoricalEvaluator,
    SearchEvaluator,
    SurrogateEvaluator,
)
from truss_research.optimizacion.politica import CandidateQNetwork, DQNParameters
from truss_research.optimizacion.propuestas import TrainingSupport, design_identity
from truss_research.optimizacion.seleccion_propuestas import proposal_selector
from truss_research.surrogates.experimentos import (
    data_signature,
    load_predictor,
    read_run,
    write_json,
)
from truss_research.surrogates.persistencia import load_model

SEARCH_VERSION = 4
SEARCH_DIRECTORY = "evolution"
SCENARIO_VALIDATION = 1_000_000
SCENARIO_TEST = 2_000_000


def search_plan(config: StudyConfig, phase: StudyPhase) -> dict[str, JsonValue]:
    config.execution.validate_campaign()
    gpu = config.tabpfn_enabled and config.tabpfn.device == "cuda"
    return {
        "mode": "plan_sin_ejecucion",
        "experiment": "evolution",
        "phase": phase,
        "data_protocol": config.split_protocol,
        "objective_mode": config.objective_mode,
        "data_dir": str(config.data_dir),
        "models": list(config.models),
        "seeds": list(config.seeds),
        "budgets": list(config.budgets),
        "initial_population": config.evolution.population_size,
        "pool_size": config.proposals.pool_size,
        "max_generated": config.proposals.max_generated,
        "policy_transitions": 0
        if phase in ("final", "predictores")
        else len(config.search.policies) * len(config.seeds) * config.rl_steps,
        "strategies": list(config.search.strategies),
        "strategy_definitions": [asdict(row) for row in config.search.definitions],
        "search_parameters": asdict(config.search),
        "evidence": ["historical_observed", "surrogate_estimated"],
        "pruning": asdict(config.pruning),
        "parameters": asdict(config.parameters),
        "execution": {
            **asdict(config.execution),
            "device": "cpu+cuda" if gpu else "cpu",
            "gpu_models": ["tabpfn"] if gpu else [],
            "gpu_workers": int(gpu),
            "workers": config.execution.effective_workers,
            "numeric_threads_per_worker": 1,
        },
        "manual_execution": True,
        "primary_component": "predictor_comparison",
        "predictor_multiobjective_benchmark": benchmark_protocol(config),
        "secondary_component": "exploratory_model_guided_search",
        "predictive_statistics": {
            "bootstrap_repeats": config.analysis.bootstrap_repeats,
            "block_generations": config.analysis.predictive_block_generations,
            "primary_controls": ["dummy", "ridge"],
            "scope": "pointwise_exploratory_intervals_no_significance_or_equivalence_claim",
        },
        "test_labels_used": False,
    }


@dataclass(frozen=True)
class PredictorReference:
    model: str
    seed: int
    record: TaskRecord


def predictor_records(study: Path) -> tuple[PredictorReference, ...]:
    rows: list[PredictorReference] = []
    for raw in read_study(study)["tasks"]:
        parts = str(raw["key"]).split("/")
        if len(parts) == 3 and parts[0] == "predictor":
            rows.append(PredictorReference(parts[1], int(parts[2]), TaskRecord(**raw)))
    return tuple(rows)


def predictor_directories(config: StudyConfig) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    references = predictor_records(config.output_dir)
    if {(row.model, row.seed) for row in references} != {
        (name, seed) for name in config.models for seed in config.seeds
    }:
        raise ValueError("Completar todas las familias y semillas predictivas")
    for reference in references:
        directory = verify_task(reference.record)
        if reference.seed == config.representative_seed:
            paths[reference.model] = directory
    if set(paths) != set(config.models):
        raise ValueError("Completar todos los predictores antes de entrenar búsquedas")
    return paths


def choose_predictors(
    config: StudyConfig, paths: dict[str, Path]
) -> tuple[str, str, str]:
    scores = {name: [] for name in config.models}
    for reference in predictor_records(config.output_dir):
        scores[reference.model].append(
            float(read_run(Path(reference.record.directory))["normalized_mae"])
        )
    utility = pd.read_csv(
        config.output_dir / "evolution/predictor_utility/validation/summary.csv"
    )
    utility = (
        utility.loc[
            utility.role.eq("regressor") & utility.budget.eq(min(config.budgets))
        ]
        .groupby("model")
        .auc.mean()
    )
    if set(utility.index) != set(paths) or not np.isfinite(utility.to_numpy()).all():
        raise ValueError(
            "Completar el benchmark multiobjetivo de todas las familias antes de elegir"
        )
    ordered = sorted(
        paths, key=lambda name: (-float(utility[name]), np.mean(scores[name]), name)
    )
    if len(ordered) < 2:
        raise ValueError("Se requieren dos familias para evaluar y auditar la búsqueda")
    winner, auditor = ordered[:2]
    cheap = "ridge" if "ridge" in paths and winner != "ridge" else auditor
    return winner, auditor, cheap


def build_context(config: StudyConfig, paths: tuple[Path, Path]) -> SearchContext:
    if manifest_protocol(read_manifest(config.data_dir)) != "first_appearance":
        raise ValueError("Los mundos evolutivos requieren primera aparición temporal")
    winner, cheap = paths
    split = load_design_split(config.data_dir)
    training = split.training
    features = training[list(INPUT_COLUMNS)].to_numpy(dtype=float)
    responses = training[list(OUTPUT_COLUMNS)].to_numpy(dtype=float)
    space = ObjectiveSpace.fit(responses, config.limits)
    metadata = read_run(winner)
    members = tuple(winner / name for name in sorted(metadata["fold_model_signatures"]))
    if len(members) < 2:
        raise ValueError("Faltan modelos externos para separar mundos y checkpoints")
    for member in members:
        from truss_research.surrogates.persistencia import model_signatures

        if model_signatures(member) != metadata["fold_model_signatures"][member.name]:
            raise ValueError("Cambió un mundo predictivo externo")
    models = tuple(load_model(member) for member in members)
    training_models = models[:-1]
    worlds = tuple(
        SurrogateEvaluator(
            training_models[:index] + training_models[index + 1 :] or (model,),
            config.proposals.uncertainty_multiplier,
        )
        for index, model in enumerate(training_models)
    )
    return SearchContext(
        features,
        responses,
        worlds,
        TrainingSupport.fit(features, config.proposals.support_quantile),
        space,
        space.hypervolume(responses),
        config.proposals,
        config.evolution,
        max(config.budgets),
        cheap=SurrogateEvaluator((load_predictor(cheap),), 0.0),
        validation_world=SurrogateEvaluator(
            (models[-1],), config.proposals.uncertainty_multiplier
        ),
        search_parameters=config.search,
    )


def campaign_identity(config: StudyConfig, paths: dict[str, Path]) -> str:
    from truss_research.optimizacion.estudio_estado import study_source_signature

    payload = {
        "version": SEARCH_VERSION,
        "config": asdict(config),
        "source": study_source_signature(config.output_dir),
        "dataset": data_signature(config.data_dir),
        "versions": environment_versions(),
        "models": {name: artifact_signatures(path) for name, path in paths.items()},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def policy_configuration(
    config: StudyConfig, identity: tuple[str, int, str]
) -> ProposalTrainingConfig:
    signature, seed, name = identity
    strategy = STRATEGY_DEFINITIONS[name]
    parameters = DQNParameters(
        **asdict(config.rl.dqn),
        transitions=config.rl_steps,
        hidden_width=config.rl_hidden,
        batch_size=config.rl_batch,
        warmup=config.rl_warmup,
        validation_interval=config.validation_interval,
    )
    policy_signature = hashlib.sha256(f"{signature}/{seed}/{name}".encode()).hexdigest()
    return ProposalTrainingConfig(
        seed,
        config.budgets,
        parameters,
        policy_signature,
        config.proposals.evaluation_episodes,
        objective="immediate" if strategy.selector == "contextual_bandit" else "return",
    )


def evaluate_episode(
    context: SearchContext,
    identity: tuple[str, int, int],
    policy: tuple[CandidateQNetwork, ProposalEncoder] | None,
) -> tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]:
    name, budget, seed = identity
    strategy = STRATEGY_DEFINITIONS[name]
    selector = proposal_selector(strategy, policy)
    parameters = (
        replace(context.parameters, pool_size=1)
        if strategy.single_proposal
        else context.parameters
    )
    context = replace(
        context,
        parameters=parameters,
        assisted=strategy.assisted,
        generator=strategy.generator,
    )
    started = perf_counter()
    search = EvolutionSearch(context, (budget, seed))
    rows: list[dict[str, JsonValue]] = []
    rng = np.random.default_rng(seed)
    selection_seconds = 0.0
    while search.prepare():
        batch = search.batch
        assert batch is not None
        selection_started = perf_counter()
        action = selector.select(search, rng)
        selection_seconds += perf_counter() - selection_started
        proposal = batch.features[action].copy()
        lineage = batch.lineage[action] if batch.lineage else ()
        parent_population = search.population_features.tolist()
        reward = search.choose(action)
        revealed = search.last_evaluation
        rows.append(
            {
                "evaluation": search.evaluations,
                "hypervolume": search.volumes[-1],
                "reward": reward,
                "projection_distance": float(revealed.distances[0])
                if revealed is not None
                else None,
                "evidence": revealed.evidence
                if revealed is not None
                else "unsupported_historical_projection",
                "proposed": proposal.tolist(),
                "origin": batch.origin,
                "generator": strategy.generator,
                "selector": strategy.selector,
                "target_and_donor_indices": list(lineage),
                "parent_population": parent_population,
                "lineage_scope": "target_then_three_distinct_donors"
                if lineage
                else "evaluated_population_exact_parent_pairs_not_recorded",
                "evaluated": revealed.features[0].tolist()
                if revealed is not None
                else None,
                "responses": revealed.responses[0].tolist()
                if revealed is not None
                else None,
            }
        )
    quality = search.volumes[-1]
    distances = design_distances(search.archive, search.archive, context.support.scales)
    pairs = distances[np.triu_indices(len(distances), k=1)]
    return {
        "auc": predicted_auc(search),
        "unsupported_queries": sum(
            row["evidence"] == "unsupported_historical_projection" for row in rows
        ),
        "uncertainty_scope": context.worlds[0].uncertainty_scope
        if isinstance(context.worlds[0], SurrogateEvaluator)
        else "observed_historical_responses",
        "projection_distance_mean": float(
            np.mean(
                [
                    row["projection_distance"]
                    for row in rows
                    if row["projection_distance"] is not None
                ]
            )
        )
        if any(row["projection_distance"] is not None for row in rows)
        else None,
        "hypervolume": quality,
        "initial_hypervolume": search.volumes[context.evolution.population_size],
        "evaluations": search.evaluations,
        "unique_evaluations": search.unique_evaluations,
        "generated": search.generated,
        "rejected": search.rejected,
        "duplicates": search.duplicates,
        "cheap_predictions": search.cheap_evaluations,
        "evaluator_member_predictions": search.member_predictions,
        "cheap_member_predictions": search.cheap_member_predictions,
        "generation_seconds": search.generation_seconds,
        "evaluation_seconds": search.evaluation_seconds,
        "cheap_seconds": search.cheap_seconds,
        "selection_seconds": selection_seconds,
        "mean_design_distance": float(pairs.mean()) if len(pairs) else 0.0,
        "archive_topology_count": len(np.unique(search.archive[:, 0])),
        "proposal_topology_counts": {
            str(category): sum(int(row["proposed"][0]) == category for row in rows)
            for category in WEB_CONFIG_VALUES
        },
        "generation_counter_scope": "returned_offspring_before_support_with_bounded_mating_attempts",
        "termination": search.reason,
        "search_seconds": perf_counter() - started,
        "curve": list(search.volumes),
        "archive_features": search.archive.tolist(),
        "archive_responses": search.responses.tolist(),
    }, rows


@dataclass(frozen=True)
class EpisodeJob:
    context: SearchContext
    mode: str
    strategy: str
    budget: int
    scenario: int
    policy_seed: int
    random_seed: int
    policy: tuple[CandidateQNetwork, ProposalEncoder] | None


def cached_episode(
    directory: Path, job: EpisodeJob, receipt: dict[str, str]
) -> tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]:
    name = f"episode_{job.mode}_{job.strategy}_{job.budget}_{job.scenario}_{job.policy_seed}.json"
    path = directory / name
    if name in receipt:
        if not path.is_file() or source_digest(path) != receipt[name]:
            raise ValueError("Cambió un episodio terminado")
        payload = read_artifact(path)
        return payload["outcome"], payload["decisions"]
    outcome, decisions = evaluate_episode(
        job.context, (job.strategy, job.budget, job.random_seed), job.policy
    )
    write_json(path, {"outcome": outcome, "decisions": decisions})
    receipt[name] = source_digest(path)
    write_json(directory / "episodes.json", {"files": receipt})
    return outcome, decisions


def evaluation_context(
    config: StudyConfig, base: SearchContext, identity: tuple[str, str, Path]
) -> SearchContext:
    mode, split_name, winner = identity
    if mode == "B":
        world: SearchEvaluator = SurrogateEvaluator(
            family_models(config, str(read_run(winner)["config"]["model"])),
            config.proposals.uncertainty_multiplier,
        )
    else:
        split = load_design_split(config.data_dir)
        table = split.validation if split_name == "validation" else split.testing
        world = HistoricalEvaluator(
            table[list(INPUT_COLUMNS)].to_numpy(dtype=float),
            table[list(OUTPUT_COLUMNS)].to_numpy(dtype=float),
            base.support.scales,
        )
    return replace(base, worlds=(world,), validation_world=None)


def family_models(config: StudyConfig, family: str) -> tuple[Pipeline, ...]:
    """Reutiliza todas las semillas de la familia congelada, sin nuevos ajustes."""
    return tuple(
        load_predictor(verify_task(reference.record))
        for reference in sorted(
            predictor_records(config.output_dir), key=lambda row: row.seed
        )
        if reference.model == family
    )


def evaluate_campaign(
    config: StudyConfig,
    base: SearchContext,
    request: tuple[str, dict[str, Path], tuple[str, str, str]],
) -> None:
    split_name, paths, selected = request
    root = config.output_dir / SEARCH_DIRECTORY
    destination = root / split_name
    if (destination / "complete.json").exists():
        verify_evaluation(destination)
        return
    destination.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, str] = (
        read_artifact(destination / "episodes.json")["files"]
        if (destination / "episodes.json").exists()
        else {}
    )
    winner, auditor, _ = selected
    rows: list[dict[str, JsonValue]] = []
    traces: list[dict[str, JsonValue]] = []
    auditor_model = SurrogateEvaluator(family_models(config, auditor), 0.0)
    policies = {
        (name, seed): load_proposal_policy(root / "policies" / name / str(seed))
        for name in config.search.policies
        for seed in config.seeds
    }
    offset = SCENARIO_VALIDATION if split_name == "validation" else SCENARIO_TEST
    for mode in ("A", "B"):
        context = evaluation_context(config, base, (mode, split_name, paths[winner]))
        reference = base.reward_scale
        for budget in config.budgets:
            for scenario in range(config.proposals.evaluation_episodes):
                scenario_seed = offset + scenario
                for name in config.search.strategies:
                    seeds = (
                        config.seeds
                        if name in config.search.policies
                        else (config.representative_seed,)
                    )
                    for seed in seeds:
                        policy = policies.get((name, seed))
                        outcome, decisions = cached_episode(
                            destination,
                            EpisodeJob(
                                context,
                                mode,
                                name,
                                budget,
                                scenario,
                                seed,
                                scenario_seed,
                                policy,
                            ),
                            receipt,
                        )
                        curve = np.array(outcome.pop("curve"))
                        archive = np.array(outcome.pop("archive_features"))
                        values = np.array(outcome.pop("archive_responses"))
                        reached = (
                            np.flatnonzero(
                                curve >= config.search.target_fraction * reference
                            )
                            if reference > 0
                            else np.array([])
                        )
                        audit = auditor_model.evaluate(archive)
                        novel = sum(
                            design_identity(row) not in base.training_identities
                            for row in archive[config.evolution.population_size :]
                        )
                        key = {
                            "experiment": mode,
                            "split": split_name,
                            "strategy": name,
                            "budget": budget,
                            "scenario": scenario,
                            "policy_seed": seed,
                        }
                        rows.append(
                            {
                                **key,
                                **outcome,
                                "reference_hypervolume": reference,
                                "target_fraction": config.search.target_fraction,
                                "target_reference_scope": "training_only_frozen_before_test",
                                "attained": bool(len(reached)),
                                "queries_to_target": int(reached[0])
                                if len(reached)
                                else None,
                                "auditor_hypervolume": base.space.hypervolume(
                                    audit.responses
                                ),
                                "novel_configurations": novel,
                                "evidence": "historical_observed"
                                if mode == "A"
                                else "surrogate_estimated",
                            }
                        )
                        traces.extend({**key, **decision} for decision in decisions)
                        archive_path = (
                            destination
                            / f"archive_{mode}_{name}_{budget}_{scenario}_{seed}.csv"
                        )
                        pd.DataFrame(
                            np.column_stack((archive, values)),
                            columns=[*INPUT_COLUMNS, *OUTPUT_COLUMNS],
                        ).to_csv(archive_path, index=False)
    pd.DataFrame(rows).to_csv(destination / "summary.csv", index=False)
    pd.DataFrame(traces).to_json(
        destination / "decisions.jsonl", orient="records", lines=True
    )
    paired_intervals(pd.DataFrame(rows), config).to_csv(
        destination / "paired_intervals.csv", index=False
    )
    write_json(
        destination / "complete.json",
        {
            "split": split_name,
            "source_signature": source_signature(),
            "files": artifact_signatures(destination),
            "complete": True,
        },
    )


def paired_intervals(table: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    from truss_research.evaluacion.contrastes_busqueda import paired_search_intervals

    return paired_search_intervals(table, config)


def verify_evaluation(directory: Path) -> None:
    metadata = read_artifact(directory / "complete.json")
    for name, digest in metadata["files"].items():
        if (
            not (directory / name).is_file()
            or source_digest(directory / name) != digest
        ):
            raise ValueError("Cambió una evaluación evolutiva completa")


def close_predictors(config: StudyConfig) -> None:
    """Evalúa predictores congelados sobre prueba sin volver a ajustarlos."""
    from truss_research.optimizacion.estudio_cierre import evaluate_test_predictor

    for reference in predictor_records(config.output_dir):
        source = verify_task(reference.record)
        destination = (
            config.output_dir
            / SEARCH_DIRECTORY
            / "test_predictors"
            / f"{reference.model}_{reference.seed}"
        )
        if (destination / "complete.json").exists():
            verify_evaluation(destination)
            continue
        if destination.exists():
            raise ValueError(
                "Evaluación predictiva incompleta; conservarla y elegir un destino nuevo para ese cierre"
            )
        attempt = (
            config.output_dir
            / SEARCH_DIRECTORY
            / "test_predictor_attempts"
            / uuid4().hex
        )
        attempt.parent.mkdir(parents=True, exist_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        workspace = config.output_dir.resolve()
        if not attempt.resolve().is_relative_to(
            workspace
        ) or not destination.resolve().is_relative_to(workspace):
            raise ValueError("La evaluación predictiva salió del estudio")
        evaluate_test_predictor(source, attempt)
        write_json(
            attempt / "complete.json",
            {"files": artifact_signatures(attempt), "complete": True},
        )
        replace_artifact(attempt, destination)


def run_evolution_study(config: StudyConfig, phase: StudyPhase, resume: bool) -> Path:
    from truss_research.optimizacion.estudio import run_study

    config = replace(
        config,
        output_dir=config.output_dir.resolve(),
        data_dir=config.data_dir.resolve(),
    )
    if manifest_protocol(read_manifest(config.data_dir)) != "first_appearance":
        raise ValueError("Preparar un dataset nuevo con primera aparición temporal")
    if phase != "final":
        run_study(config, "predictores", resume)
    with study_lock(config.output_dir):
        paths = predictor_directories(config)
        evaluate_predictor_utility(config, "validation")
        root = config.output_dir / SEARCH_DIRECTORY
        signature = campaign_identity(config, paths)
        manifest_path = root / "search.json"
        if (
            manifest_path.exists()
            and read_artifact(manifest_path)["signature"] != signature
        ):
            raise ValueError("Cambió el protocolo de búsqueda; elegir un destino nuevo")
        selected = choose_predictors(config, paths)
        if (
            manifest_path.exists()
            and tuple(read_artifact(manifest_path)["selected"]) != selected
        ):
            raise ValueError("Cambió la selección predictiva")
        root.mkdir(exist_ok=True)
        write_json(
            manifest_path,
            {
                "signature": signature,
                "selected": list(selected),
                "version": SEARCH_VERSION,
                "phase": "development",
            },
        )
        base = build_context(config, (paths[selected[0]], paths[selected[2]]))
        write_json(
            root / "coverage.json",
            {
                "partition": "first_appearance",
                "diagnostics": read_manifest(config.data_dir).get("diagnostics", {}),
                "historical_projection": "same_topology_nearest_design",
                "unsupported_topology": "consume_query_no_improvement_continue",
                "target_reference_scope": "training_only_frozen_before_test",
                "target_hypervolume": config.search.target_fraction * base.reward_scale,
            },
        )
        if phase != "final":
            from truss_research.evaluacion.reutilizacion_predictores import (
                verify_reused_validation,
            )
            from truss_research.optimizacion.paralelismo_politicas import (
                run_policy_campaign,
            )

            verify_reused_validation(config, selected)
            run_policy_campaign(
                config,
                base,
                (
                    signature,
                    "cuda" in str(read_run(paths[selected[0]]).get("device", "cpu")),
                ),
            )
            evaluate_campaign(config, base, ("validation", paths, selected))
        if campaign_identity(config, paths) != signature:
            raise ValueError(
                "Los datos, modelos o fuentes cambiaron durante la búsqueda"
            )
        if phase in ("final", "todo"):
            verify_evaluation(root / "validation")
            policy_files = artifact_signatures(root / "policies")
            frozen_models = {
                f"{name}/{config.representative_seed}": {
                    "directory": str(path),
                    "files": artifact_signatures(path),
                }
                for name, path in paths.items()
            }
            for reference in predictor_records(config.output_dir):
                directory = verify_task(reference.record)
                frozen_models[f"{reference.model}/{reference.seed}"] = {
                    "directory": str(directory),
                    "files": reference.record.signatures,
                }
            freeze = {
                "signature": signature,
                "selected": list(selected),
                "policies": policy_files,
                "validation": artifact_signatures(root / "validation"),
                "predictor_utility_validation": artifact_signatures(
                    root / "predictor_utility/validation"
                ),
                "test_labels_used_for_training": False,
                "models": frozen_models,
                "policy_execution": artifact_signatures(root / "policy_execution"),
                "predictor_import": source_digest(
                    config.output_dir / "predictor_import.json"
                )
                if (config.output_dir / "predictor_import.json").exists()
                else None,
            }
            freeze_path = config.output_dir / "freeze.json"
            if freeze_path.exists() and read_artifact(freeze_path) != freeze:
                raise ValueError("Cambió el cierre congelado")
            write_json(freeze_path, freeze)
            close_predictors(config)
            evaluate_predictor_utility(config, "test")
            evaluate_campaign(config, base, ("test", paths, selected))
            if any(
                artifact_signatures(Path(model["directory"])) != model["files"]
                for model in frozen_models.values()
            ):
                raise ValueError("Cambió un predictor después de congelar el cierre")
            if campaign_identity(config, paths) != signature:
                raise ValueError(
                    "Los datos, modelos o fuentes cambiaron durante el cierre"
                )
            write_json(
                root / "closed.json",
                {
                    "signature": signature,
                    "selected": list(selected),
                    "version": SEARCH_VERSION,
                    "phase": "complete",
                    "test_predictors": artifact_signatures(root / "test_predictors"),
                    "predictor_utility_test": artifact_signatures(
                        root / "predictor_utility/test"
                    ),
                },
            )
    return config.output_dir
