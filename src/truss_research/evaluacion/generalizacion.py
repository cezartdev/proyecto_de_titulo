"""Transferencia cronológica, control de tamaño y selección retenida."""

from dataclasses import dataclass, replace
from pathlib import Path
from shutil import copytree

import pandas as pd

from truss_research.cerchas.schemas import OUTPUT_COLUMNS, WEB_CONFIG_VALUES
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.particion_equilibrada import (
    chronological_generation_split,
    matched_design_control,
    verify_group_exclusion,
)
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import load_processed_table
from truss_research.datos.schemas import (
    DESIGN_MEMBERSHIP_FILENAME,
    FULL_FILENAME,
    TableSplit,
)
from truss_research.evaluacion.diagnosticos_incertidumbre import summarize_coverage
from truss_research.evaluacion.incertidumbre_seleccion import selection_intervals
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.evaluacion.prediccion_retenida import assess_retained, fit_retained
from truss_research.evaluacion.recursos import measure_phase
from truss_research.evaluacion.seleccion_retenida import (
    RetainedSelection,
    evaluate_retained_selection,
)
from truss_research.evaluacion.trabajos_analisis import execute_analysis_jobs
from truss_research.optimizacion.estudio_estado import StudyConfig
from truss_research.optimizacion.trabajos_predictivos import (
    JobOutcome,
    JobResources,
    run_measured_job,
)
from truss_research.surrogates.calibracion import SplitConformal
from truss_research.surrogates.configuracion import PredictorConfig, response_scale
from truss_research.surrogates.experimentos import data_signature, write_json
from truss_research.surrogates.persistencia import load_model

GENERALIZATION_TABLES = (
    "metrics",
    "intervals",
    "selection",
    "category_coverage",
    "purged_designs",
    "partition_candidates",
    "feasibility",
    "coverage",
    "selection_episodes",
    "selection_intervals",
    "generation_overlap",
)
EXPERIMENTS = ("chronological", "design_overlap_control")


@dataclass(frozen=True)
class GeneralizationJob:
    config: StudyConfig
    destination: Path
    seed: int

    @property
    def serial(self) -> bool:
        return self.config.tabpfn_enabled


def evaluate_group_models(
    job: GeneralizationJob,
    context: tuple[str, TableSplit],
    references: tuple[tuple[float, ...], ObjectiveSpace],
) -> None:
    experiment, split = context
    scales, evaluation_space = references
    config, destination = job.config, job.destination / experiment
    destination.mkdir(parents=True, exist_ok=True)
    models = (
        config.models
        if experiment == "chronological"
        else tuple(
            name
            for name in config.analysis.generalization.control_models
            if name in config.models
        )
    )
    rows, intervals, feasibility, scores, episodes = [], [], [], [], []
    for name in models:
        output = destination / name
        requested = PredictorConfig(
            config.data_dir,
            output,
            model=name,
            seed=config.representative_seed,
            folds=config.inner_folds or config.final_tuning_folds,
            trials=0 if name == "tabpfn" else config.trials,
            tabpfn=config.tabpfn,
            pruning=config.pruning,
            device="cuda" if name in config.execution.gpu_models else "cpu",
            parameters=config.parameters,
            cv_seed=job.seed,
            group_membership=(config.data_dir / DESIGN_MEMBERSHIP_FILENAME)
            if experiment == "chronological"
            else None,
            cv_protocol="chronological" if experiment == "chronological" else "design",
            study_path=destination / "studies" / f"{name}.sqlite",
            candidate_models=tuple(m for m in config.models if m != "targetwise"),
        )
        with measure_phase(f"{experiment}_{name}_fit_validation"):
            model, calibrated = fit_retained(requested, split, name != "tabpfn")
            measured, bounds, failures, score = assess_retained(
                (split.training, model, calibrated, config.limits),
                split.validation,
                scales,
            )
        identity = {
            "model": name,
            "split": "validation",
            "experiment": experiment,
            "evaluation_seed": job.seed,
        }
        rows.append(measured.assign(**identity))
        intervals.append(bounds.assign(**identity))
        feasibility.append(failures.assign(**identity))
        scores.append({**identity, "normalized_mae": score})
    winner = min(scores, key=lambda row: row["normalized_mae"])["model"]
    write_json(
        destination / "selection_freeze.json",
        {
            "winner": winner,
            "criterion": "validation_normalized_mae",
            "validation_scores": scores,
            "test_used_for_selection": False,
        },
    )
    for name in models:
        output = destination / name
        model = load_model(output)
        payload = read_artifact(output / "calibration.json")
        calibrated = SplitConformal(payload["residuals"])
        with measure_phase(f"{experiment}_{name}_test"):
            measured, bounds, failures, score = assess_retained(
                (split.training, model, calibrated, config.limits),
                split.testing,
                scales,
            )
        identity = {
            "model": name,
            "split": "test",
            "experiment": experiment,
            "evaluation_seed": job.seed,
        }
        rows.append(measured.assign(**identity))
        intervals.append(bounds.assign(**identity))
        feasibility.append(failures.assign(**identity))
        scores.append({**identity, "normalized_mae": score})
        if name == winner:
            request = RetainedSelection(
                config,
                split.training,
                destination / "selection",
                source_digest(output / "model.joblib"),
                data_signature(config.data_dir),
                calibrated,
                evaluation_space,
            )
            with measure_phase(f"{experiment}_selection"):
                for label, table in (
                    ("validation", split.validation),
                    ("test", split.testing),
                ):
                    episodes.append(
                        evaluate_retained_selection(
                            request, model, (label, table)
                        ).assign(
                            experiment=experiment, evaluation_seed=job.seed, model=name
                        )
                    )
    interval_table = pd.concat(intervals, ignore_index=True)
    score_table = pd.DataFrame(scores)
    score_table["selected_on_validation"] = score_table.model.eq(winner)
    tables = {
        "metrics": pd.concat(rows, ignore_index=True),
        "intervals": interval_table,
        "feasibility": pd.concat(feasibility, ignore_index=True),
        "selection": score_table,
        "coverage": summarize_coverage(
            interval_table, pd.read_csv(config.data_dir / DESIGN_MEMBERSHIP_FILENAME)
        ),
        "selection_episodes": pd.concat(episodes, ignore_index=True),
    }
    tables["selection_intervals"] = selection_intervals(
        tables["selection_episodes"], config.analysis
    ).assign(experiment=experiment, evaluation_seed=job.seed)
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False)


def execute_generalization(job: GeneralizationJob) -> None:
    config, destination = job.config, job.destination
    membership = pd.read_csv(
        config.data_dir / DESIGN_MEMBERSHIP_FILENAME, float_precision="round_trip"
    )
    full = load_processed_table(config.data_dir / FULL_FILENAME)
    settings = replace(
        config.analysis.generalization,
        min_evaluation=max(
            config.catalog_size, config.analysis.generalization.min_evaluation
        ),
    )
    partition = chronological_generation_split(full, membership, (job.seed, settings))
    verify_group_exclusion(partition.tables, membership)
    split = TableSplit(
        partition.tables["train"],
        partition.tables["validation"],
        partition.tables["test"],
        partition.tables["calibration"],
    )
    destination.mkdir(parents=True, exist_ok=False)
    failed = sorted(
        destination.parent.glob(destination.name + "_failed_*"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if failed:
        previous = failed[-1]
        for studies in previous.glob("*/studies"):
            copytree(studies, destination / studies.relative_to(previous))
    write_json(
        destination / "protocol.json",
        {
            "scope": "single_optimization_trajectory",
            "partition": "chronological_contiguous_purged",
            "fractions": list(settings.fractions),
            "conformal_coverage": "empirical_under_temporal_shift",
            "evaluation_seed": job.seed,
        },
    )
    partition.assignments.to_csv(
        destination / "generation_assignments.csv", index=False
    )
    partition.excluded.to_csv(destination / "purged_designs.csv", index=False)
    partition.candidates.to_csv(destination / "partition_candidates.csv", index=False)
    coverage, overlap = [], []
    control = matched_design_control(full, split, job.seed)
    scales = response_scale(split.training.loc[:, list(OUTPUT_COLUMNS)])
    evaluation_space = ObjectiveSpace.fit(
        split.training.loc[:, list(OUTPUT_COLUMNS)].to_numpy(),
        config.limits,
    )
    for experiment, current in zip(EXPERIMENTS, (split, control), strict=True):
        groups = {}
        for label, table in current.as_mapping().items():
            table[["iter_id", "generation", "web_config"]].to_csv(
                destination / f"{experiment}_{label}_ids.csv", index=False
            )
            groups[label] = set(
                membership.loc[
                    membership.representative_iter_id.isin(table.iter_id), "generation"
                ]
            )
            coverage.extend(
                {
                    "experiment": experiment,
                    "split": label,
                    "web_config": value,
                    "designs": int(table.web_config.eq(value).sum()),
                    "absent": not table.web_config.eq(value).any(),
                }
                for value in WEB_CONFIG_VALUES
            )
        overlap.extend(
            {
                "experiment": experiment,
                "split": name,
                "shared_training_generations": len(groups["train"] & groups[name]),
                "generations": len(groups[name]),
            }
            for name in ("calibration", "validation", "test")
        )
        evaluate_group_models(
            job,
            (experiment, current),
            (scales, evaluation_space),
        )
    pd.DataFrame(coverage).to_csv(destination / "category_coverage.csv", index=False)
    pd.DataFrame(overlap).to_csv(destination / "generation_overlap.csv", index=False)
    for name in (
        "metrics",
        "intervals",
        "selection",
        "coverage",
        "feasibility",
        "selection_episodes",
        "selection_intervals",
    ):
        pd.concat(
            [
                pd.read_csv(destination / experiment / f"{name}.csv", low_memory=False)
                for experiment in EXPERIMENTS
            ],
            ignore_index=True,
        ).to_csv(destination / f"{name}.csv", index=False)


def run_generalization_job(job: GeneralizationJob) -> JobOutcome:
    return run_measured_job(
        JobResources(
            f"generalization/{job.seed}",
            job.destination,
            job.config.data_dir,
            job.config.measurement,
        ),
        lambda: execute_generalization(job),
    )


def run_generalization(config: StudyConfig, destination: Path) -> None:
    jobs = [
        GeneralizationJob(config, destination / f"seed_{seed}", seed)
        for seed in config.analysis.generalization_seeds
    ]
    if not jobs:
        return
    execute_analysis_jobs(jobs, run_generalization_job, destination)
    for name in GENERALIZATION_TABLES:
        pd.concat(
            [
                pd.read_csv(job.destination / f"{name}.csv", low_memory=False).assign(
                    evaluation_seed=job.seed
                )
                for job in jobs
            ],
            ignore_index=True,
        ).to_csv(destination / f"{name}.csv", index=False)
