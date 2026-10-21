"""Eficiencia de datos con parámetros fijos y referencias de evaluación comunes."""

from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copyfile

import pandas as pd
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import load_design_split
from truss_research.datos.schemas import DESIGN_MEMBERSHIP_FILENAME, TableSplit
from truss_research.datos.subconjuntos import nested_subsets
from truss_research.evaluacion.diagnosticos_incertidumbre import summarize_coverage
from truss_research.evaluacion.incertidumbre_seleccion import selection_intervals
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.evaluacion.prediccion_retenida import assess_retained, fit_retained
from truss_research.evaluacion.recursos import measure_phase
from truss_research.evaluacion.seleccion_retenida import (
    CURVE_STRATEGIES,
    RetainedSelection,
    evaluate_retained_selection,
)
from truss_research.evaluacion.sensibilidad import (
    combine_sensitivity_tables,
    sensitivity_inputs,
)
from truss_research.evaluacion.trabajos_analisis import (
    analysis_session,
    execute_analysis_jobs,
)
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    artifact_signatures,
    load_study_config,
    source_signature,
    study_lock,
)
from truss_research.optimizacion.trabajos_predictivos import (
    JobOutcome,
    JobResources,
    run_measured_job,
)
from truss_research.surrogates.calibracion import SplitConformal
from truss_research.surrogates.configuracion import (
    ModelName,
    ModelParameters,
    PredictorConfig,
    response_scale,
)
from truss_research.surrogates.experimentos import data_signature, write_json
from truss_research.surrogates.persistencia import load_model

CURVE_SCHEMA_VERSION = 3
CURVE_TABLES = (
    "metrics",
    "scores",
    "intervals",
    "feasibility",
    "selection_episodes",
    "selection_intervals",
)


@dataclass(frozen=True)
class LearningCurveJob:
    config: StudyConfig
    destination: Path
    model: ModelName
    train_size: int
    subset_seed: int | None
    serial: bool = False


def tuned_parameters(config: StudyConfig, model: ModelName) -> ModelParameters:
    if not (config.output_dir / "study.json").is_file():
        return config.parameters
    from truss_research.evaluacion.informe_estudio import signed_records

    key = f"predictor/{model}/{config.representative_seed}"
    for record in signed_records(config.output_dir):
        if record["key"] == key:
            saved = read_artifact(Path(str(record["directory"])) / "run.json")
            raw = saved.get("config", {}).get("parameters")
            if isinstance(raw, dict):
                return ModelParameters(**raw)
    return config.parameters


def curve_jobs(
    config: StudyConfig, destination: Path, total: int
) -> list[LearningCurveJob]:
    settings = config.analysis.learning_curves
    if max(settings.train_sizes) > total:
        raise ValueError("Una curva solicita más diseños que el ajuste disponible")
    jobs = []
    models = settings.models
    for model in models:
        for size in sorted(set((*settings.train_sizes, total))):
            seeds = (None,) if size == total else settings.subset_seeds
            for seed in seeds:
                suffix = "full" if seed is None else f"seed_{seed}"
                jobs.append(
                    LearningCurveJob(
                        config,
                        destination / f"{model}_n{size}_{suffix}",
                        model,
                        size,
                        seed,
                        config.tabpfn_enabled and model in ("tabpfn", "targetwise"),
                    )
                )
    return jobs


def execute_curve(job: LearningCurveJob) -> None:
    config = job.config
    split = load_design_split(config.data_dir)
    training = (
        split.training
        if job.subset_seed is None
        else nested_subsets(split.training, (job.train_size,), job.subset_seed)[
            job.train_size
        ]
    )
    current = TableSplit(training, split.validation, split.testing, split.calibration)
    job.destination.mkdir(parents=True, exist_ok=False)
    requested = PredictorConfig(
        config.data_dir,
        job.destination / "predictor",
        model=job.model,
        seed=config.representative_seed,
        parameters=tuned_parameters(config, job.model),
        folds=config.final_tuning_folds,
        trials=0,
        tabpfn=config.tabpfn,
        pruning=config.pruning,
        device="cuda" if job.model in config.execution.gpu_models else "cpu",
    )
    reused = job.model == "tabpfn" and job.subset_seed is None
    with measure_phase("predictor_reuse" if reused else "fixed_predictor_fit"):
        model, calibrated = (
            reuse_full_tabpfn(job)
            if reused
            else fit_retained(requested, current, False)
        )
    identity = {
        "model": job.model,
        "train_size": len(training),
        "subset_seed": job.subset_seed,
        "model_seed": requested.seed,
        "tuning": False,
    }
    scales = response_scale(split.training.loc[:, list(OUTPUT_COLUMNS)])
    evaluation_space = ObjectiveSpace.fit(
        split.training.loc[:, list(OUTPUT_COLUMNS)].to_numpy(),
        config.limits,
    )
    request = RetainedSelection(
        config,
        training,
        job.destination / "selection",
        source_digest(requested.output_dir / "model.joblib"),
        data_signature(config.data_dir),
        calibrated,
        evaluation_space,
        CURVE_STRATEGIES,
    )
    rows, scores, intervals, failures, episodes, comparisons = [], [], [], [], [], []
    for label, table in (("validation", split.validation), ("test", split.testing)):
        with measure_phase(f"prediction_{label}"):
            metrics, bounds, feasibility, score = assess_retained(
                (training, model, calibrated, config.limits), table, scales
            )
        rows.append(metrics.assign(split=label, **identity))
        scores.append({"split": label, **identity, "normalized_mae": score})
        intervals.append(bounds.assign(split=label, **identity))
        failures.append(feasibility.assign(split=label, **identity))
        with measure_phase(f"selection_{label}"):
            selected = evaluate_retained_selection(request, model, (label, table))
        episodes.append(selected.assign(**identity))
        comparisons.append(
            selection_intervals(selected, config.analysis).assign(**identity)
        )
    tables = {
        "metrics": pd.concat(rows, ignore_index=True),
        "scores": pd.DataFrame(scores),
        "intervals": pd.concat(intervals, ignore_index=True),
        "feasibility": pd.concat(failures, ignore_index=True),
        "selection_episodes": combine_sensitivity_tables(episodes),
        "selection_intervals": pd.concat(comparisons, ignore_index=True),
    }
    for name, table in tables.items():
        table.to_csv(job.destination / f"{name}.csv", index=False)
    training[["iter_id", "web_config"]].to_csv(
        job.destination / "subset_ids.csv", index=False
    )
    write_json(
        job.destination / "run.json",
        {
            **identity,
            "fit_ids": training.iter_id.tolist(),
            "evaluation_space": asdict(evaluation_space),
            "normalization_scale": list(scales),
            "reference_scope": "full_training_evaluation_only",
            "full_fit_reused_across_subset_seeds": job.subset_seed is None,
            "main_fit_reused": reused,
        },
    )


def reuse_full_tabpfn(job: LearningCurveJob) -> tuple[Pipeline, SplitConformal]:
    from truss_research.evaluacion.informe_estudio import signed_records

    root = next(
        Path(record["directory"])
        for record in signed_records(job.config.output_dir)
        if record["key"] == f"predictor/tabpfn/{job.config.representative_seed}"
    )
    destination = job.destination / "predictor"
    destination.mkdir()
    for path in root.iterdir():
        if (
            path.name in ("model.joblib", "calibration.json", "run.json")
            or path.suffix == ".tabpfn_fit"
        ):
            copyfile(path, destination / path.name)
    metadata = read_artifact(destination / "run.json")
    expected = load_design_split(job.config.data_dir).training.iter_id.tolist()
    if metadata["fit_ids"] != expected or metadata["config"]["tabpfn"] != asdict(
        job.config.tabpfn
    ):
        raise ValueError("El ajuste completo TabPFN no corresponde a la curva")
    metadata.update({"tuning": False, "main_fit_reused": True})
    write_json(destination / "run.json", metadata)
    return load_model(destination), SplitConformal(
        read_artifact(destination / "calibration.json")["residuals"]
    )


def run_curve_job(job: LearningCurveJob) -> JobOutcome:
    return run_measured_job(
        JobResources(
            job.destination.name,
            job.destination,
            job.config.data_dir,
            job.config.measurement,
        ),
        lambda: execute_curve(job),
    )


def curve_figures(
    scores: pd.DataFrame, selection: pd.DataFrame, destination: Path
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for model, group in scores.loc[scores.split.eq("test")].groupby("model"):
        values = group.groupby("train_size").normalized_mae.mean()
        axes[0].plot(values.index, values, marker="o", label=model)
    for model, group in selection.loc[
        selection.split.eq("test")
        & selection.strategy.eq("greedy")
        & selection.budget.eq(selection.budget.min())
    ].groupby("model"):
        values = group.groupby("train_size").normalized_auc.mean()
        axes[1].plot(values.index, values, marker="o", label=model)
    for axis, ylabel in zip(axes, ("MAE normalizado", "AUC de greedy"), strict=True):
        axis.set(xlabel="Diseños de ajuste", ylabel=ylabel)
        axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "learning_curves.png", dpi=150)


def run_learning_curves(study: Path, destination: Path) -> Path:
    if not (study / "study.json").is_file():
        raise ValueError("Falta completar make study y make final")
    if destination.resolve().is_relative_to(study.resolve()):
        raise ValueError("Guardar curvas fuera del estudio firmado, en reports")
    with study_lock(study):
        state, _ = sensitivity_inputs(study)
        config = load_study_config(study)
        if not config.analysis.learning_curves.enabled:
            raise ValueError(
                "Las curvas no están habilitadas en el protocolo congelado"
            )
        if (destination / "provenance.json").exists():
            raise ValueError("Las curvas ya están completas; elegir otra exportación")
        signatures = {
            name: source_digest(study / name) for name in ("study.json", "freeze.json")
        }
        jobs = curve_jobs(
            config, destination, len(load_design_split(config.data_dir).training)
        )
        analysis_session(config, destination)
        execute_analysis_jobs(jobs, run_curve_job, destination)
        collected = {}
        for name in CURVE_TABLES:
            table = pd.concat(
                [
                    pd.read_csv(job.destination / f"{name}.csv", low_memory=False)
                    for job in jobs
                ],
                ignore_index=True,
            )
            table.to_csv(destination / f"{name}.csv", index=False)
            collected[name] = table
        summarize_coverage(
            collected["intervals"],
            pd.read_csv(config.data_dir / DESIGN_MEMBERSHIP_FILENAME),
        ).to_csv(destination / "coverage.csv", index=False)
        collected["selection_episodes"].groupby(
            ["model", "train_size", "split", "strategy", "budget"], as_index=False
        )[["normalized_auc", "relative_hypervolume"]].mean().to_csv(
            destination / "selection_summary.csv", index=False
        )
        curve_figures(collected["scores"], collected["selection_episodes"], destination)
        if (
            any(
                source_digest(study / name) != digest
                for name, digest in signatures.items()
            )
            or data_signature(config.data_dir) != state["dataset_signature"]
        ):
            raise ValueError("La campaña o los datos cambiaron durante las curvas")
        analysis_session(config, destination)
        write_json(
            destination / "provenance.json",
            {
                "schema_version": CURVE_SCHEMA_VERSION,
                "study": str(study.resolve()),
                "state_signature": signatures["study.json"],
                "freeze_signature": signatures["freeze.json"],
                "dataset_signature": state["dataset_signature"],
                "analysis_source_signature": source_signature(),
                "parameters": asdict(config.analysis.learning_curves),
                "distinct_fits": sum(
                    not (job.model == "tabpfn" and job.subset_seed is None)
                    for job in jobs
                ),
                "reused_main_fits": int(config.tabpfn_enabled),
                "report_signatures": artifact_signatures(destination),
            },
        )
    return destination


def latest_learning_curves(study: Path) -> Path | None:
    from truss_research.evaluacion.registro import results_root

    for candidate in sorted(results_root(study).glob("curvas_*"), reverse=True):
        path = candidate / "provenance.json"
        if not path.is_file():
            continue
        saved = read_artifact(path)
        if (
            saved.get("schema_version") != CURVE_SCHEMA_VERSION
            or Path(str(saved.get("study"))).resolve() != study.resolve()
        ):
            continue
        if saved.get("state_signature") != source_digest(
            study / "study.json"
        ) or saved.get("freeze_signature") != source_digest(study / "freeze.json"):
            continue
        measured = {
            name: value
            for name, value in artifact_signatures(candidate).items()
            if name != "provenance.json"
        }
        if not measured or saved.get("report_signatures") != measured:
            raise ValueError(
                "Cambió una tabla, traza o figura de curvas de aprendizaje"
            )
        return candidate
    return None
