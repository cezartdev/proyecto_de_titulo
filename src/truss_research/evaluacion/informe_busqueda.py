"""Informes de búsqueda sobre artefactos cerrados, sin volver a entrenar ni evaluar."""

import html
import json
from math import ceil
from pathlib import Path

import pandas as pd

from truss_research.datos.artefactos import read_artifact
from truss_research.evaluacion.benchmark_multiobjetivo import (
    export_benchmark_reports,
    verify_benchmark,
)
from truss_research.evaluacion.busqueda_evolutiva import (
    SEARCH_DIRECTORY,
    verify_evaluation,
)
from truss_research.evaluacion.comparacion_predictiva import (
    export_predictive_comparison,
)
from truss_research.optimizacion.estrategias_busqueda import STRATEGY_DEFINITIONS
from truss_research.optimizacion.estudio_estado import (
    artifact_signatures,
    load_study_config,
)
from truss_research.surrogates.experimentos import write_json

NOTEBOOK_TOPICS = (
    "datos_temporales",
    "predictores",
    "busqueda_historica",
    "busqueda_generativa",
)


def verify_closed_search(study: Path) -> None:
    root = study / SEARCH_DIRECTORY
    freeze = read_artifact(study / "freeze.json")
    manifest = read_artifact(root / "closed.json")
    if freeze["signature"] != manifest["signature"] or manifest["phase"] != "complete":
        raise ValueError("Completar make final antes de exportar el cierre")
    if freeze["policies"] != artifact_signatures(root / "policies"):
        raise ValueError("Las políticas difieren del cierre congelado")
    if freeze["validation"] != artifact_signatures(root / "validation"):
        raise ValueError("Cambió la validación congelada")
    for model in freeze["models"].values():
        if artifact_signatures(Path(model["directory"])) != model["files"]:
            raise ValueError("Cambió un predictor congelado")
    verify_evaluation(root / "validation")
    verify_evaluation(root / "test")
    if manifest["test_predictors"] != artifact_signatures(root / "test_predictors"):
        raise ValueError("Cambió la evaluación predictiva final")
    for name, signatures in (
        ("validation", freeze["predictor_utility_validation"]),
        ("test", manifest["predictor_utility_test"]),
    ):
        directory = root / "predictor_utility" / name
        verify_benchmark(directory)
        if signatures != artifact_signatures(directory):
            raise ValueError("Cambió la comparación multiobjetivo de predictores")


def export_search(study: Path, destination: Path, kind: str) -> Path:
    verify_closed_search(study)
    if destination.exists():
        raise ValueError("La exportación requiere un destino nuevo")
    destination.mkdir(parents=True)
    config = load_study_config(study)
    root = study / SEARCH_DIRECTORY
    summaries = pd.concat(
        [pd.read_csv(root / name / "summary.csv") for name in ("validation", "test")]
    )
    summary = summaries.groupby(
        ["split", "experiment", "strategy", "budget"], as_index=False
    ).agg(
        mean_auc=("auc", "mean"),
        mean_hypervolume=("hypervolume", "mean"),
        attainment_rate=("attained", "mean"),
        mean_seconds=("search_seconds", "mean"),
        mean_evaluations=("evaluations", "mean"),
        mean_generated=("generated", "mean"),
        mean_rejected=("rejected", "mean"),
        mean_duplicates=("duplicates", "mean"),
        mean_unsupported_queries=("unsupported_queries", "mean"),
        mean_projection_distance=("projection_distance_mean", "mean"),
        mean_cheap_predictions=("cheap_predictions", "mean"),
        mean_generation_seconds=("generation_seconds", "mean"),
        mean_selection_seconds=("selection_seconds", "mean"),
        mean_evaluation_seconds=("evaluation_seconds", "mean"),
        mean_cheap_seconds=("cheap_seconds", "mean"),
        mean_design_distance=("mean_design_distance", "mean"),
        mean_topology_count=("archive_topology_count", "mean"),
    )
    summary.to_csv(destination / "comparison.csv", index=False)
    export_predictive_comparison(study, destination)
    export_benchmark_reports(root / "predictor_utility", destination)
    predictive_metrics = [
        pd.read_csv(path) for path in (root / "test_predictors").glob("*/metrics.csv")
    ]
    if predictive_metrics:
        pd.concat(predictive_metrics, ignore_index=True).to_csv(
            destination / "predictive_test_metrics.csv", index=False
        )
    policy_costs = [
        (path, read_artifact(path)) for path in (root / "policies").glob("*/*/run.json")
    ]
    costs = pd.DataFrame(
        [
            {
                "policy": path.parent.parent.name,
                "seed": int(path.parent.name),
                "learning_objective": row["config"]["objective"],
                **{
                    f"training_{key}": value
                    for key, value in row["training_costs"].items()
                },
                **{
                    f"validation_{key}": value
                    for key, value in row["validation_costs"].items()
                },
                **{
                    key: row.get(key)
                    for key in (
                        "wall_seconds",
                        "cpu_seconds",
                        "training_surrogate_evaluations",
                        "optimizer_updates",
                        "validation_surrogate_evaluations",
                    )
                },
            }
            for path, row in policy_costs
        ]
    )
    costs.to_csv(destination / "policy_costs.csv", index=False)
    amortization: list[dict[str, str | int | float | None]] = []
    final = summary.loc[summary.split.eq("test")]
    for (experiment, budget), group in final.groupby(["experiment", "budget"]):
        for policy in config.search.policies:
            baseline = "greedy" if STRATEGY_DEFINITIONS[policy].assisted else "random"
            control = group.loc[group.strategy.eq(baseline), "mean_seconds"]
            if control.empty:
                continue
            baseline_seconds = float(control.iloc[0])
            seconds = float(
                group.loc[group.strategy.eq(policy), "mean_seconds"].iloc[0]
            )
            training = float(costs.loc[costs.policy.eq(policy), "wall_seconds"].mean())
            saving = baseline_seconds - seconds
            amortization.append(
                {
                    "experiment": experiment,
                    "budget": int(budget),
                    "policy": policy,
                    "baseline": baseline,
                    "extra_policy_training_seconds": training,
                    "search_saving_seconds": saving,
                    "searches_to_amortize": ceil(training / saving)
                    if saving > 0
                    else None,
                    "scope": "exploratory_policy_cost_only_shared_predictors_excluded_no_fem",
                }
            )
    pd.DataFrame(amortization).to_csv(destination / "amortization.csv", index=False)
    summaries.groupby(
        ["split", "experiment", "strategy", "budget", "termination"], as_index=False
    ).size().to_csv(destination / "termination_counts.csv", index=False)
    summaries[
        [
            "split",
            "experiment",
            "strategy",
            "budget",
            "scenario",
            "policy_seed",
            "attained",
            "queries_to_target",
            "target_fraction",
            "reference_hypervolume",
            "target_reference_scope",
        ]
    ].to_csv(destination / "target_attainment.csv", index=False)
    state = read_artifact(study / "study.json")
    predictors = [
        read_artifact(Path(task["directory"]) / "run.json")
        for task in state["tasks"]
        if str(task["key"]).startswith("predictor/")
    ]
    pd.DataFrame(
        [
            {
                "model": row["config"]["model"],
                "seed": row["config"]["seed"],
                **{
                    key: row.get(key)
                    for key in (
                        "normalized_mae",
                        "cv_normalized_mae",
                        "tuning_seconds",
                        "crossfit_seconds",
                        "fit_seconds",
                        "requested_trials",
                        "trial_count",
                        "nested_trial_count",
                        "targetwise_family_trials",
                        "search_reused",
                    )
                },
            }
            for row in predictors
        ],
        columns=[
            "model",
            "seed",
            "normalized_mae",
            "cv_normalized_mae",
            "tuning_seconds",
            "crossfit_seconds",
            "fit_seconds",
            "requested_trials",
            "trial_count",
            "nested_trial_count",
            "targetwise_family_trials",
            "search_reused",
        ],
    ).to_csv(destination / "predictor_costs.csv", index=False)
    report = (
        "# Comparación de predictores y propuesta de búsqueda guiada\n\n"
        "El resultado principal es la utilidad multiobjetivo de cada regresor: predictor_utility_summary.csv "
        "y predictor_utility_paired_intervals.csv. La calidad HV/AUC usa respuestas históricas observadas "
        "tras seleccionar; todos comparten catálogo, inicialización, ranking y presupuesto. La familia "
        "se elige por AUC de validación en el menor presupuesto configurado, con MAE como desempate. "
        "El oráculo aplica el mismo ranking con etiquetas: no es un óptimo global ni una cota de AUC. "
        "Los presupuestos efectivos se reducen si el catálogo se agota, sin consultas ficticias. "
        "La precisión y costo complementan esta utilidad: predictive_model_summary.csv, predictive_metric_summary.csv, "
        "predictive_seed_metrics.csv y predictive_paired_errors.csv. Diferencias negativas "
        "favorecen al modelo frente a la referencia. Los contrastes principales usan Dummy y Ridge; "
        "los restantes son secundarios. Los intervalos del 95 % son puntuales, exploratorios "
        "y condicionados a los ajustes; no son pruebas de significación ni de equivalencia. "
        "generation_block remuestrea bloques completos; design_iid_sensitivity es un control. "
        "Pocos bloques limitan la estabilidad. Las semillas no multiplican el tamaño muestral.\n\n"
        "category_coverage.csv identifica categorías sin evaluación y physical_provenance.csv "
        "documenta condiciones conocidas y pendientes. El MAE normalizado usa desviaciones "
        "de entrenamiento y pesos iguales; no es un porcentaje de error. Los errores físicos "
        "y costos fundamentan la interpretación; no hay tolerancias de equivalencia inventadas.\n\n"
        "La búsqueda es una aplicación secundaria exploratoria. "
        "A usa respuestas históricas mediante proyección; B usa estimaciones de un sustituto. "
        "No se acredita factibilidad física de configuraciones nuevas ni ahorro FEM.\n\n"
        "Las consultas históricas sin topología compatible consumen presupuesto y no terminan "
        "el episodio. Su frecuencia se informa junto con distancias de proyección. B usa las "
        "semillas de la familia elegida; el desacuerdo puede ser cero en modelos deterministas "
        "y no constituye incertidumbre física calibrada.\n\n"
        "predictive_inference_costs.csv separa latencia individual y por lote. "
        "predictor_costs.csv registra intentos y ajustes anidados; treinta intentos no "
        "garantizan igual tiempo de cómputo entre familias. TabPFN incorpora preentrenamiento "
        "externo cuyo costo no se midió; Targetwise incluye selección adicional por respuesta.\n\n"
        "Las diferencias e intervalos pareados están en evolution/test/paired_intervals.csv. "
        "Las consultas no alcanzadas siguen censuradas en summary.csv.\n\n"
        "El umbral de alcance usa exclusivamente la referencia de entrenamiento. "
        "Los intervalos son exploratorios, sin corrección por comparaciones múltiples.\n\n"
        "El tiempo de entrenamiento de políticas se registra aparte del tiempo por búsqueda. "
        "No sumar fases anidadas del entrenamiento predictivo.\n"
    )
    (destination / "RESULTS.md").write_text(report, encoding="utf-8")
    if kind == "notebooks":
        export_notebooks(destination, (study, summary))
    write_json(
        destination / "technical_closure.json",
        {
            "technical_complete": True,
            "evidence": "historical_and_predictive_separate",
            "source_study": str(study),
            "config_experiment": config.experiment,
            "structural_certification": False,
        },
    )
    write_json(
        destination / "provenance.json",
        {
            "study": str(study),
            "kind": kind,
            "frozen": read_artifact(study / "freeze.json"),
            "outputs": artifact_signatures(destination),
        },
    )
    return destination


def export_notebooks(destination: Path, content: tuple[Path, pd.DataFrame]) -> None:
    """Presenta datos exportados; no ejecuta código ni modifica el estudio."""
    study, summary = content
    for topic in NOTEBOOK_TOPICS:
        title = topic.replace("_", " ")
        text = f"# {title}\n\nEstudio: {study.name}.\n\n"
        text += "A: evidencia histórica proyectada. B: calidad estimada, pendiente de FEM.\n\n"
        selected = summary
        if topic == "datos_temporales":
            config = load_study_config(study)
            manifest = read_artifact(config.data_dir / "manifest.json")
            selected = pd.DataFrame(
                [
                    {
                        "split": name,
                        "rows": diagnostic["rows"],
                        "min_generation": min(diagnostic["generations"]),
                        "max_generation": max(diagnostic["generations"]),
                    }
                    for name, diagnostic in manifest["diagnostics"].items()
                ]
            )
            text += "Las soluciones son índices poblacionales, no un orden de evaluación.\n\n"
        if topic == "predictores":
            selected = pd.read_csv(destination / "predictor_utility_summary.csv")
            text += "Comparación principal por HV/AUC observado. Intervalos de utilidad en predictor_utility_paired_intervals.csv; diferencias de error en predictive_paired_errors.csv y cobertura en category_coverage.csv. No se declara equivalencia ni seguridad estructural.\n\n"
        if topic.startswith("busqueda_"):
            selected = summary.loc[
                summary.experiment.eq("A" if topic.endswith("historica") else "B")
            ]
        csv_text = selected.to_csv(index=False)
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": text},
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "```csv\n" + csv_text + "```",
                },
            ],
        }
        (destination / f"{topic}.ipynb").write_text(
            json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (destination / f"{topic}.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>"
            + html.escape(title)
            + "</title><h1>"
            + html.escape(title)
            + "</h1><p>"
            + html.escape(text)
            + "</p>"
            + selected.to_html(index=False),
            encoding="utf-8",
        )
