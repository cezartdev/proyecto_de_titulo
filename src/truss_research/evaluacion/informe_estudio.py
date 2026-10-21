"""Consolidación de artefactos firmados; reconstruir un informe nunca entrena."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue

METRICS = ("mae", "rmse", "r2", "bias", "p95_absolute_error")

SELECTION_METRICS = (
    "normalized_auc",
    "relative_hypervolume",
    "pareto_recall",
    "igd_plus",
    "observed_failures",
    "decision_seconds",
    "preparation_seconds",
    "strategy_seconds",
    "queries_to_target",
    "attempts_to_target",
    "target_reached",
    "attempts",
    "unique_queries",
    "repeat_attempts",
    "repeat_rate",
    "penalized_return",
)


def read_object(path: Path) -> dict[str, JsonValue]:
    return read_artifact(path)


def signed_records(directory: Path) -> tuple[dict[str, JsonValue], ...]:
    from truss_research.optimizacion.estudio_estado import (
        TaskRecord,
        read_study,
        verify_task,
    )

    state = read_study(directory)
    for run in cast(list[dict[str, JsonValue]], state.get("execution_runs", [])):
        if (
            run.get("telemetry_signature")
            and source_digest(Path(str(run["telemetry_file"])))
            != run["telemetry_signature"]
        ):
            raise ValueError("Cambió la telemetría de la campaña")
    records = cast(list[dict[str, JsonValue]], state["tasks"])
    complete = tuple(record for record in records if record["status"] == "complete")
    preparation = cast(list[dict[str, JsonValue]], state.get("preparation_tasks", []))
    for record in preparation:
        if record["status"] == "complete":
            verify_task(TaskRecord(**record))
    for record in complete:
        root = Path(str(record["directory"]))
        expected = record["signatures"]
        actual = {
            path.relative_to(root).as_posix(): source_digest(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        if not expected or actual != expected:
            raise ValueError(f"Cambió un artefacto de {record['key']}")
        verify_task(TaskRecord(**record))
    if state.get("test_started") is True:
        validate_freeze(directory, state)
    return complete


def validate_freeze(directory: Path, state: dict[str, JsonValue]) -> None:
    from truss_research.surrogates.configuracion import MODEL_NAMES

    frozen = read_object(directory / "freeze.json")
    for key in ("config", "dataset_signature", "source_signature", "versions"):
        if frozen[key] != state[key]:
            raise ValueError("El manifiesto congelado no coincide con el estudio")
    records = cast(list[dict[str, JsonValue]], state["tasks"])
    lookup = {str(record["key"]): record for record in records}
    lookup.update(
        {
            str(record["key"]): record
            for record in cast(
                list[dict[str, JsonValue]], state.get("preparation_tasks", [])
            )
        }
    )
    artifacts = cast(dict[str, dict[str, JsonValue]], frozen["artifacts"])
    for key, artifact in artifacts.items():
        record = lookup.get(key)
        if (
            record is None
            or record["status"] != "complete"
            or any(
                record[name] != artifact[name] for name in ("directory", "signatures")
            )
        ):
            raise ValueError("Cambió una ejecución utilizada para congelar decisiones")
    config = cast(dict[str, JsonValue], state["config"])
    scores: dict[str, list[float]] = {}
    for record in records:
        key = str(record["key"])
        if not key.startswith("predictor/"):
            continue
        _, model, _ = key.split("/")
        metadata = read_object(Path(str(record["directory"])) / "run.json")
        scores.setdefault(model, []).append(float(str(metadata["normalized_mae"])))
    order = cast(list[str], config["models"])
    winner = min(
        order, key=lambda name: (float(np.mean(scores[name])), MODEL_NAMES.index(name))
    )
    expected = lookup[f"predictor/{winner}/{config['representative_seed']}"][
        "directory"
    ]
    if frozen["winner"] != expected:
        raise ValueError(
            "El predictor congelado no coincide con la selección de validación"
        )


def collect_predictors(
    records: tuple[dict[str, JsonValue], ...], destination: Path
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    details: dict[str, list[pd.DataFrame]] = {
        "category_metrics": [],
        "region_metrics": [],
        "cv_fold_metrics": [],
        "prediction_intervals": [],
    }
    costs: list[dict[str, JsonValue]] = []
    compliance: list[dict[str, JsonValue]] = []
    specialized: list[dict[str, JsonValue]] = []
    for record in records:
        key = str(record["key"])
        if not key.startswith(("predictor/", "test/predictor/")):
            continue
        root = Path(str(record["directory"]))
        metadata = read_object(root / "run.json")
        settings = cast(dict[str, JsonValue], metadata["config"])
        identity = {
            "model": str(settings["model"]),
            "seed": int(str(settings["seed"])),
            "split": str(metadata["evaluation_split"]),
            "role": str(metadata.get("role", "study_candidate")),
            "winner_eligible": metadata.get("winner_eligible", True),
        }
        if identity["split"] == "validation":
            specialized.extend(
                {
                    **identity,
                    "target": item["target"],
                    "family": item["family"],
                    "trial_count": item["trial_count"],
                    "search_reused": item["search_reused"],
                    "parameters": str(item["parameters"]),
                }
                for item in metadata.get("targetwise_selection", [])
            )
        rows.append(pd.read_csv(root / "metrics.csv").assign(**identity))
        costs.append(
            {
                **identity,
                "normalized_mae": metadata["normalized_mae"],
                "fit_seconds": metadata["fit_seconds"],
                "crossfit_seconds": metadata["crossfit_seconds"],
                "tuning_seconds": metadata["tuning_seconds"],
                "trial_count": metadata["trial_count"],
                "nested_trial_count": metadata.get("nested_trial_count", 0),
                "search_reused": metadata["search_reused"],
                "cv_reused": metadata["cv_reused"],
                "reuse_seconds": metadata.get("reuse_seconds", 0),
                "prediction_seconds": metadata.get("prediction_seconds", 0),
                **cast(dict[str, JsonValue], metadata["timing"]),
            }
        )
        compliance.append({**identity, "report": read_object(root / "compliance.json")})
        for name, frames in details.items():
            path = root / f"{name}.csv"
            if path.exists() and not (
                name == "cv_fold_metrics" and metadata["cv_reused"]
            ):
                frames.append(pd.read_csv(path).assign(**identity))
    metrics = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if metrics.empty:
        return metrics
    metrics.to_csv(destination / "predictive_runs.csv", index=False)
    metrics.groupby(["split", "model", "target"])[list(METRICS)].agg(
        ["mean", "std", "count"]
    ).to_csv(destination / "predictive_summary.csv")
    pd.DataFrame(costs).to_csv(destination / "predictive_costs.csv", index=False)
    if specialized:
        pd.DataFrame(specialized).to_csv(
            destination / "targetwise_components.csv", index=False
        )
    write_object(destination / "predictive_compliance.json", compliance)
    for name, frames in details.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                destination / f"{name}.csv", index=False
            )
    if details["prediction_intervals"]:
        intervals = pd.concat(details["prediction_intervals"], ignore_index=True)
        keys = ["split", "model", "target", "nominal_coverage"]
        intervals.groupby(keys).agg(
            empirical_coverage=("covered", "mean"),
            mean_width=("width", "mean"),
            unique_designs=("iter_id", "nunique"),
        ).reset_index().to_csv(destination / "interval_summary.csv", index=False)
        for region in ("web_config", "distance_region"):
            if region in intervals:
                intervals.groupby([*keys, region]).agg(
                    empirical_coverage=("covered", "mean"),
                    mean_width=("width", "mean"),
                    unique_designs=("iter_id", "nunique"),
                ).reset_index().to_csv(
                    destination / f"interval_by_{region}.csv", index=False
                )
    return metrics


def predictive_diagnostics(
    records: tuple[dict[str, JsonValue], ...], destination: Path
) -> None:
    from truss_research.cerchas.schemas import OUTPUT_COLUMNS
    from truss_research.datos.procesados import load_design_split
    from truss_research.evaluacion.auditoria_estudio import (
        error_by_distance,
        nearest_designs,
    )
    from truss_research.evaluacion.metricas import paired_prediction_errors
    from truss_research.evaluacion.robustez import (
        export_error_diagnostics,
        export_evidence_context,
    )
    from truss_research.surrogates.configuracion import response_scale

    predictions: list[pd.DataFrame] = []
    variations: list[dict[str, str | int]] = []
    predictor_records = tuple(
        record
        for record in records
        if str(record["key"]).startswith(("predictor/", "test/predictor/"))
    )
    if not predictor_records:
        return
    first = read_object(Path(str(predictor_records[0]["directory"])) / "run.json")
    data = load_design_split(Path(str(first["config"]["data_dir"])))
    for record in predictor_records:
        root = Path(str(record["directory"]))
        metadata = read_object(root / "run.json")
        split = str(metadata["evaluation_split"])
        settings = metadata["config"]
        predictions.append(
            pd.read_csv(root / f"{split}_predictions.csv").assign(
                split=split, model=settings["model"], seed=settings["seed"]
            )
        )
    table = pd.concat(predictions, ignore_index=True)
    export_error_diagnostics(table, destination)
    export_evidence_context(data, destination)
    prediction_columns = [f"pred_{name}" for name in OUTPUT_COLUMNS]
    dispersion = table.groupby(["split", "model", "iter_id"])[prediction_columns].agg(
        ["mean", "std", "min", "max", "count"]
    )
    dispersion.columns = [
        f"{target}_{statistic}" for target, statistic in dispersion.columns
    ]
    dispersion.reset_index().to_csv(
        destination / "predictive_seed_dispersion.csv", index=False
    )
    paired_prediction_errors(
        table, response_scale(data.training.loc[:, list(OUTPUT_COLUMNS)])
    ).to_csv(destination / "predictive_paired_errors.csv", index=False)
    for (split, model), group in table.groupby(["split", "model"]):
        unique = {
            frame.sort_values("iter_id")[[f"pred_{name}" for name in OUTPUT_COLUMNS]]
            .to_numpy()
            .tobytes()
            for _, frame in group.groupby("seed")
        }
        variations.append(
            {
                "split": split,
                "model": model,
                "training_seeds": group["seed"].nunique(),
                "distinct_prediction_sets": len(unique),
                "interpretation": "single_training_seed_variation_not_estimated"
                if group["seed"].nunique() == 1
                else "deterministic_under_fixed_configuration"
                if len(unique) == 1
                else "final_fit_variation_fixed_split_and_parameters",
                "unique_designs": group["iter_id"].nunique(),
            }
        )
    pd.DataFrame(variations).to_csv(
        destination / "predictive_seed_variation.csv", index=False
    )
    neighbors, _ = nearest_designs(data)
    neighbors.to_csv(destination / "nearest_designs.csv", index=False)
    error_by_distance(records, neighbors).to_csv(
        destination / "errors_by_distance.csv", index=False
    )


def collect_selection(
    records: tuple[dict[str, JsonValue], ...],
    destination: Path,
    adaptive: Path | None = None,
) -> pd.DataFrame:
    decisions: list[pd.DataFrame] = []
    rankings: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    costs: list[dict[str, JsonValue]] = []
    for record in records:
        key, root = str(record["key"]), Path(str(record["directory"]))
        split = "test" if key.startswith("test/") else "validation"
        if "ranking/" in key:
            parts = key.split("/")
            rankings.append(
                pd.read_csv(root / "summary.csv").assign(
                    model=parts[-2], predictor_seed=int(parts[-1]), split=split
                )
            )
        if key not in ("decision/validation", "test/decision"):
            continue
        costs.append({"split": split, **read_object(root / "run.json")["timing"]})
        decisions.append(
            pd.read_csv(root / "summary.csv").assign(
                split=split, evidence_phase="original"
            )
        )
        trace = pd.read_csv(root / "traces.csv")
        reference = decisions[-1].set_index(["episode_id", "comparison_id"])[
            "reference_hypervolume"
        ]
        trace["reference_hypervolume"] = trace.set_index(
            ["episode_id", "comparison_id"]
        ).index.map(reference)
        trace["relative_hypervolume"] = trace["hypervolume"].div(
            trace["reference_hypervolume"].where(trace["reference_hypervolume"] > 0)
        )
        traces.append(trace.assign(split=split, evidence_phase="original"))
    if adaptive is not None:
        additions = pd.read_csv(adaptive / "summary.csv")
        decisions.append(additions)
        added_traces = pd.read_csv(adaptive / "traces.csv")
        reference = additions.set_index(["split", "episode_id", "comparison_id"])[
            "reference_hypervolume"
        ]
        added_traces["reference_hypervolume"] = added_traces.set_index(
            ["split", "episode_id", "comparison_id"]
        ).index.map(reference)
        added_traces["relative_hypervolume"] = added_traces["hypervolume"].div(
            added_traces["reference_hypervolume"].where(
                added_traces["reference_hypervolume"] > 0
            )
        )
        traces.append(added_traces)
    if rankings:
        ranking = pd.concat(rankings, ignore_index=True)
        ranking.to_csv(destination / "ranking_runs.csv", index=False)
        ranking.groupby(["split", "model", "predictor_seed", "budget"])[
            ["normalized_auc", "relative_hypervolume"]
        ].mean().to_csv(destination / "ranking_per_seed.csv")
    if not decisions:
        return pd.DataFrame()
    result = pd.concat(decisions, ignore_index=True)
    pd.DataFrame(costs).to_csv(destination / "selection_shared_costs.csv", index=False)
    result["target_reached"] = np.where(
        result["reference_hypervolume"] > 0,
        result["queries_to_target"].notna().astype(float),
        np.nan,
    )
    result.to_csv(destination / "selection_runs.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        destination / "selection_traces.csv", index=False
    )
    per_seed = (
        result.groupby(
            ["split", "strategy", "policy_training_seed", "budget"], dropna=False
        )[list(SELECTION_METRICS)]
        .mean()
        .reset_index()
    )
    per_seed.to_csv(destination / "selection_per_seed.csv", index=False)
    catalog_blocks = (
        result.groupby(["split", "strategy", "catalog_seed", "budget"], dropna=False)[
            list(SELECTION_METRICS)
        ]
        .mean()
        .reset_index()
    )
    catalog_blocks.to_csv(destination / "selection_catalog_blocks.csv", index=False)
    catalog_blocks.groupby(["split", "strategy", "budget"])[
        list(SELECTION_METRICS)
    ].agg(["mean", "std", "count"]).to_csv(destination / "selection_summary.csv")
    result.groupby(
        ["split", "strategy", "policy_training_seed", "catalog_seed", "budget"],
        dropna=False,
    )[list(SELECTION_METRICS)].mean().to_csv(
        destination / "selection_catalog_variation.csv"
    )
    result.groupby(["split", "strategy", "budget"])[
        list(SELECTION_METRICS)
    ].count().to_csv(destination / "selection_defined_counts.csv")
    paired_comparisons(result).to_csv(
        destination / "paired_differences.csv", index=False
    )
    return result


def paired_comparisons(table: pd.DataFrame) -> pd.DataFrame:
    from truss_research.evaluacion.incertidumbre_seleccion import (
        paired_selection_differences,
    )

    return paired_selection_differences(table)


def collect_explanations(
    records: tuple[dict[str, JsonValue], ...], destination: Path
) -> pd.DataFrame:
    from scipy.stats import spearmanr

    frames: list[pd.DataFrame] = []
    values: list[pd.DataFrame] = []
    costs: list[dict[str, JsonValue]] = []
    for record in records:
        key = str(record["key"])
        if not key.startswith("explain/"):
            continue
        _, model, seed = key.split("/")
        root = Path(str(record["directory"]))
        metadata = read_object(root / "run.json")
        values.append(
            pd.read_csv(root / "shap_values.csv").assign(model=model, seed=int(seed))
        )
        costs.append(
            {
                "model": model,
                "seed": int(seed),
                "method": metadata["method"],
                "explanation_seconds": metadata.get("explanation_seconds"),
            }
        )
        frames.append(
            pd.read_csv(root / "global_importance.csv").assign(
                model=model,
                seed=int(seed),
                method=read_object(root / "run.json")["method"],
            )
        )
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(destination / "shap_global.csv", index=False)
    local = pd.concat(values, ignore_index=True)
    local.to_csv(destination / "shap_values.csv", index=False)
    examples = sorted(local["iter_id"].unique())[:3]
    local[local["iter_id"].isin(examples)].to_csv(
        destination / "shap_examples.csv", index=False
    )
    pd.DataFrame(costs).to_csv(destination / "shap_costs.csv", index=False)
    stability = []
    for (model, target), group in table.groupby(["model", "target"]):
        pivot = group.pivot(index="variable", columns="seed", values="abs_shap")
        for left, seed in enumerate(pivot.columns):
            for other in pivot.columns[left + 1 :]:
                value = (
                    float(spearmanr(pivot[seed], pivot[other]).statistic)
                    if pivot[seed].nunique() > 1 and pivot[other].nunique() > 1
                    else np.nan
                )
                stability.append(
                    {
                        "model": model,
                        "target": target,
                        "seed": seed,
                        "other_seed": other,
                        "spearman": value,
                    }
                )
    pd.DataFrame(
        stability, columns=["model", "target", "seed", "other_seed", "spearman"]
    ).to_csv(destination / "shap_stability.csv", index=False)
    return table


def write_object(path: Path, payload: JsonValue) -> None:
    from truss_research.surrogates.experimentos import write_json

    write_json(path, payload)


def markdown_table(table: pd.DataFrame) -> str:
    values = table.round(5).astype(object).where(table.notna(), "—")
    return "\n".join(
        [
            "| " + " | ".join(map(str, table.columns)) + " |",
            "| " + " | ".join("---" for _ in table.columns) + " |",
            *(
                "| " + " | ".join(map(str, row)) + " |"
                for row in values.itertuples(index=False, name=None)
            ),
        ]
    )


def summarize_study(directory: Path, destination: Path) -> Path:
    from shutil import copyfile, copytree, ignore_patterns

    from truss_research.evaluacion.costos import export_effective_costs
    from truss_research.evaluacion.incertidumbre_seleccion import selection_intervals
    from truss_research.evaluacion.informe_costos import attach_benchmark
    from truss_research.evaluacion.seleccion_adaptativa import (
        EVIDENCE_NOTE,
        latest_adaptive_selection,
    )
    from truss_research.evaluacion.utilidad_predictiva import export_prediction_utility
    from truss_research.optimizacion.estudio_estado import load_study_config

    if destination.exists():
        raise ValueError("Elegir un destino nuevo para el informe")
    records = signed_records(directory)
    state = read_object(directory / "study.json")
    destination.mkdir(parents=True, exist_ok=False)
    from truss_research.evaluacion.robustez import export_convergence

    export_convergence(state, destination)
    from truss_research.evaluacion.evolucion_datos import export_evolution_history

    export_evolution_history(load_study_config(directory).data_dir, destination)
    metrics = collect_predictors(records, destination)
    predictive_diagnostics(records, destination)
    adaptive = (
        latest_adaptive_selection(directory)
        if (directory / "freeze.json").is_file()
        else None
    )
    if adaptive is not None:
        copytree(
            adaptive,
            destination / "adaptive",
            ignore=ignore_patterns("provenance.json", "preparation", "blocks"),
        )
    selection = collect_selection(records, destination, adaptive)
    if (directory / "freeze.json").is_file() and (
        destination / "prediction_intervals.csv"
    ).is_file():
        export_selected_intervals(directory, destination)
    from truss_research.evaluacion.sensibilidad import latest_sensitivity

    sensitivity = (
        latest_sensitivity(directory) if (directory / "freeze.json").is_file() else None
    )
    if sensitivity is not None:
        for name in (
            "objective_sensitivity.csv",
            "catalog_summary.csv",
            "serviceability_sensitivity.csv",
        ):
            copyfile(sensitivity / name, destination / name)
        grouped = sensitivity / "generalization"
        for path in grouped.glob("*.csv"):
            copyfile(path, destination / f"chronological_{path.name}")
    from truss_research.evaluacion.informe_analisis import attach_analysis

    learning_curves = attach_analysis(directory, destination)

    utility = export_prediction_utility(records, destination)
    preparation = tuple(
        cast(list[dict[str, JsonValue]], state.get("preparation_tasks", []))
    )
    export_effective_costs(records, destination, preparation)
    benchmark = attach_benchmark(directory, destination)
    if not selection.empty:
        selection_intervals(selection, load_study_config(directory).analysis).to_csv(
            destination / "selection_intervals.csv", index=False
        )
    collect_explanations(records, destination)
    from truss_research.evaluacion.auditoria_estudio import diagnostic_inventory

    diagnostic_inventory(records).to_csv(destination / "diagnostics.csv", index=False)
    execution = pd.DataFrame(
        [
            {name: record[name] for name in ("key", "status", "seconds", "error")}
            for record in (
                *cast(list[dict[str, JsonValue]], state["tasks"]),
                *preparation,
            )
        ]
    )
    execution.to_csv(destination / "execution.csv", index=False)
    costs = (
        pd.read_csv(destination / "predictive_costs.csv")
        if not metrics.empty
        else pd.DataFrame()
    )
    comparison = (
        costs.groupby(["split", "model"])["normalized_mae"]
        .agg(["mean", "std", "count"])
        .reset_index()
        if not costs.empty
        else pd.DataFrame()
    )
    if not comparison.empty:
        comparison["winner_eligible"] = True
        comparison["role"] = "study_candidate"
    comparison.to_csv(destination / "family_comparison.csv", index=False)
    from truss_research.evaluacion.informe_rl import export_policy_diagnostics

    export_policy_diagnostics(records, destination)
    from truss_research.datos.procedencia import confirmations_digest
    from truss_research.evaluacion.robustez import EVIDENCE_NOTES

    sections = [
        "# Resultados del estudio de predicción y selección de cerchas",
        "La comparación RL incluye familias y ablaciones declaradas en el protocolo. policy_training_summary.csv identifica transiciones, actualizaciones, muestras procesadas, parámetros y checkpoint seleccionado. Las curvas frente a CPU y duración excluyen validación; igualdad de transiciones no implica igualdad de costo. Las pérdidas de distintos algoritmos no son comparables. La evaluación usa argmax disponible, también para PPO.",
        f"Etiqueta del protocolo: {state['config']['label']}. Las comprobaciones de funcionamiento no son experimentos de tesis.",
        "Este informe se reconstruye a partir de artefactos firmados. No ejecuta entrenamiento.",
        "La estabilidad se limita a variación del ajuste final con partición e hiperparámetros fijos. No mide estabilidad cinemática. La procedencia física y las definiciones de respuestas siguen pendientes en docs/SCOPE.md.",
        f"Tareas completas: {len(records)}. Evaluación de prueba iniciada: {state['test_started']}.",
        *EVIDENCE_NOTES.values(),
        "error_tail_summary.csv y largest_prediction_errors.csv separan extremos e influencia de las métricas principales. category_coverage.csv y physical_provenance.csv delimitan cobertura y afirmaciones físicas.",
        "## Predicción",
        "TabPFN participa como familia elegible con parámetros fijos y sin Optuna. Puede alimentar RL, el compuesto y la selección adaptativa. Su preentrenamiento externo no se mide.",
        markdown_table(comparison)
        if not comparison.empty
        else "Sin resultados predictivos completos.",
        "El criterio es MAE dividido por la desviación estándar de entrenamiento, promediado entre respuestas. La familia se selecciona con la media de validación entre semillas; el predictor de decisión usa la semilla representativa fijada en el protocolo.",
        "La búsqueda y las OOF anidadas se comparten entre semillas de cada familia. predictive_seed_variation.csv informa conjuntos de predicciones distintos. predictive_paired_errors.csv remuestrea diseños pareados 2.000 veces después de promediar pérdidas entre semillas; sus intervalos son descriptivos y condicionados a estos modelos y datos, sin interpretación confirmatoria ni corrección por comparaciones múltiples.",
        "errors_by_distance.csv muestra errores por distancia al entrenamiento; las transformaciones y regiones se fijan únicamente con entrenamiento.",
        "predictive_seed_dispersion.csv informa media, desviación muestral, rango y cantidad de semillas por diseño y respuesta. Con una sola semilla la desviación queda indefinida; predicciones idénticas entre varias semillas tienen desviación cero.",
        "## Selección",
        "La medida principal es AUC normalizada, que prioriza calidad temprana. RL usa incrementos marginales ponderados equivalentes al AUC, con ablaciones trapezoidales, descuento 1 y normalizador privado solo en entrenamiento. Las referencias completas se reservan a métricas retrospectivas durante evaluación.",
        "La selección principal bloquea repeticiones. rl_repeat_control y rl_repeat_penalty permiten intentos repetidos sin otra consulta al evaluador; la segunda resta λ/B. El AUC se mide sobre intentos y el retorno penalizado se informa por separado. Intentos, consultas únicas y repeticiones no son intercambiables. Las comparaciones entre ambas variantes y rl conservan pares de semillas y catálogos.",
        "evolution_history.csv y evolution_history.png describen novedad, diversidad y calidad por generación de la trayectoria original. Sus apariciones repetidas no se ponderan como nuevas muestras ni permiten deducir un ahorro FEM medido. El piloto opcional make proposals exporta evidencia predictiva en otro destino, sin incorporarla a estas tablas.",
        "Se minimizan masa y desplazamiento y se maximiza utilización promedio simultáneamente, respetando los límites operativos. El frente de referencia es empírico y restringido a cada catálogo; no acredita optimalidad global.",
        markdown_table(
            selection.groupby(["split", "strategy", "budget"])[
                ["normalized_auc", "relative_hypervolume", "target_reached"]
            ]
            .mean()
            .reset_index()
        )
        if not selection.empty
        else "Sin resultados de decisión completos.",
        "Las tablas por semilla separan variación de entrenamiento y de catálogos. Las consultas al 90 % se promedian solo entre episodios que lo alcanzan; target_reached informa la fracción alcanzada entre referencias positivas. Los conteos de métricas definidas se exportan por separado.",
        "selection_intervals.csv remuestrea bloques de catálogo y semillas de política conservando pares; sus intervalos percentiles del 95 % son descriptivos y condicionados al dataset. selection_summary.csv resume bloques de catálogo también para baselines.",
        "## Selección adaptativa con incertidumbre",
        EVIDENCE_NOTE
        + " Los resultados ensemble_mean, ensemble_ucb y sus variantes online se incorporan a las tablas de selección. adaptive/ADAPTIVE.md conserva comparaciones pareadas y costos adicionales, sin repetir los ajustes del estudio. Los tiempos de esta extensión no se mezclan con el benchmark oficial ni se interpretan como comparables a tiempos históricos recogidos bajo otra concurrencia."
        if adaptive is not None
        else "Extensión opcional pendiente: make adaptive con el mismo ESTUDIO; después make report y make notebooks. No requiere reentrenar.",
        "## Precisión y utilidad",
        markdown_table(utility)
        if not utility.empty
        else "Sin rankings completos para relacionar precisión y utilidad.",
        "prediction_utility_spearman.csv y las figuras de error frente a AUC usan medias por familia y presupuesto. Informan cantidad de familias y motivos de indefinición, sin pruebas de significancia.",
        "## Costos efectivos",
        "Benchmark pendiente: ejecutar make benchmark y después make report."
        if benchmark is None
        else "Benchmark completo: consultar benchmark/COSTS.md, sus tablas y figuras. resource_phases.csv registra CPU, duración y RSS por fase e intento; research_cost.csv cuenta únicamente totales de tarea.",
        "effective_costs.csv separa fases y cost_totals.csv las consolida. task_wall_seconds es el total acumulado por tareas, incluyendo intentos; no se suma nuevamente a los componentes. Los registros de prueba no imputan búsqueda, OOF ni ajustes previos. La validación de políticas se separa de optimización. No se atribuye aceleración frente al solver original.",
        "Con paralelismo, la suma de duraciones representa trabajo acumulado y no tiempo transcurrido de campaña. campaign_execution.csv registra las invocaciones y worker_jobs.csv permite verificar concurrencia. La memoria máxima corresponde a un trabajador y los tiempos de entrenamiento se observan bajo contención; inferencia y selección se comparan secuencialmente.",
        "## Explicabilidad y límites",
        "strategy_seconds incluye preparación de estrategia y elecciones por episodio. selection_shared_costs.csv registra por separado, una vez por comparación, preparación del banco, predicción cacheada y carga de políticas; no se multiplican por episodio.",
        "diagnostics.csv registra advertencias finales, de búsqueda y OOF, además de alternativas SHAP. La convergencia no se deduce de obtener predicciones finitas.",
        "SHAP describe contribuciones a las predicciones en unidades originales; no demuestra causalidad. La estabilidad entre semillas se informa mediante correlación de rangos de importancia.",
        "Los CSV separan validación y prueba; los notebooks generan sus figuras. La prueba ya observada no constituye una validación externa nueva. La etiqueta de comprobación identifica ejecuciones técnicas breves.",
        "Las repeticiones sobre un mismo banco no aumentan la cantidad de diseños independientes. La media y desviación entre semillas son descriptivas, no intervalos de confianza poblacionales. No se realizan pruebas de significancia usando episodios superpuestos como muestras independientes.",
        "La selección consulta exclusivamente diseños registrados. Sus resultados no prueban optimización de geometrías nuevas, seguridad por barra ni ahorro temporal frente al software de origen desconocido. Las categorías raras y la proximidad entre diseños limitan la generalización.",
    ]
    (destination / "REPORT.md").write_text(
        "\n\n".join(sections) + "\n", encoding="utf-8"
    )
    write_object(
        destination / "provenance.json",
        {
            "study": str(directory.resolve()),
            "state_signature": source_digest(directory / "study.json"),
            "freeze_signature": source_digest(directory / "freeze.json")
            if (directory / "freeze.json").exists()
            else None,
            "complete_tasks": len(records),
            "physical_provenance_signature": confirmations_digest(),
            "benchmark_dir": str(benchmark) if benchmark is not None else None,
            "benchmark_signature": source_digest(benchmark / "provenance.json")
            if benchmark is not None
            else None,
            "adaptive_dir": str(adaptive) if adaptive is not None else None,
            "adaptive_signature": source_digest(adaptive / "provenance.json")
            if adaptive is not None
            else None,
            "sensitivity_dir": str(sensitivity) if sensitivity is not None else None,
            "learning_curves_dir": str(learning_curves)
            if learning_curves is not None
            else None,
            "learning_curves_signature": source_digest(
                learning_curves / "provenance.json"
            )
            if learning_curves is not None
            else None,
            "sensitivity_signature": source_digest(sensitivity / "provenance.json")
            if sensitivity is not None
            else None,
            "report_signatures": {
                path.relative_to(destination).as_posix(): source_digest(path)
                for path in destination.rglob("*")
                if path.is_file()
            },
        },
    )
    return destination


def export_selected_intervals(study: Path, destination: Path) -> None:
    """Cobertura del predictor congelado en diseños consultados, sin garantía adaptativa."""
    winner = Path(str(read_object(study / "freeze.json")["winner"]))
    settings = read_object(winner / "run.json")["config"]
    intervals = pd.read_csv(destination / "prediction_intervals.csv")
    intervals = intervals.loc[
        intervals.model.eq(settings["model"]) & intervals.seed.eq(settings["seed"])
    ]
    traces = pd.read_csv(destination / "selection_traces.csv")
    measured = traces[["split", "strategy", "episode_id", "iter_id"]].merge(
        intervals, on=["split", "iter_id"], how="left", validate="many_to_many"
    )
    keys = ["split", "strategy", "target", "nominal_coverage"]
    summary = (
        measured.groupby(keys)
        .agg(
            empirical_coverage=("covered", "mean"),
            mean_width=("width", "mean"),
            unique_designs=("iter_id", "nunique"),
            query_observations=("iter_id", "size"),
        )
        .reset_index()
    )
    summary["scope"] = (
        "frozen_predictor_on_selected_designs_not_corrected_predictions; descriptive_no_adaptive_guarantee"
    )
    summary.to_csv(destination / "selected_interval_coverage.csv", index=False)


def latest_study_report(directory: Path) -> Path | None:
    """Localiza un informe vigente y comprueba sus tablas y figuras."""
    from truss_research.datos.procedencia import confirmations_digest

    state_path = directory / "study.json"
    if not state_path.is_file():
        return None
    signature = source_digest(state_path)
    from truss_research.evaluacion.benchmark import latest_benchmark

    benchmark = (
        latest_benchmark(directory) if (directory / "freeze.json").is_file() else None
    )
    benchmark_signature = (
        source_digest(benchmark / "provenance.json") if benchmark is not None else None
    )
    from truss_research.evaluacion.seleccion_adaptativa import latest_adaptive_selection

    adaptive = (
        latest_adaptive_selection(directory)
        if (directory / "freeze.json").is_file()
        else None
    )
    adaptive_signature = (
        source_digest(adaptive / "provenance.json") if adaptive is not None else None
    )
    from truss_research.evaluacion.sensibilidad import latest_sensitivity

    sensitivity = (
        latest_sensitivity(directory) if (directory / "freeze.json").is_file() else None
    )
    sensitivity_signature = (
        source_digest(sensitivity / "provenance.json")
        if sensitivity is not None
        else None
    )
    from truss_research.evaluacion.curvas_aprendizaje import latest_learning_curves

    curves = (
        latest_learning_curves(directory)
        if (directory / "freeze.json").is_file()
        else None
    )
    curves_signature = (
        source_digest(curves / "provenance.json") if curves is not None else None
    )
    from truss_research.evaluacion.registro import results_root

    external = results_root(directory)
    candidates = [
        *(directory / "reports").glob("report_*"),
        *external.glob("informe_*"),
    ]
    for candidate in sorted(
        candidates, key=lambda path: path.stat().st_mtime_ns, reverse=True
    ):
        provenance_path = candidate / "provenance.json"
        if not provenance_path.is_file():
            continue
        provenance = read_object(provenance_path)
        if provenance.get("physical_provenance_signature") != confirmations_digest():
            continue
        if provenance.get("state_signature") != signature:
            continue
        if provenance.get("benchmark_signature") != benchmark_signature:
            continue
        if provenance.get("adaptive_signature") != adaptive_signature:
            continue
        if provenance.get("sensitivity_signature") != sensitivity_signature:
            continue
        if provenance.get("learning_curves_signature") != curves_signature:
            continue
        expected = cast(dict[str, str], provenance.get("report_signatures", {}))
        actual = {
            path.relative_to(candidate).as_posix(): source_digest(path)
            for path in candidate.rglob("*")
            if path.is_file() and path != candidate / "provenance.json"
        }
        if not expected or actual != expected:
            raise ValueError("Cambió una tabla o figura del informe")
        return candidate
    return None
