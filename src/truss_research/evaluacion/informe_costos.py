"""Tablas de recursos, costos de uso y amortización sin duplicar dependencias."""

import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.recursos import valid_seconds
from truss_research.optimizacion.estudio_estado import load_study_config, read_study

RESOURCE_COLUMNS = (
    "device",
    "gpu_name",
    "gpu_allocated_mib",
    "gpu_reserved_mib",
    "gpu_peak_allocated_mib",
    "gpu_peak_reserved_mib",
    "gpu_peak_scope",
    "task",
    "attempt",
    "phase",
    "status",
    "wall_seconds",
    "cpu_seconds",
    "initial_rss_mib",
    "peak_rss_mib",
    "peak_increment_mib",
    "memory_interval_ms",
    "samples",
    "cpu_scope",
    "cpu_coverage",
    "cpu_tree_coverage",
    "coverage_policy_version",
    "coverage",
)


def resource_inventory(study: Path) -> pd.DataFrame:
    state = read_study(study)
    rows: list[dict[str, JsonValue]] = []
    records = [
        *cast(list[dict[str, JsonValue]], state["tasks"]),
        *cast(list[dict[str, JsonValue]], state.get("preparation_tasks", [])),
    ]
    for task in records:
        files = cast(list[str], task.get("measurement_files", []))
        signatures = cast(dict[str, str], task.get("measurement_signatures", {}))
        for number, path in enumerate(files, start=1):
            file = Path(path)
            if not file.is_file():
                rows.append(
                    {
                        "task": task["key"],
                        "attempt": number,
                        "phase": "task_total",
                        "coverage": "missing",
                    }
                )
                continue
            if path in signatures and source_digest(file) != signatures[path]:
                raise ValueError("Cambió un registro de recursos del estudio")
            payload = read_artifact(file)
            for phase in cast(list[dict[str, JsonValue]], payload["phases"]):
                complete = (
                    phase["finished"] is True
                    and not phase.get("errors")
                    and phase["cpu_coverage"] == "complete"
                    and phase["memory_coverage"] == "sampled"
                    and all(
                        valid_seconds(float(phase[name]))
                        for name in ("wall_seconds", "cpu_seconds")
                    )
                    and path in signatures
                )
                rows.append(
                    {
                        "task": task["key"],
                        "attempt": number,
                        **phase,
                        "coverage": "complete" if complete else "incomplete",
                    }
                )
        if len(files) != len(cast(list[str], task["attempts"])):
            rows.append(
                {
                    "task": task["key"],
                    "attempt": None,
                    "phase": "task_total",
                    "coverage": "missing",
                }
            )
    return pd.DataFrame(rows, columns=[*RESOURCE_COLUMNS, "errors", "children_seen"])


def export_resource_report(study: Path, destination: Path) -> bool:
    table = resource_inventory(study)
    table.to_csv(destination / "resource_phases.csv", index=False)
    totals = table.loc[table.phase.eq("task_total")]
    summary = totals.groupby("task", as_index=False).agg(
        wall_seconds=("wall_seconds", lambda values: values.sum(min_count=1)),
        cpu_seconds=("cpu_seconds", lambda values: values.sum(min_count=1)),
        peak_rss_mib=("peak_rss_mib", "max"),
        attempts=("attempt", "count"),
    )
    summary.to_csv(destination / "resource_tasks.csv", index=False)
    state = read_study(study)
    runs = cast(list[dict[str, JsonValue]], state.get("execution_runs", []))
    jobs = [
        {"task": task["key"], "attempt": number, **event}
        for task in [*state["tasks"], *state.get("preparation_tasks", [])]
        for number, event in enumerate(task.get("worker_history", []), start=1)
    ]
    pd.DataFrame(
        jobs, columns=["task", "attempt", "pid", "started_at", "finished_at", "success"]
    ).to_csv(destination / "worker_jobs.csv", index=False)
    pd.DataFrame(
        runs, columns=["phase", "status", "wall_seconds", "cpu_workers", "work_seconds"]
    ).to_csv(destination / "campaign_execution.csv", index=False)
    telemetry: list[pd.DataFrame] = []
    for number, run in enumerate(runs, start=1):
        if run.get("telemetry_file"):
            path = Path(str(run["telemetry_file"]))
            if (
                run.get("telemetry_signature")
                and source_digest(path) != run["telemetry_signature"]
            ):
                raise ValueError("Cambió la telemetría de campaña")
            telemetry.append(
                pd.read_csv(path).assign(invocation=number, phase=run["phase"])
            )
    if telemetry:
        pd.concat(telemetry, ignore_index=True).to_csv(
            destination / "campaign_telemetry.csv", index=False
        )
    pd.DataFrame(
        [
            {
                "scope": "research_all_recorded_attempts",
                "wall_seconds": totals.wall_seconds.sum(min_count=1),
                "wall_seconds_scope": "sum_of_job_durations_not_campaign_elapsed",
                "campaign_wall_seconds": sum(float(run["wall_seconds"]) for run in runs)
                if runs
                else None,
                "cpu_workers": load_study_config(study).execution.effective_workers,
                "cpu_seconds": totals.cpu_seconds.sum(min_count=1),
                "peak_rss_mib": totals.peak_rss_mib.max(),
                "peak_rss_scope": "maximum_single_worker_not_total_simultaneous_memory",
                "complete": bool(
                    not table.empty and table.coverage.eq("complete").all()
                ),
            }
        ]
    ).to_csv(destination / "research_cost.csv", index=False)
    return bool(not table.empty and table.coverage.eq("complete").all())


def family_phases(
    resources: pd.DataFrame, context: tuple[str, int, int]
) -> pd.DataFrame:
    model, seed, representative = context
    dependencies = (
        (f"predictor/{model}/{seed}", ("fit", "calibration")),
        (f"predictor/{model}/{representative}", ("tuning", "crossfit")),
    )
    result = pd.concat(
        [
            resources.loc[resources.task.eq(task) & resources.phase.isin(phases)]
            for task, phases in dependencies
        ],
        ignore_index=True,
    )
    stages = resources.loc[
        resources.task.str.startswith(f"predictor/{model}/{representative}/")
        & resources.phase.isin(("tuning", "crossfit"))
    ]
    return pd.concat([result, stages], ignore_index=True)


def family_preparation(resources: pd.DataFrame, context: tuple[str, int, int]) -> float:
    return float(family_phases(resources, context).wall_seconds.sum(min_count=1))


def amortization_rows(
    preparation: float, usage: tuple[float, float, float], counts: tuple[int, ...]
) -> list[dict[str, JsonValue]]:
    loading, prediction, selection = usage
    if not all(valid_seconds(value) for value in (preparation, *usage)):
        raise ValueError("La amortización requiere tiempos finitos y no negativos")
    return [
        {
            "scenario": scenario,
            "reuse_count": count,
            "preparation_seconds": preparation,
            "loading_seconds": loading,
            "prediction_seconds": prediction,
            "selection_seconds": selection,
            "total_seconds": preparation
            + loading
            + count * selection
            + (count if scenario == "independent_catalogs" else 1) * prediction,
        }
        for count in counts
        for scenario in ("independent_catalogs", "repeated_selection_same_catalog")
    ]


def benchmark_summaries(destination: Path) -> None:
    inference = pd.read_csv(destination / "inference_samples.csv")
    groups = ["model", "seed", "split", "operation", "batch_size"]
    summary = inference.groupby(groups, as_index=False).agg(
        median_seconds=("wall_seconds", "median"),
        q25_seconds=("wall_seconds", lambda values: values.quantile(0.25)),
        q75_seconds=("wall_seconds", lambda values: values.quantile(0.75)),
        median_cpu_seconds=("cpu_seconds", "median"),
        repetitions=("wall_seconds", "count"),
    )
    summary["cpu_timing_status"] = np.where(
        summary.median_cpu_seconds.eq(0),
        "below_resolution",
        "quantized_individual_measurement",
    )
    summary.loc[summary.median_cpu_seconds.eq(0), "median_cpu_seconds"] = float("nan")
    summary["designs_per_second"] = summary.batch_size.div(
        summary.median_seconds.where(summary.median_seconds.gt(0))
    ).where(summary.operation.eq("predict"))
    summary.to_csv(destination / "inference_summary.csv", index=False)
    selection = pd.read_csv(destination / "selection_samples.csv")
    groups = [
        "split",
        "catalog_size",
        "strategy",
        "comparison_id",
        "catalog_seed",
        "episode_id",
        "budget",
    ]
    metrics = [
        "selection_seconds",
        "prediction_seconds",
        "independent_seconds",
        "cached_seconds",
        "environment_seconds",
        "decision_cpu_seconds",
        "selection_cpu_seconds",
        "independent_cpu_seconds",
        "simulation_seconds",
        "normalized_auc",
        "relative_hypervolume",
        "queries",
    ]
    grouped = selection.groupby(groups)[metrics]
    median = grouped.median().add_suffix("_median")
    lower = grouped.quantile(0.25).add_suffix("_q25")
    upper = grouped.quantile(0.75).add_suffix("_q75")
    for column in [name for name in median if "cpu_seconds" in name]:
        median[column] = median[column].mask(median[column].eq(0))
    median.join(lower).join(upper).reset_index().to_csv(
        destination / "selection_summary.csv", index=False
    )
    per_alternative = (
        selection.groupby(
            [
                "split",
                "catalog_size",
                "strategy",
                "comparison_id",
                "episode_id",
                "budget",
            ]
        )[metrics]
        .median()
        .reset_index()
    )
    strategy_summary = per_alternative.groupby(
        ["split", "catalog_size", "strategy", "budget"]
    )[metrics].agg(["mean", "std", "count"])
    strategy_summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in strategy_summary.columns
    ]
    for column in [name for name in strategy_summary if "cpu_seconds" in name]:
        strategy_summary[column] = strategy_summary[column].mask(
            strategy_summary[column].eq(0)
        )
    strategy_summary["cpu_timing_status"] = (
        "quantized_sections_use_cpu_operation_summary_for_grouped_measurements"
    )
    strategy_summary.reset_index().to_csv(
        destination / "strategy_cost_summary.csv", index=False
    )


def export_benchmark_report(study: Path, destination: Path) -> None:
    from truss_research.evaluacion.informe_estudio import markdown_table
    from truss_research.evaluacion.precision_cpu import export_cpu_summary

    export_cpu_summary(destination)
    config = load_study_config(study)
    resources = resource_inventory(study)
    complete = export_resource_report(study, destination)
    if not complete:
        raise ValueError(
            "Faltan mediciones completas de recursos; se requiere una campaña nueva"
        )
    benchmark_summaries(destination)
    samples = pd.read_csv(destination / "selection_samples.csv")
    loading = pd.read_csv(destination / "selection_loading.csv")
    frozen = read_artifact(study / "freeze.json")
    winner = read_artifact(Path(str(frozen["winner"])) / "run.json")
    winner_name = str(winner["config"]["model"])
    rows: list[dict[str, JsonValue]] = []
    for (split, size, name, strategy, budget), group in samples.groupby(
        ["split", "catalog_size", "comparison_id", "strategy", "budget"]
    ):
        assisted = bool(group.uses_predictions.iloc[0])
        from truss_research.optimizacion.variantes_rl import VARIANT_REGISTRY

        policy = str(strategy) in VARIANT_REGISTRY
        seed = int(name.rsplit("_", 1)[1]) if policy else config.representative_seed
        preparation = (
            family_preparation(
                resources,
                (winner_name, config.representative_seed, config.representative_seed),
            )
            if assisted or policy
            else 0.0
        )
        if policy:
            preparation += float(
                resources.loc[
                    resources.task.eq(f"policy/{strategy}/{seed}")
                    & resources.phase.eq("task_total"),
                    "wall_seconds",
                ].sum()
            )
        load_seconds = (
            float(
                loading.loc[
                    loading.operation.eq("load_predictor"), "wall_seconds"
                ].sum()
            )
            if assisted
            else 0.0
        )
        if policy:
            load_seconds += float(
                loading.loc[loading.comparison_id.eq(name), "wall_seconds"].sum()
            )
        per_episode = (
            group.groupby("episode_id")[["prediction_seconds", "selection_seconds"]]
            .median()
            .mean()
        )
        usage = (
            load_seconds,
            float(per_episode.prediction_seconds),
            float(per_episode.selection_seconds),
        )
        rows.extend(
            {
                "alternative": str(name),
                "strategy": str(strategy),
                "policy_seed": seed if policy else None,
                "split": str(split),
                "catalog_size": int(size),
                "budget": int(budget),
                "preparation_seconds_scope": "sum_of_required_worker_phases",
                **row,
            }
            for row in amortization_rows(
                preparation, usage, config.measurement.reuse_counts
            )
        )
    inference = pd.read_csv(destination / "inference_summary.csv")
    predictor_rows: list[dict[str, JsonValue]] = []
    for (model, seed), group in inference.groupby(["model", "seed"]):
        phases = family_phases(
            resources, (str(model), int(seed), config.representative_seed)
        )
        preparation = family_preparation(
            resources, (str(model), int(seed), config.representative_seed)
        )
        load_seconds = float(
            group.loc[group.operation.eq("load"), "median_seconds"].iloc[0]
        )
        predictor_rows.extend(
            {
                "model": str(model),
                "seed": int(seed),
                "split": str(row.split),
                "batch_size": int(row.batch_size),
                "preparation_seconds": preparation,
                "preparation_seconds_scope": "sum_of_required_worker_phases",
                "training_timing_scope": "concurrent_cpu_jobs"
                if config.execution.effective_workers > 1
                else "serial_cpu",
                "preparation_cpu_seconds": float(phases.cpu_seconds.sum(min_count=1)),
                "training_peak_rss_mib": float(phases.peak_rss_mib.max()),
                "fit_seconds": float(
                    phases.loc[phases.phase.eq("fit"), "wall_seconds"].sum(min_count=1)
                ),
                "loading_seconds": load_seconds,
                "median_seconds": float(row.median_seconds),
                "median_cpu_seconds": float(row.median_cpu_seconds),
                "q25_seconds": float(row.q25_seconds),
                "q75_seconds": float(row.q75_seconds),
                "designs_per_second": float(row.designs_per_second),
            }
            for row in group.loc[group.operation.eq("predict")].itertuples()
        )
    predictors = pd.DataFrame(predictor_rows)
    memory = pd.read_csv(destination / "inference_memory.csv")
    predictors = predictors.merge(
        memory[
            [
                "model",
                "seed",
                "split",
                "batch_size",
                "initial_rss_mib",
                "peak_rss_mib",
                "peak_increment_mib",
            ]
        ],
        on=["model", "seed", "split", "batch_size"],
        validate="one_to_one",
    )
    sizes = pd.read_csv(destination / "artifact_sizes.csv")
    predictors = predictors.merge(
        sizes.loc[sizes.artifact.eq("predictor"), ["model", "seed", "size_mib"]],
        on=["model", "seed"],
        validate="many_to_one",
    )
    predictors.to_csv(destination / "predictor_costs.csv", index=False)
    family_summary = predictors.groupby(["model", "split", "batch_size"])[
        [
            "preparation_seconds",
            "preparation_cpu_seconds",
            "training_peak_rss_mib",
            "median_seconds",
            "peak_rss_mib",
            "peak_increment_mib",
            "size_mib",
        ]
    ].agg(["mean", "std", "count"])
    family_summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in family_summary.columns
    ]
    family_summary.reset_index().to_csv(
        destination / "predictor_family_summary.csv", index=False
    )
    pd.DataFrame(rows).to_csv(destination / "amortization.csv", index=False)
    from truss_research.evaluacion.robustez import export_tradeoffs

    export_tradeoffs(destination)
    cost_figures(study, destination)
    (destination / "COSTS.md").write_text(
        "# Calidad y costo computacional\n\nCPU agrupada: consultar cpu_operation_summary.csv. La CPU de episodio incluye preparación y entorno simulado y no representa CPU exclusiva de decisiones. Valores individuales bajo resolución se presentan indefinidos; las muestras crudas conservan ceros.\n\n"
        + markdown_table(pd.read_csv(destination / "research_cost.csv"))
        + "\n\nLos totales de investigación suman únicamente task_total por intento; las fases son inclusivas y no se suman nuevamente. "
        "Con concurrencia, esa suma es trabajo acumulado: campaign_execution.csv informa tiempo transcurrido por invocación y worker_jobs.csv los intervalos de cada trabajo. "
        "Los tiempos de entrenamiento se observan bajo contención entre procesos; el benchmark de inferencia y selección es secuencial. "
        "El manifiesto del benchmark conserva la firma del código de medición y la del entrenamiento cuando está disponible; se pueden medir artefactos congelados después de actualizar el código sin repetir ajustes. Las trayectorias principales se verifican contra las guardadas. "
        "La RSS máxima corresponde a un trabajador, no al pico simultáneo del pool. La amortización utiliza trabajo acumulado de preparación. "
        "La preparación por alternativa incluye búsqueda y OOF de su familia, un ajuste y, para RL, la política correspondiente; no imputa otras familias ni réplicas. "
        "La ablación conserva las dependencias del pipeline ejecutado aunque no consuma predicciones al decidir. "
        "independent_catalogs paga predicción por catálogo; repeated_selection_same_catalog paga una predicción compartida. La carga se paga una vez. "
        "Las repeticiones temporales no son muestras estructurales independientes. La mediana se calcula primero por episodio y después se promedia entre episodios. "
        "La RSS es el máximo muestreado del proceso, con bibliotecas y modelos residentes; su línea base se informa y no equivale al tamaño exclusivo de un modelo. "
        "Las curvas temporales separan selección y entorno simulado. El acceso al dataset no representa un solver estructural ni acredita aceleración frente a él. "
        "Las curvas de calidad se calculan después de medir las decisiones; su evaluación retrospectiva queda fuera de esos tiempos.\n",
        encoding="utf-8",
    )


def cost_figures(study: Path, destination: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    state = read_study(study)
    errors = []
    for task in state["tasks"]:
        if not str(task["key"]).startswith("predictor/"):
            continue
        metadata = read_artifact(Path(str(task["directory"])) / "run.json")
        errors.append(
            {
                "model": metadata["config"]["model"],
                "fit_seconds": metadata["fit_seconds"],
                "normalized_mae": metadata["normalized_mae"],
            }
        )
    predictive = pd.DataFrame(errors).groupby("model", as_index=False).mean()
    figure = Figure(figsize=(9, 5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for row in predictive.itertuples():
        axis.scatter(row.fit_seconds, row.normalized_mae, label=row.model)
    axis.set(
        xlabel="Tiempo de ajuste final (s)", ylabel="MAE normalizado de validación"
    )
    axis.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(destination / "prediction_quality_cost.png", dpi=150)
    figure.clear()
    samples = pd.read_csv(destination / "selection_samples.csv")
    traces = pd.read_csv(destination / "selection_time_traces.csv")
    for (split, size, budget), group in samples.groupby(
        ["split", "catalog_size", "budget"]
    ):
        figure = Figure(figsize=(11, 5))
        FigureCanvasAgg(figure)
        quality_axis, curve_axis = figure.subplots(1, 2)
        averages = group.groupby("strategy", as_index=False)[
            ["independent_seconds", "normalized_auc"]
        ].mean()
        for row in averages.itertuples():
            quality_axis.scatter(
                row.independent_seconds, row.normalized_auc, label=row.strategy
            )
        selected = traces.loc[
            traces.split.eq(split)
            & traces.catalog_size.eq(size)
            & traces.budget.eq(budget)
        ]
        averaged = selected.groupby(["strategy", "query"], as_index=False)[
            [
                "independent_cumulative_seconds",
                "decision_cumulative_seconds",
                "environment_cumulative_seconds",
                "relative_hypervolume",
            ]
        ].mean()
        for strategy, values in averaged.groupby("strategy"):
            ordered = values.sort_values("query")
            curve_axis.plot(
                ordered.independent_cumulative_seconds,
                ordered.relative_hypervolume,
                marker=".",
                label=str(strategy),
            )
        averaged.to_csv(
            destination / f"time_curves_{split}_{size}_{budget}.csv", index=False
        )
        quality_axis.set(
            xlabel="Predicción e inicialización y decisiones (s)",
            ylabel="AUC normalizada",
        )
        curve_axis.set(
            xlabel="Tiempo acumulado de predicción y selección (s)",
            ylabel="Hipervolumen relativo",
        )
        curve_axis.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.suptitle(f"{split}: {size} candidatos, presupuesto {budget}")
        figure.tight_layout()
        figure.savefig(
            destination / f"selection_quality_cost_{split}_{size}_{budget}.png", dpi=150
        )
        figure.clear()


def attach_benchmark(study: Path, destination: Path) -> Path | None:
    from truss_research.evaluacion.benchmark import latest_benchmark

    export_resource_report(study, destination)
    benchmark = latest_benchmark(study) if (study / "freeze.json").is_file() else None
    if benchmark is None:
        return None
    target = destination / "benchmark"
    shutil.copytree(benchmark, target, ignore=shutil.ignore_patterns("provenance.json"))
    return benchmark
