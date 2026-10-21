"""Errores extremos, cobertura y límites descriptivos sin reajustar modelos."""

from collections.abc import Iterator
from math import ceil
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import OUTPUT_COLUMNS, WEB_CONFIG_VALUES
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.procesados import JsonValue
from truss_research.datos.schemas import TableSplit

TAIL_FRACTION = 0.05
TOP_ERRORS = 5
SCARCE_DESIGNS = 10

EVIDENCE_NOTES = {
    "prediction": "MAE normalizado divide por desviaciones de entrenamiento; no es porcentaje de error relativo. El remuestreo predictivo usa diseños pareados después de promediar pérdidas entre semillas.",
    "selection": "La búsqueda es una aplicación exploratoria secundaria. Sus intervalos remuestrean semillas de aprendizaje y escenarios emparejados; no son nuevos problemas estructurales independientes. El acuerdo entre evaluador y auditor no verifica diseños nuevos.",
    "resources": "Se midieron duración, CPU y RSS máxima muestreada. RSS incluye bibliotecas y puede perder picos breves; no es memoria exclusiva del modelo ni un máximo físico continuo. El tiempo acumulado de trabajos difiere del tiempo transcurrido de campaña.",
    "cpu": "La latencia individual se conserva. Los ceros de CPU cruda pueden estar bajo resolución; la pasada CPU agrupada declara llamadas, granularidad y alcance. CPU de episodio simulado no es CPU exclusiva de decisiones ni costo de un solver.",
    "scope": "La prueba ya observada conserva carácter exploratorio. Se comparan diseños registrados de una fuente y partición fija; no se acredita extrapolación a problemas nuevos, generación de topologías, seguridad por barra ni aceleración del solver original.",
    "stability": "La estabilidad describe ajustes finales con partición e hiperparámetros fijos. Predicciones idénticas entre semillas indican determinismo condicionado, no réplicas independientes ni certeza poblacional.",
    "tails": "Los errores extremos se conservan. Las métricas sin el peor caso son únicamente un diagnóstico de influencia y no sustituyen las métricas principales ni justifican excluir diseños.",
}


def export_error_diagnostics(table: pd.DataFrame, destination: Path) -> None:
    identities = ["split", "model", "seed", "iter_id", "web_config"]
    errors = pd.concat(
        [
            table[identities].assign(
                target=target,
                observed=table[target],
                predicted=table[f"pred_{target}"],
                error=table[f"pred_{target}"] - table[target],
            )
            for target in OUTPUT_COLUMNS
        ],
        ignore_index=True,
    )
    errors["absolute_error"] = errors.error.abs()
    errors["squared_error"] = errors.error.pow(2)
    errors.to_csv(destination / "prediction_errors.csv", index=False)
    summaries: list[dict[str, JsonValue]] = []
    extremes: list[pd.DataFrame] = []
    keys = ["split", "model", "seed", "target"]
    for identity, group in errors.groupby(keys):
        ordered = group.sort_values(
            ["absolute_error", "iter_id"], ascending=[False, True], kind="stable"
        )
        n = len(group)
        total = float(group.squared_error.sum())
        remaining = ordered.iloc[1:]
        summaries.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "unique_designs": group.iter_id.nunique(),
                "mae": float(group.absolute_error.mean()),
                "rmse": float(np.sqrt(group.squared_error.mean())),
                "p95_absolute_error": float(group.absolute_error.quantile(0.95)),
                "p99_absolute_error": float(group.absolute_error.quantile(0.99)),
                "max_absolute_error": float(ordered.absolute_error.iloc[0]),
                "upper_five_percent_mean": float(
                    ordered.absolute_error.iloc[
                        : max(1, ceil(TAIL_FRACTION * n))
                    ].mean()
                ),
                "worst_squared_error_share": float(
                    ordered.squared_error.iloc[0] / total
                )
                if total > 0
                else None,
                "mae_without_worst_diagnostic": float(remaining.absolute_error.mean())
                if n > 1
                else None,
                "rmse_without_worst_diagnostic": float(
                    np.sqrt(remaining.squared_error.mean())
                )
                if n > 1
                else None,
                "interpretation": "diagnostic_only_all_designs_retained",
            }
        )
        extremes.append(
            ordered.head(TOP_ERRORS).assign(
                error_rank=np.arange(1, min(TOP_ERRORS, n) + 1)
            )
        )
    pd.DataFrame(summaries).to_csv(destination / "error_tail_summary.csv", index=False)
    pd.concat(extremes, ignore_index=True).to_csv(
        destination / "largest_prediction_errors.csv", index=False
    )


def coverage_tables(split: TableSplit) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    generations = []
    for name, table in split.as_mapping().items():
        counts = (
            table.groupby("web_config")
            .iter_id.nunique()
            .reindex(WEB_CONFIG_VALUES, fill_value=0)
        )
        frames.append(
            counts.rename("unique_designs")
            .reset_index()
            .assign(
                split=name,
                scarce_evidence=counts.to_numpy() < SCARCE_DESIGNS,
                scarcity_threshold=SCARCE_DESIGNS,
                interpretation="descriptive_threshold_not_statistical_power",
            )
        )
        if "generation" in table:
            generations.append(table[["generation", "iter_id"]].assign(split=name))
    memberships = (
        pd.concat(generations)
        .groupby("generation")
        .agg(partitions=("split", "nunique"), unique_designs=("iter_id", "nunique"))
        if generations
        else pd.DataFrame(columns=["partitions", "unique_designs"])
    )
    memberships["shared_across_partitions"] = memberships.partitions.gt(1)
    return pd.concat(frames, ignore_index=True), memberships.reset_index()


def provenance_table() -> pd.DataFrame:
    from truss_research.datos.procedencia import load_confirmations

    labels = {
        "pending": "Pendiente",
        "author_reported": "Informado por el autor",
        "confirmed": "Confirmado",
    }
    return pd.DataFrame(
        [
            {
                "element": confirmation.element,
                "status": labels[confirmation.status],
                "evidence": confirmation.evidence,
                "confirmation_required": confirmation.confirmation_required,
                "confirmation_source": confirmation.evidence_source,
                "confirmed_at": confirmation.confirmed_at,
            }
            for confirmation in load_confirmations()
        ]
    )


def export_evidence_context(split: TableSplit, destination: Path) -> None:
    coverage, generations = coverage_tables(split)
    coverage.to_csv(destination / "category_coverage.csv", index=False)
    generations.to_csv(destination / "generation_overlap.csv", index=False)
    provenance_table().to_csv(destination / "physical_provenance.csv", index=False)
    (destination / "EVIDENCE.md").write_text(
        "# Alcance de la evidencia\n\n"
        + "\n\n".join(EVIDENCE_NOTES.values())
        + f"\n\nGrupos con menos de {SCARCE_DESIGNS} diseños únicos se señalan como evidencia escasa, sin garantizar suficiencia para grupos mayores. Las confirmaciones físicas requieren fuente y fecha; no se infieren del entrenamiento.\n",
        encoding="utf-8",
    )


def export_tradeoffs(destination: Path) -> None:
    strategy = pd.read_csv(destination / "strategy_cost_summary.csv")
    costs = pd.read_csv(destination / "amortization.csv")
    preparation = costs.groupby(
        ["split", "catalog_size", "strategy", "budget", "scenario", "reuse_count"],
        as_index=False,
    )[["preparation_seconds", "loading_seconds", "total_seconds"]].mean()
    result = preparation.merge(
        strategy[
            [
                "split",
                "catalog_size",
                "strategy",
                "budget",
                "normalized_auc_mean",
                "relative_hypervolume_mean",
                "independent_seconds_mean",
                "cached_seconds_mean",
            ]
        ],
        on=["split", "catalog_size", "strategy", "budget"],
        validate="many_to_one",
    )
    result["interpretation"] = (
        "time_tradeoffs_do_not_imply_equal_quality_or_solver_acceleration"
    )
    result.to_csv(destination / "decision_tradeoffs.csv", index=False)


def optimizer_events(
    value: JsonValue, location: str = "run"
) -> Iterator[tuple[str, dict[str, JsonValue]]]:
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from optimizer_events(child, f"{location}/{index}")
    if not isinstance(value, dict):
        return
    if "attempts" in value and "optimizer" in value:
        for attempt in cast(list[dict[str, JsonValue]], value["attempts"]):
            yield location, {"target_index": value["target_index"], **attempt}
        return
    for key, child in value.items():
        if key.endswith("convergence_failure") and isinstance(child, list):
            for attempt in cast(list[dict[str, JsonValue]], child):
                yield f"{location}/{key}", attempt
        else:
            yield from optimizer_events(child, f"{location}/{key}")


def export_convergence(state: dict[str, JsonValue], destination: Path) -> None:
    records = cast(list[dict[str, JsonValue]], state.get("preparation_tasks", []))
    rows: list[dict[str, JsonValue]] = []
    for task in records:
        if task["status"] != "complete" or not str(task["key"]).startswith(
            "predictor/"
        ):
            continue
        payload = read_artifact(Path(str(task["directory"])) / "run.json")
        rows.extend(
            {"task": task["key"], "location": location, **event}
            for location, event in optimizer_events(payload)
        )
    for task in cast(list[dict[str, JsonValue]], state["tasks"]):
        if task["status"] != "complete" or not str(task["key"]).startswith(
            "predictor/"
        ):
            continue
        payload = read_artifact(Path(str(task["directory"])) / "run.json")
        rows.extend(
            {"task": task["key"], "location": location, **event}
            for location, event in optimizer_events(payload.get("convergence", []))
        )
    pd.DataFrame(
        rows,
        columns=[
            "task",
            "location",
            "target_index",
            "attempt",
            "success",
            "status",
            "message",
            "iterations",
            "evaluations",
            "objective",
            "gradient_max_abs",
            "near_bound",
            "max_line_search",
            "theta",
        ],
    ).to_csv(destination / "convergence_diagnostics.csv", index=False)
