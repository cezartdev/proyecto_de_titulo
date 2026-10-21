"""Auditoría retrospectiva sin ajustar predictores ni modificar estudios."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from truss_research.cerchas.schemas import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    WEB_CONFIG_VALUES,
)
from truss_research.datos.artefactos import dataset_identity
from truss_research.datos.auditoria import audit_split_overlap
from truss_research.datos.distancias import design_distances
from truss_research.datos.procedencia import confirmations_digest
from truss_research.datos.procesados import JsonValue, load_design_split, read_manifest
from truss_research.datos.schemas import TableSplit
from truss_research.evaluacion.informe_estudio import (
    markdown_table,
    read_object,
    signed_records,
)
from truss_research.evaluacion.utilidad_predictiva import utility_comparison
from truss_research.surrogates.experimentos import write_json


@dataclass(frozen=True)
class AuditConfig:
    study_dir: Path
    data_dir: Path
    output_dir: Path


def audit_calibration_exclusions(
    records: tuple[dict[str, JsonValue], ...], split: TableSplit
) -> None:
    """Comprueba procedencia de ajuste, reserva y pliegues sin reajustar modelos."""
    if split.calibration is None:
        return
    fitting, calibration = set(split.training.iter_id), set(split.calibration.iter_id)
    for record in records:
        key = str(record["key"])
        if not key.startswith(("predictor/",)):
            continue
        root = Path(str(record["directory"]))
        metadata = read_object(root / "run.json")
        if (
            set(metadata.get("fit_ids", [])) != fitting
            or set(metadata.get("calibration_ids", [])) != calibration
        ):
            raise ValueError(
                "Procedencia de ajuste o calibración incompatible con los datos"
            )
        calibrated = read_object(root / "calibration.json")
        if set(calibrated["calibration_ids"]) != calibration:
            raise ValueError("Los residuos no corresponden a la reserva de calibración")
        oof = pd.read_csv(root / "training_oof.csv")
        if not oof.iter_id.is_unique or set(oof.iter_id) != fitting:
            raise ValueError("Las OOF no cubren exclusivamente el ajuste")
        members = metadata.get("fold_model_signatures", {})
        if set(members) != {f"fold_{int(number)}" for number in oof.fold.unique()}:
            raise ValueError("Los miembros externos no corresponden a las OOF")
        nested_path = root / "nested_folds.json"
        if not nested_path.is_file():
            continue
        folds = read_object(nested_path)["folds"]
        for fold in folds:
            fit, held = set(fold["fit_ids"]), set(fold["held_ids"])
            if fit & held or fit | held != fitting or (fit | held) & calibration:
                raise ValueError(
                    "Un pliegue externo incluye datos excluidos de su ajuste"
                )


def nearest_designs(split: TableSplit) -> tuple[pd.DataFrame, list[float]]:
    training = split.training
    if len(training) < 2:
        raise ValueError(
            "La auditoría de proximidad requiere al menos dos diseños de entrenamiento"
        )
    reference = training.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float)
    scales = tuple(float(value) for value in np.ptp(reference, axis=0))
    self_distances = design_distances(reference, reference, scales)
    np.fill_diagonal(self_distances, np.inf)
    boundaries = np.quantile(self_distances.min(axis=1), [0.25, 0.5, 0.75]).tolist()
    rows = []
    for name, table in (("validation", split.validation), ("test", split.testing)):
        matrix = design_distances(
            table.loc[:, list(INPUT_COLUMNS)].to_numpy(dtype=float),
            reference,
            scales,
        )
        indices = matrix.argmin(axis=1)
        distances = matrix[np.arange(len(table)), indices]
        rows.append(
            pd.DataFrame(
                {
                    "split": name,
                    "iter_id": table["iter_id"].to_numpy(),
                    "web_config": table["web_config"].to_numpy(),
                    "nearest_training_id": training.iloc[indices]["iter_id"].to_numpy(),
                    "distance": distances,
                    "distance_region": np.searchsorted(
                        boundaries, distances, side="right"
                    ),
                }
            )
        )
    return pd.concat(rows, ignore_index=True), boundaries


def error_by_distance(
    records: tuple[dict[str, JsonValue], ...], neighbors: pd.DataFrame
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for record in records:
        key = str(record["key"])
        if not key.removeprefix("test/").startswith(("predictor/",)):
            continue
        root = Path(str(record["directory"]))
        split = "test" if key.startswith("test/") else "validation"
        _, model, seed = key.removeprefix("test/").split("/")
        predictions = pd.read_csv(root / f"{split}_predictions.csv")
        matched = predictions.merge(
            neighbors.loc[neighbors["split"].eq(split)],
            on=["iter_id", "web_config"],
            validate="one_to_one",
        )
        if len(matched) != len(predictions):
            raise ValueError("Las predicciones y los diseños auditados no coinciden")
        for target in OUTPUT_COLUMNS:
            error = matched[f"pred_{target}"] - matched[target]
            measured = matched.assign(
                absolute_error=error.abs(), squared_error=error**2, bias=error
            )
            grouped = measured.groupby("distance_region").agg(
                count=("iter_id", "size"),
                mae=("absolute_error", "mean"),
                mse=("squared_error", "mean"),
                bias=("bias", "mean"),
                p95_absolute_error=(
                    "absolute_error",
                    lambda values: values.quantile(0.95),
                ),
            )
            grouped["rmse"] = np.sqrt(grouped.pop("mse"))
            rows.append(
                grouped.reset_index().assign(
                    split=split, model=model, seed=int(seed), target=target
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prediction_utility(
    records: tuple[dict[str, JsonValue], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = utility_comparison(records)
    if joined.empty:
        return pd.DataFrame(), pd.DataFrame()
    families = (
        joined.groupby(["split", "model", "budget"])[
            ["normalized_mae", "normalized_auc", "relative_hypervolume"]
        ]
        .mean()
        .reset_index()
    )
    rows = []
    for (split, budget), group in families.groupby(["split", "budget"]):
        for metric in ("normalized_auc", "relative_hypervolume"):
            valid = group[["normalized_mae", metric]].dropna()
            defined = len(valid) >= 3 and valid.nunique().gt(1).all()
            rows.append(
                {
                    "split": split,
                    "budget": budget,
                    "utility_metric": metric,
                    "family_count": len(valid),
                    "spearman_error_utility": float(
                        spearmanr(valid["normalized_mae"], valid[metric]).statistic
                    )
                    if defined
                    else None,
                }
            )
    return families, pd.DataFrame(rows)


def warning_count(value: JsonValue) -> int:
    if isinstance(value, dict):
        return (
            int(str(value["count"]))
            if "category" in value and "count" in value
            else sum(warning_count(item) for item in value.values())
        )
    if isinstance(value, list):
        return sum(warning_count(item) for item in value)
    return 0


def diagnostic_inventory(records: tuple[dict[str, JsonValue], ...]) -> pd.DataFrame:
    rows = []
    for record in records:
        root = Path(str(record["directory"]))
        metadata = (
            read_object(root / "run.json") if (root / "run.json").is_file() else {}
        )
        warnings = metadata.get("warnings")
        nested_path = root / "nested_folds.json"
        crossfit_diagnostics = (
            read_object(nested_path)
            if nested_path.is_file()
            else metadata.get("cv_fold_warnings", [])
        )
        count = (
            sum(
                int(str(item["count"]))
                for item in cast(list[dict[str, JsonValue]], warnings)
            )
            if isinstance(warnings, list)
            else None
        )
        rows.append(
            {
                "task": record["key"],
                "warning_capture": "recorded"
                if isinstance(warnings, list)
                else "not_recorded",
                "warning_count": count,
                "cv_warning_count": warning_count(crossfit_diagnostics),
                "tuning_warning_count": warning_count(
                    metadata.get("tuning_fold_warnings", [])
                ),
                "cv_reused": metadata.get("cv_reused"),
                "rl_algorithm": metadata.get("algorithm"),
                "rl_use_predictions": metadata.get("use_predictions"),
                "rl_optimizer_updates": metadata.get("optimizer_updates"),
                "rl_samples_used": metadata.get("samples_used"),
                "rl_parameter_count": metadata.get("parameter_count"),
                "warnings": str(warnings) if warnings else None,
                "convergence": str(metadata.get("convergence", [])),
                "shap_requested_method": metadata.get("requested_method"),
                "shap_fallback_reason": metadata.get("fallback_reason"),
                "shap_final_additivity_error": str(
                    metadata.get("final_additivity_error_by_target")
                ),
                "oof_protocol": metadata.get("oof_protocol"),
                "shap_method": metadata.get("method"),
                "tree_additivity_error": str(
                    metadata.get("tree_additivity_error_by_target")
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_study(config: AuditConfig) -> Path:
    from truss_research.evaluacion.benchmark import latest_benchmark
    from truss_research.evaluacion.informe_costos import resource_inventory

    if config.output_dir.exists():
        raise ValueError("Elegir un destino nuevo para la auditoría")
    records = signed_records(config.study_dir)
    state = read_object(config.study_dir / "study.json")
    benchmark = (
        latest_benchmark(config.study_dir) if state["test_started"] is True else None
    )
    resources = resource_inventory(config.study_dir)
    if state["test_started"] is True and (
        benchmark is None
        or resources.empty
        or not resources.coverage.eq("complete").all()
    ):
        raise ValueError(
            "La auditoría final requiere recursos completos y make benchmark"
        )
    if state["dataset_signature"] != dataset_identity(config.data_dir):
        raise ValueError("Los datos no corresponden al estudio auditado")
    split = load_design_split(config.data_dir)
    audit_calibration_exclusions(records, split)
    from truss_research.evaluacion.seleccion_adaptativa import latest_adaptive_selection
    from truss_research.evaluacion.sensibilidad import latest_sensitivity
    from truss_research.optimizacion.estudio_estado import load_study_config

    protocol = load_study_config(config.study_dir)
    from truss_research.evaluacion.informe_analisis import audit_analysis

    audit_analysis(config.study_dir)
    if (
        state["test_started"] is True
        and protocol.analysis.generalization_seeds
        and latest_sensitivity(config.study_dir) is None
    ):
        raise ValueError(
            "La auditoría requiere completar make sensitivity con las divisiones agrupadas"
        )
    if (
        state["test_started"] is True
        and protocol.adaptive.online_residual
        and latest_adaptive_selection(config.study_dir) is None
    ):
        raise ValueError(
            "La auditoría requiere completar make adaptive con el protocolo congelado"
        )
    neighbors, boundaries = nearest_designs(split)
    coverage = pd.concat(
        [
            table.groupby("web_config")
            .size()
            .reindex(WEB_CONFIG_VALUES, fill_value=0)
            .rename("count")
            .reset_index()
            .assign(split=name)
            for name, table in split.as_mapping().items()
        ],
        ignore_index=True,
    )
    families, correlations = prediction_utility(records)
    diagnostics = diagnostic_inventory(records)
    errors = error_by_distance(records, neighbors)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    tables = {
        "coverage": coverage,
        "nearest_designs": neighbors,
        "errors_by_distance": errors,
        "prediction_utility": families,
        "utility_correlations": correlations,
        "diagnostics": diagnostics,
        "resources": resources,
    }
    for name, table in tables.items():
        table.to_csv(config.output_dir / f"{name}.csv", index=False)
    from truss_research.evaluacion.cierre_tecnico import export_technical_closure

    export_technical_closure(config.study_dir, config.output_dir)
    from truss_research.evaluacion.robustez import (
        EVIDENCE_NOTES,
        export_convergence,
        export_evidence_context,
        provenance_table,
    )

    export_convergence(state, config.output_dir)
    export_evidence_context(split, config.output_dir)
    from truss_research.evaluacion.informe_rl import export_policy_diagnostics

    export_policy_diagnostics(records, config.output_dir)
    manifest = read_manifest(config.data_dir)
    write_json(
        config.output_dir / "audit.json",
        {
            "study": str(config.study_dir.resolve()),
            "dataset_identity": dataset_identity(config.data_dir),
            "design_overlap": audit_split_overlap(split),
            "design_audit": manifest.get("design_audit"),
            "distance_boundaries_from_training": boundaries,
            "complete_tasks": len(records),
            "classification": "retrospective_descriptive",
            "physical_provenance_signature": confirmations_digest(),
            "physical_provenance_pending": bool(
                provenance_table().status.ne("Confirmado").any()
            ),
            "restricted_data": True,
            "benchmark_dir": str(benchmark) if benchmark is not None else None,
            "resource_coverage_complete": bool(
                not resources.empty and resources.coverage.eq("complete").all()
            ),
        },
    )
    sections = [
        "# Auditoría retrospectiva del estudio",
        "No se entrenaron modelos ni se modificó el estudio. Los archivos derivados contienen información restringida.",
        *EVIDENCE_NOTES.values(),
        "Consultar category_coverage.csv, generation_overlap.csv y physical_provenance.csv: ausencia de duplicados exactos no demuestra independencia entre problemas.",
        "## Cobertura",
        markdown_table(coverage),
        "## Precisión y utilidad de ranking",
        markdown_table(correlations)
        if not correlations.empty
        else "No hay rankings completos para comparar.",
        "Las correlaciones se calculan entre medias por familia. No se exportan valores p ni se consideran las semillas como datasets independientes.",
        "La distancia usa estandarización numérica y codificación categórica ajustadas exclusivamente con entrenamiento; sus regiones se fijan con distancias entre diseños de entrenamiento.",
        "La prueba ya observada no se convierte en una validación externa nueva. Los conteos bajos, diseños próximos y catálogos superpuestos limitan las conclusiones.",
        "Las advertencias finales, de búsqueda y OOF, los métodos SHAP y sus alternativas se registran en diagnostics.csv; la convergencia no se deduce de predicciones finitas.",
    ]
    (config.output_dir / "AUDIT.md").write_text(
        "\n\n".join(sections) + "\n", encoding="utf-8"
    )
    return config.output_dir
