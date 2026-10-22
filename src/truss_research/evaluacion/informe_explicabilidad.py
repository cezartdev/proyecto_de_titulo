"""Tablas SHAP, costos y exportación de etapas independientes ya terminadas."""

import json
import logging
from pathlib import Path
from typing import cast

import pandas as pd

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.busqueda_evolutiva import predictor_records
from truss_research.evaluacion.registro import results_root
from truss_research.optimizacion.estudio_estado import load_study_config
from truss_research.surrogates.experimentos import data_signature, read_run, write_json
from truss_research.surrogates.explicabilidad import ExplanationConfig
from truss_research.surrogates.persistencia import model_signatures

logger = logging.getLogger(__name__)

EXPLANATION_FILES = (
    "shap_global.csv",
    "shap_values.csv",
    "shap_examples.csv",
    "shap_costs.csv",
    "shap_stability.csv",
)


def measured_costs(config: ExplanationConfig) -> dict[str, JsonValue]:
    attempts = [
        read_artifact(path)
        for path in sorted(
            config.output_dir.with_suffix(".attempts").glob("attempt_*.json")
        )
    ]
    rss = [row["resources"].get("peak_rss_mib") for row in attempts]
    vram = [row["resources"].get("gpu_peak_allocated_mib") for row in attempts]
    return {
        "attempts": len(attempts),
        "attempt_wall_seconds": sum(float(row["wall_seconds"]) for row in attempts),
        "attempt_cpu_seconds": sum(
            float(row["resources"]["cpu_seconds"]) for row in attempts
        ),
        "peak_rss_mib": max(
            (float(value) for value in rss if value is not None), default=None
        ),
        "peak_gpu_allocated_mib": max(
            (float(value) for value in vram if value is not None), default=None
        ),
    }


def aggregate_explanations(plan: dict[str, JsonValue], destination: Path) -> None:
    from truss_research.evaluacion.explicabilidad import job_config

    values: list[pd.DataFrame] = []
    costs: list[dict[str, JsonValue]] = []
    for raw in cast(list[dict[str, JsonValue]], plan["jobs"]):
        job = job_config(raw)
        metadata = read_run(job.output_dir)
        values.append(
            pd.read_csv(job.output_dir / "shap_values.csv").assign(
                model=raw["model"], seed=raw["model_seed"]
            )
        )
        costs.append(
            {
                "model": raw["model"],
                "seed": raw["model_seed"],
                **{
                    key: metadata.get(key)
                    for key in (
                        "method",
                        "requested_method",
                        "fallback_reason",
                        "execution_device",
                        "explanation_seconds",
                        "counted_prediction_calls",
                        "counted_prediction_rows",
                        "permutation_completed_seconds",
                        "batch_size",
                        "permutation_rounds",
                    )
                },
                "evaluation_designs": len(metadata["evaluation_ids"]),
                "background_designs": len(metadata["background_ids"]),
                "scope": plan["mode"],
                "component_methods": json.dumps(
                    metadata["component_methods"], ensure_ascii=False
                ),
                "background_semantics": metadata["background_semantics"],
                **measured_costs(job),
                "estimated_full_permutation_seconds": (
                    float(metadata["permutation_completed_seconds"])
                    * int(metadata["full_validation_designs"])
                    / len(metadata["evaluation_ids"])
                    * min(
                        int(plan["parameters"]["background_size"]),
                        int(metadata["full_training_designs"]),
                    )
                    / len(metadata["background_ids"])
                    * int(plan["parameters"]["permutation_rounds"])
                    / job.permutation_rounds
                ),
                "estimate_scope": "rough_permutation_scaling_only_excludes_load_trees_and_verification_not_a_runtime_guarantee",
            }
        )
    table = pd.concat(values, ignore_index=True)
    table.to_csv(destination / "shap_values.csv", index=False)
    global_values = (
        table.assign(abs_shap=lambda frame: frame.shap_value.abs())
        .groupby(["model", "seed", "target", "variable"], as_index=False)
        .abs_shap.mean()
    )
    global_values.to_csv(destination / "shap_global.csv", index=False)
    examples = sorted(table.iter_id.unique())[:3]
    table.loc[table.iter_id.isin(examples)].to_csv(
        destination / "shap_examples.csv", index=False
    )
    pd.DataFrame(costs).to_csv(destination / "shap_costs.csv", index=False)
    export_stability(global_values, destination)


def export_stability(table: pd.DataFrame, destination: Path) -> None:
    """Describe acuerdo de rankings entre semillas, sin tratarlas como réplicas físicas."""
    from itertools import combinations

    from scipy.stats import spearmanr

    rows: list[dict[str, JsonValue]] = []
    for (model, target), group in table.groupby(["model", "target"]):
        matrix = group.pivot(index="variable", columns="seed", values="abs_shap")
        for left, right in combinations(matrix.columns, 2):
            value = (
                float(spearmanr(matrix[left], matrix[right]).statistic)
                if min(matrix[left].nunique(), matrix[right].nunique()) > 1
                else None
            )
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "seed": int(left),
                    "other_seed": int(right),
                    "spearman": value,
                }
            )
    pd.DataFrame(
        rows, columns=["model", "target", "seed", "other_seed", "spearman"]
    ).to_csv(destination / "shap_stability.csv", index=False)


def export_explanations(study: Path, destination: Path) -> None:
    """Adjunta una etapa completa compatible; un piloto nunca entra al informe principal."""
    from shutil import copyfile

    from truss_research.evaluacion.explicabilidad import (
        selected_families,
        verified_files,
    )

    candidates = sorted(results_root(study).glob("explicabilidad_*"), reverse=True)
    complete = [
        path
        for path in candidates
        if (path / "complete.json").is_file()
        and read_artifact(path / "complete.json").get("mode") == "full"
    ]
    if not complete:
        logger.info("SHAP pendiente: ejecutar make explain y regenerar el informe")
        return
    source = complete[0]
    receipt = verified_files(source)
    config = load_study_config(study)
    if (
        Path(receipt["study"]).resolve() != study.resolve()
        or receipt["selected"] != selected_families(study)
        or receipt["dataset_signature"] != data_signature(config.data_dir)
    ):
        raise ValueError(
            "La etapa SHAP no corresponde a la selección o dataset del informe"
        )
    current = {f"{row.model}/{row.seed}": row for row in predictor_records(study)}
    for predictor in receipt["predictors"]:
        key = f"{predictor['model']}/{predictor['seed']}"
        if (
            key not in current
            or Path(str(current[key].record.directory)).resolve()
            != Path(predictor["directory"]).resolve()
            or model_signatures(Path(predictor["directory"])) != predictor["files"]
        ):
            raise ValueError("Cambió un predictor explicado")
    for name in EXPLANATION_FILES:
        copyfile(source / name, destination / name)
    write_json(
        destination / "shap_provenance.json",
        {
            "directory": str(source),
            "complete_sha256": source_digest(source / "complete.json"),
            "scope": "validation_only",
        },
    )
