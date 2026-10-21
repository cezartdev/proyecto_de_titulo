"""Preflight de análisis adicionales sin ajustar modelos ni escribir resultados."""

from dataclasses import replace

import pandas as pd

from truss_research.datos.particion_equilibrada import chronological_generation_split
from truss_research.datos.procesados import JsonValue, load_processed_table
from truss_research.datos.schemas import (
    DESIGN_MEMBERSHIP_FILENAME,
    FULL_FILENAME,
    SPLIT_FILENAMES,
)
from truss_research.optimizacion.estudio_estado import StudyConfig


def analysis_plan(config: StudyConfig) -> dict[str, JsonValue]:
    groups = config.analysis.generalization_seeds
    control_models = [
        name
        for name in config.analysis.generalization.control_models
        if name in config.models
    ]
    if groups and not control_models:
        raise ValueError(
            "La evaluación cronológica necesita al menos una familia de control incluida en el estudio"
        )
    result: dict[str, JsonValue] = {
        "chronological_predictor_fits": len(groups) * len(config.models),
        "matched_control_fits": len(groups) * len(control_models),
        "matched_control_models": control_models,
        "learning_curves_enabled": config.analysis.learning_curves.enabled,
        "preflight": "requires_make_prepare",
    }
    if config.tabpfn_enabled:
        from truss_research.evaluacion.piloto_tabpfn import require_pilot
        from truss_research.surrogates.tabpfn import CHECKPOINT_SHA256, PACKAGE_VERSION

        result["tabpfn"] = {
            "model": "tabpfn",
            "seed": config.representative_seed,
            "winner_eligible": True,
            "package_version": PACKAGE_VERSION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "main_fits_including_oof": len(config.seeds) + config.outer_folds,
            "group_and_control_fits": len(groups) * (1 + ("tabpfn" in control_models)),
            "targetwise_candidate": "targetwise" in config.models,
            "targetwise_context_fits_upper_bound": (
                1 + config.outer_folds + len(groups)
            )
            * (config.inner_folds or config.final_tuning_folds)
            if "targetwise" in config.models
            else 0,
            "targetwise_scalar_fits_upper_bound": 3
            * (config.outer_folds + len(config.seeds) + 2 * len(groups))
            if "targetwise" in config.models
            else 0,
            "curve_fits": len(config.analysis.learning_curves.train_sizes)
            * len(config.analysis.learning_curves.subset_seeds)
            if config.analysis.learning_curves.enabled
            else 0,
            "workers": 1,
        }
        try:
            pilot = require_pilot(config)
            result["tabpfn"]["pilot"] = pilot
            fit_count = (
                len(config.seeds)
                + config.outer_folds
                + int(result["tabpfn"]["group_and_control_fits"])
                + int(result["tabpfn"]["curve_fits"])
            )
            result["tabpfn"]["fit_work_seconds_proxy"] = (
                float(pilot["rows"][-1]["fit_seconds"]) * fit_count
            )
            result["tabpfn"]["estimate_scope"] = (
                "fit_only; all fits approximated by largest pilot size; excludes conditional targetwise fits, evaluation, SHAP and benchmark"
            )
        except (ValueError, OSError) as error:
            result["tabpfn"]["pilot"] = {
                "status": "required",
                "reason": str(error),
            }
    full_path = config.data_dir / FULL_FILENAME
    if not full_path.is_file():
        return result
    full = load_processed_table(full_path)
    membership = pd.read_csv(config.data_dir / DESIGN_MEMBERSHIP_FILENAME)
    settings = replace(
        config.analysis.generalization,
        min_evaluation=max(
            config.catalog_size, config.analysis.generalization.min_evaluation
        ),
    )
    partitions = []
    for seed in groups:
        partition = chronological_generation_split(full, membership, (seed, settings))
        partitions.append(
            {
                "seed": seed,
                "sizes": {name: len(table) for name, table in partition.tables.items()},
                "purged": len(partition.excluded),
                "partition_policy": "fixed_chronological_contiguous_purged",
            }
        )
    result["chronological_partitions"] = partitions
    result["preflight"] = "complete"
    if config.analysis.learning_curves.enabled:
        settings_curve = config.analysis.learning_curves
        total = len(load_processed_table(config.data_dir / SPLIT_FILENAMES["train"]))
        if max(settings_curve.train_sizes) > total:
            raise ValueError(
                "Las curvas requieren más diseños que el entrenamiento disponible"
            )
        sizes = sorted(set((*settings_curve.train_sizes, total)))
        result["learning_curve_sizes"] = sizes
        result["learning_curve_distinct_fits"] = len(settings_curve.models) * (
            sum(size < total for size in sizes) * len(settings_curve.subset_seeds) + 1
        )
    return result
