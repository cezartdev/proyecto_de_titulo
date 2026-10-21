"""Utilidad de cada regresor en catálogos retenidos comunes, sin generar diseños."""

import hashlib
import json
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.particion import IndexArray
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.pareto import ObjectiveSpace, meets_thresholds
from truss_research.optimizacion.estrategias import pareto_ranking
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    artifact_signatures,
    source_signature,
)
from truss_research.surrogates.configuracion import FloatArray, response_scale
from truss_research.surrogates.experimentos import write_json

CONTROL_SEED = -1
RANDOM_CONTROL = "random_control"
ORACLE_CONTROL = "oracle_ranking"
SCENARIO_OFFSETS = {"validation": 3_000_000, "test": 4_000_000}
QUANTILES = (0.025, 0.975)


def benchmark_protocol(config: StudyConfig) -> dict[str, JsonValue]:
    """Decisiones fijadas antes del test; el oráculo solo es un diagnóstico."""
    return {
        "objectives": config.objective_mode,
        "directions": [
            "min_mass",
            "max_mean_utilization_under_limits",
            "min_displacement",
        ]
        if config.objective_mode == "three"
        else ["min_mass", "min_displacement"],
        "selection_rule": "fixed_feasibility_violation_pareto_rank_crowding_seeded_ties",
        "family_selection": "validation_mean_auc_smallest_budget_then_mae_then_name",
        "selection_budget": min(config.budgets),
        "quality_source": "historical_observed_not_model_predictions",
        "catalogue_size_requested": max(config.catalog_size, max(config.budgets)),
        "initial_population": config.evolution.population_size,
        "budgets": list(config.budgets),
        "seeds": list(config.seeds),
        "scenarios": config.proposals.evaluation_episodes,
        "bootstrap_repeats": config.analysis.bootstrap_repeats,
        "bootstrap_seed": config.analysis.bootstrap_seed,
        "interval_scope": "pointwise_exploratory_seed_averaged_paired_scenarios",
        "oracle_scope": "same_ranking_with_hidden_labels_not_global_optimum_or_auc_upper_bound",
        "reference_scope": "training_scales_and_reference_point; full_catalogue_hv_diagnostic_only",
        "scope": "exploratory_same_trajectory_fixed_fits_overlapping_catalogues",
    }


def observed_curve(
    initial: FloatArray, catalogue: tuple[FloatArray, IndexArray], space: ObjectiveSpace
) -> FloatArray:
    """Solo respuestas reveladas contribuyen al hipervolumen observado."""
    responses, order = catalogue
    selected = np.vstack((initial, responses[order]))
    return np.array(
        [
            0.0,
            *[space.hypervolume(selected[:end]) for end in range(1, len(selected) + 1)],
        ]
    )


def objective_space(training: pd.DataFrame, config: StudyConfig) -> ObjectiveSpace:
    space = ObjectiveSpace.fit(training[list(OUTPUT_COLUMNS)].to_numpy(), config.limits)
    return (
        replace(space, active_dimensions=(0, 2))
        if config.objective_mode == "mass_displacement"
        else space
    )


def catalogue_comparison(
    training: pd.DataFrame, predictions: pd.DataFrame, config: StudyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from truss_research.evaluacion.comparacion_predictiva import paired_losses

    paired_losses(predictions, response_scale(training[list(OUTPUT_COLUMNS)]))
    split_names = predictions.split.unique()
    if len(split_names) != 1 or split_names[0] not in SCENARIO_OFFSETS:
        raise ValueError("El benchmark requiere una partición de validación o prueba")
    split = str(split_names[0])
    training_values = training[list(OUTPUT_COLUMNS)].to_numpy()
    space = objective_space(training, config)
    size = config.evolution.population_size
    if len(training) < size:
        raise ValueError("Faltan diseños de entrenamiento para la inicialización común")
    if min(config.budgets) <= size:
        raise ValueError("El presupuesto debe superar la inicialización del benchmark")
    if "iter_id" in training and set(training.iter_id) & set(predictions.iter_id):
        raise ValueError(
            "El catálogo retenido no debe contener diseños de entrenamiento"
        )
    reference = space.hypervolume(training_values)
    table = predictions.drop_duplicates("iter_id").set_index("iter_id").sort_index()
    actual = table[list(OUTPUT_COLUMNS)].to_numpy()
    models = {
        (str(model), int(seed)): group.set_index("iter_id")
        .loc[table.index, [f"pred_{name}" for name in OUTPUT_COLUMNS]]
        .to_numpy()
        for (model, seed), group in predictions.groupby(["model", "seed"])
    }
    summaries: list[dict[str, JsonValue]] = []
    traces: list[dict[str, JsonValue]] = []
    for scenario in range(config.proposals.evaluation_episodes):
        rng = np.random.default_rng(SCENARIO_OFFSETS[split] + scenario)
        initial_indices = rng.choice(len(training), size=size, replace=False)
        initial = training_values[initial_indices]
        indices = rng.choice(
            len(table),
            size=min(len(table), max(config.catalog_size, max(config.budgets))),
            replace=False,
        )
        values = actual[indices]
        attainable = space.hypervolume(np.vstack((initial, values)))
        orders = {
            key: pareto_ranking(estimate[indices], space)
            for key, estimate in models.items()
        }
        orders[(RANDOM_CONTROL, CONTROL_SEED)] = rng.permutation(len(indices))
        orders[(ORACLE_CONTROL, CONTROL_SEED)] = pareto_ranking(values, space)
        for (model, seed), order in orders.items():
            order = order[: max(config.budgets) - size]
            curve = observed_curve(initial, (values, order), space)
            for position, selected in enumerate(order, start=size + 1):
                traces.append(
                    {
                        "split": split,
                        "model": model,
                        "seed": seed,
                        "scenario": scenario,
                        "query": position,
                        "iter_id": str(table.index[indices[selected]]),
                        "hypervolume": float(curve[position]),
                    }
                )
            for budget in config.budgets:
                effective = min(budget, len(curve) - 1)
                hv = float(curve[effective])
                summaries.append(
                    {
                        "split": split,
                        "model": model,
                        "seed": seed,
                        "scenario": scenario,
                        "role": "diagnostic_oracle"
                        if model == ORACLE_CONTROL
                        else "random_control"
                        if model == RANDOM_CONTROL
                        else "regressor",
                        "budget": budget,
                        "effective_budget": effective,
                        "catalogue_size": len(indices),
                        "catalogue_exhausted": effective < budget,
                        "initial_indices": ",".join(map(str, initial_indices)),
                        "hypervolume": hv,
                        "auc": float(np.trapezoid(curve[: effective + 1]) / effective),
                        "normalized_auc": float(
                            np.trapezoid(curve[: effective + 1])
                            / (effective * reference)
                        )
                        if reference > 0
                        else None,
                        "relative_hypervolume": hv / reference
                        if reference > 0
                        else None,
                        "training_reference_hypervolume": reference,
                        "catalogue_attainable_hypervolume": attainable,
                        "observed_failures": int(
                            (
                                ~meets_thresholds(
                                    values[order[: effective - size]], config.limits
                                )
                            ).sum()
                        ),
                        "evidence": "historical_observed",
                        "objective_mode": config.objective_mode,
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(traces)


def utility_intervals(table: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows: list[dict[str, JsonValue]] = []
    for budget, group in table.groupby("budget"):
        averaged = group.groupby(["scenario", "model"])["auc"].mean().unstack("model")
        if averaged.isna().any().any():
            raise ValueError(
                "Faltan escenarios comunes en la comparación multiobjetivo"
            )
        rng = np.random.default_rng(config.analysis.bootstrap_seed)
        sampled = rng.integers(
            len(averaged), size=(config.analysis.bootstrap_repeats, len(averaged))
        )
        for left, right in combinations(averaged.columns, 2):
            differences = (averaged[left] - averaged[right]).to_numpy()
            low, high = np.quantile(differences[sampled].mean(axis=1), QUANTILES)
            rows.append(
                {
                    "budget": int(budget),
                    "model": left,
                    "reference_model": right,
                    "mean_auc_difference": float(differences.mean()),
                    "ci_lower": float(low) if len(averaged) > 1 else None,
                    "ci_upper": float(high) if len(averaged) > 1 else None,
                    "scenario_pairs": len(averaged),
                    "scope": "pointwise_exploratory_fixed_fits_seed_averaged_overlapping_catalogues",
                }
            )
    return pd.DataFrame(rows)


def write_benchmark(
    context: tuple[pd.DataFrame, pd.DataFrame], config: StudyConfig, destination: Path
) -> pd.DataFrame:
    """Guarda evidencia observada y contratos; no ajusta ningún modelo."""
    training, predictions = context
    identity = hashlib.sha256(
        (
            training.to_csv(index=False)
            + predictions.to_csv(index=False)
            + json.dumps(asdict(config), sort_keys=True, default=str)
            + source_signature()
        ).encode()
    ).hexdigest()
    if (destination / "complete.json").is_file():
        verify_benchmark(destination)
        if read_artifact(destination / "complete.json")["identity"] != identity:
            raise ValueError(
                "El benchmark pertenece a otros datos o protocolo; elegir un estudio nuevo"
            )
        return pd.read_csv(destination / "summary.csv")
    destination.mkdir(parents=True, exist_ok=True)
    summary, traces = catalogue_comparison(training, predictions, config)
    summary.to_csv(destination / "summary.csv", index=False)
    traces.to_csv(destination / "decisions.csv", index=False)
    summary.groupby(["split", "model", "role", "budget"], as_index=False).agg(
        mean_auc=("auc", "mean"),
        mean_hypervolume=("hypervolume", "mean"),
        mean_normalized_auc=("normalized_auc", "mean"),
        effective_budget=("effective_budget", "first"),
        catalogue_size=("catalogue_size", "first"),
    ).to_csv(destination / "family_summary.csv", index=False)
    utility_intervals(summary, config).to_csv(
        destination / "paired_intervals.csv", index=False
    )
    write_json(
        destination / "protocol.json",
        {
            **benchmark_protocol(config),
            "space": asdict(objective_space(training, config)),
        },
    )
    write_json(
        destination / "complete.json",
        {"files": artifact_signatures(destination), "identity": identity},
    )
    return summary


def verify_benchmark(destination: Path) -> None:
    files = read_artifact(destination / "complete.json")["files"]
    if any(
        not (destination / name).is_file()
        or source_digest(destination / name) != digest
        for name, digest in files.items()
    ):
        raise ValueError("Cambió un benchmark multiobjetivo completo")


def export_benchmark_reports(root: Path, destination: Path) -> None:
    """Consolida utilidad y precisión sin volver a ejecutar los episodios."""
    for filename, output in (
        ("summary.csv", "predictor_utility_runs.csv"),
        ("family_summary.csv", "predictor_utility_summary.csv"),
        ("paired_intervals.csv", "predictor_utility_paired_intervals.csv"),
    ):
        pd.concat(
            [
                pd.read_csv(root / name / filename).assign(split=name)
                for name in ("validation", "test")
            ],
            ignore_index=True,
        ).to_csv(destination / output, index=False)
    utility = pd.read_csv(destination / "predictor_utility_summary.csv")
    accuracy = pd.read_csv(destination / "predictive_model_summary.csv")[
        ["split", "model", "mean"]
    ].rename(columns={"mean": "normalized_mae"})
    utility.merge(
        accuracy, on=["split", "model"], how="left", validate="many_to_one"
    ).to_csv(destination / "prediction_utility.csv", index=False)


def evaluate_predictor_utility(config: StudyConfig, split_name: str) -> pd.DataFrame:
    from truss_research.datos.procesados import load_design_split
    from truss_research.evaluacion.busqueda_evolutiva import predictor_records
    from truss_research.optimizacion.estudio_estado import verify_task

    frames: list[pd.DataFrame] = []
    for reference in predictor_records(config.output_dir):
        root = (
            verify_task(reference.record)
            if split_name == "validation"
            else config.output_dir
            / "evolution/test_predictors"
            / f"{reference.model}_{reference.seed}"
        )
        frames.append(
            pd.read_csv(root / f"{split_name}_predictions.csv").assign(
                split=split_name, model=reference.model, seed=reference.seed
            )
        )
    return write_benchmark(
        (
            load_design_split(config.data_dir).training,
            pd.concat(frames, ignore_index=True),
        ),
        config,
        config.output_dir / "evolution/predictor_utility" / split_name,
    )
