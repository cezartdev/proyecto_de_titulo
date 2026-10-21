"""Sensibilidades predefinidas sobre modelos congelados y trayectorias verificadas."""

from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue, load_design_split
from truss_research.evaluacion.informe_estudio import markdown_table, signed_records
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.evaluacion.seleccion import EpisodeReport, selection_metrics
from truss_research.optimizacion.catalogos import SelectionBank, load_bank
from truss_research.optimizacion.comparacion import (
    PolicyReference,
    evaluate_selection_suite,
)
from truss_research.optimizacion.contratos import SelectionConfig
from truss_research.optimizacion.estudio_estado import (
    artifact_signatures,
    load_study_config,
    read_study,
    source_signature,
)
from truss_research.surrogates.configuracion import FloatArray
from truss_research.surrogates.experimentos import data_signature, write_json

SENSITIVITY_SCHEMA_VERSION = 4


def combine_sensitivity_tables(frames: list[pd.DataFrame]) -> pd.DataFrame:
    columns = list(dict.fromkeys(column for frame in frames for column in frame))
    return pd.concat(
        [frame.dropna(axis=1, how="all") for frame in frames], ignore_index=True
    ).reindex(columns=columns)


def reference_space(
    space: ObjectiveSpace, responses: FloatArray, margin: float
) -> ObjectiveSpace:
    mass = responses[:, 0]
    span = float(np.ptp(mass)) or float(mass.max())
    reference = (
        float(mass.max()) + margin * span,
        0.0,
        space.reference_physical[2],
    )
    return replace(space, reference_physical=reference)


def recompute_reference(
    bank: SelectionBank, traces: pd.DataFrame, space: ObjectiveSpace
) -> pd.DataFrame:
    specs = {spec.episode_id: spec for spec in bank.episodes}
    rows: list[dict[str, JsonValue]] = []
    for (episode_id, comparison_id), group in traces.groupby(
        ["episode_id", "comparison_id"]
    ):
        spec = specs[str(episode_id)]
        responses = bank.table.loc[
            list(spec.candidate_ids), list(OUTPUT_COLUMNS)
        ].to_numpy(dtype=float)
        ordered = group.sort_values("query")
        selected = tuple(
            spec.candidate_ids.index(str(value)) for value in ordered["iter_id"]
        )
        repeated = len(set(selected)) != len(selected)
        repeat_variant = str(ordered.iloc[0]["strategy"]) in (
            "rl_repeat_control",
            "rl_repeat_penalty",
        )
        if len(selected) != spec.budget or (repeated and not repeat_variant):
            raise ValueError("Trayectoria incompleta o con repeticiones no autorizadas")
        volumes = tuple(
            space.hypervolume(responses[list(selected[:number])])
            for number in range(1, len(selected) + 1)
        )
        report = EpisodeReport(responses, selected, volumes, space)
        first = ordered.iloc[0]
        rows.append(
            {
                "episode_id": str(episode_id),
                "comparison_id": str(comparison_id),
                "strategy": str(first["strategy"]),
                "catalog_seed": spec.seed,
                "policy_training_seed": None
                if pd.isna(first["policy_training_seed"])
                else int(first["policy_training_seed"]),
                "budget": spec.budget,
                **selection_metrics(report),
                "queries": len(set(selected)),
                "unique_queries": len(set(selected)),
                "attempts": len(selected),
                "repeat_attempts": len(selected) - len(set(selected)),
                "repeat_rate": 1 - len(set(selected)) / len(selected),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_figures(table: pd.DataFrame, variable: str, destination: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    for (split, budget), group in table.groupby(["split", "budget"]):
        averaged = group.groupby(["strategy", variable], as_index=False)[
            "normalized_auc"
        ].mean()
        figure = Figure(figsize=(8, 5))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for strategy, values in averaged.groupby("strategy"):
            ordered = values.sort_values(variable)
            axis.plot(
                ordered[variable],
                ordered["normalized_auc"],
                marker="o",
                label=str(strategy),
            )
        axis.set(
            xlabel="Candidatos"
            if variable == "catalog_size"
            else "Margen de referencia de masa",
            ylabel="AUC normalizada media",
            title=f"Sensibilidad — {split}, presupuesto {budget}",
        )
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(destination / f"{variable}_{split}_{budget}.png", dpi=150)
        figure.clear()


def sensitivity_inputs(study: Path) -> tuple[dict[str, JsonValue], dict[str, Path]]:
    state = read_study(study)
    if state["test_started"] is not True or not (study / "freeze.json").is_file():
        raise ValueError(
            "La sensibilidad requiere cierre completo; ejecutar make final"
        )
    records = signed_records(study)
    tasks = {str(record["key"]): Path(str(record["directory"])) for record in records}
    config = load_study_config(study)
    expected = {
        *(
            f"test/predictor/{model}/{seed}"
            for model in config.models
            for seed in config.seeds
        ),
        *(
            f"test/ranking/{model}/{seed}"
            for model in config.models
            for seed in config.seeds
        ),
        "test/decision",
        "decision/validation",
    }
    if not expected <= tasks.keys() or any(
        record["status"] != "complete" for record in state["tasks"]
    ):
        raise ValueError("Completar todas las tareas antes de sensibilidad")
    if data_signature(config.data_dir) != state["dataset_signature"]:
        raise ValueError("Los datos cambiaron después del cierre")
    if config.analysis.catalog_sizes[0] != config.catalog_size:
        raise ValueError(
            "La sensibilidad debe comenzar con el tamaño de catálogo principal"
        )
    split = load_design_split(config.data_dir)
    if max(config.analysis.catalog_sizes) > min(
        len(split.validation), len(split.testing)
    ):
        raise ValueError(
            "Los catálogos de sensibilidad superan los diseños disponibles"
        )
    return state, tasks


def run_sensitivity(study: Path, destination: Path) -> Path:
    from truss_research.optimizacion.estudio_estado import study_lock

    if not (study / "study.json").is_file():
        raise ValueError("Falta completar make study y make final")
    if destination.resolve().is_relative_to(study.resolve()):
        raise ValueError("Guardar sensibilidad fuera del estudio firmado, en reports")
    with study_lock(study):
        return execute_sensitivity(study, destination)


def execute_sensitivity(study: Path, destination: Path) -> Path:
    from truss_research.evaluacion.trabajos_analisis import analysis_session

    if (destination / "provenance.json").exists():
        raise ValueError("La sensibilidad ya está completa; elegir un destino nuevo")
    state, tasks = sensitivity_inputs(study)
    signatures = {
        name: source_digest(study / name) for name in ("study.json", "freeze.json")
    }
    config = load_study_config(study)
    frozen = read_artifact(study / "freeze.json")
    winner = Path(str(frozen["winner"]))
    policies = tuple(
        PolicyReference(tasks[f"policy/{variant}/{seed}"], seed, variant)
        for variant in config.rl.variants
        for seed in config.seeds
    )
    training = (
        load_design_split(config.data_dir)
        .training.loc[:, list(OUTPUT_COLUMNS)]
        .to_numpy(dtype=float)
    )
    analysis_session(config, destination)
    from truss_research.evaluacion.generalizacion import run_generalization

    run_generalization(config, destination / "generalization")
    catalogs: list[pd.DataFrame] = []
    references: list[pd.DataFrame] = []
    formulations: list[pd.DataFrame] = []
    serviceability: list[pd.DataFrame] = []
    for split_name, key in (
        ("validation", "decision/validation"),
        ("test", "test/decision"),
    ):
        root = tasks[key]
        primary = pd.read_csv(root / "summary.csv")
        traces = pd.read_csv(root / "traces.csv")
        catalogs.append(
            primary.assign(split=split_name, catalog_size=config.catalog_size)
        )
        selection = SelectionConfig(
            config.data_dir,
            winner,
            destination / "unused",
            split=split_name,
            catalog_size=config.catalog_size,
            budgets=config.budgets,
            seeds=config.seeds,
            episodes_per_seed=config.episodes_per_seed,
            evolution=config.evolution,
        )
        bank = load_bank(selection)
        serviceability.append(
            recompute_reference(bank, traces, bank.space).assign(
                split=split_name,
                serviceability="L700",
                trajectory_scope="fixed_L700_policy_trajectories",
            )
        )
        original_space = ObjectiveSpace.fit(
            training, replace(bank.space.limits, serviceability_divisor=None)
        )
        serviceability.append(
            recompute_reference(bank, traces, original_space).assign(
                split=split_name,
                serviceability="original_without_displacement_limit",
                trajectory_scope="fixed_L700_policy_trajectories",
            )
        )
        formulations.append(
            primary.assign(
                split=split_name,
                formulation="mass_utilization_displacement",
                trajectory_scope="fixed_triobjective_trajectories",
            )
        )
        formulations.append(
            recompute_reference(
                bank, traces, replace(bank.space, active_dimensions=(0, 2))
            ).assign(
                split=split_name,
                formulation="mass_displacement_with_util_constraint",
                trajectory_scope="fixed_triobjective_trajectories",
            )
        )
        for margin in config.analysis.reference_margins:
            space = reference_space(bank.space, training, margin)
            references.append(
                recompute_reference(bank, traces, space).assign(
                    split=split_name, reference_margin=margin
                )
            )
        for size in config.analysis.catalog_sizes[1:]:
            output = destination / f"catalog_{size}" / split_name
            output.parent.mkdir(exist_ok=True)
            expanded = replace(
                selection,
                output_dir=output,
                catalog_size=size,
                base_catalog_size=config.catalog_size,
            )
            catalogs.append(
                cached_catalog(expanded, policies).assign(
                    split=split_name, catalog_size=size
                )
            )
    catalog_columns = list(
        dict.fromkeys(column for frame in catalogs for column in frame)
    )
    catalog_table = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in catalogs], ignore_index=True
    ).reindex(columns=catalog_columns)
    reference_table = combine_sensitivity_tables(references)
    catalog_table.to_csv(destination / "catalog_sensitivity.csv", index=False)
    reference_table.to_csv(destination / "reference_sensitivity.csv", index=False)
    combine_sensitivity_tables(formulations).to_csv(
        destination / "objective_sensitivity.csv", index=False
    )
    combine_sensitivity_tables(serviceability).to_csv(
        destination / "serviceability_sensitivity.csv", index=False
    )
    sensitivity_figures(catalog_table, "catalog_size", destination)
    sensitivity_figures(reference_table, "reference_margin", destination)
    summary = catalog_table.groupby(
        ["split", "catalog_size", "strategy", "budget"], as_index=False
    )[["normalized_auc", "relative_hypervolume"]].mean()
    summary.to_csv(destination / "catalog_summary.csv", index=False)
    (destination / "SENSITIVITY.md").write_text(
        "# Sensibilidades predefinidas\n\nLa transferencia cronológica reentrena predictores y controles y evalúa selección retenida. "
        "Las sensibilidades de catálogo, referencia y formulación reutilizan los modelos principales congelados. "
        "Las políticas se entrenaron con el catálogo principal. Los catálogos ampliados conservan inicio y candidatos base. "
        "La sensibilidad de referencia recalcula las mismas trayectorias. El escenario original sin límite Z reevalúa trayectorias de políticas entrenadas con L/700; no compara políticas reentrenadas para ese escenario. Sus referencias HV difieren, por lo que los valores absolutos entre escenarios no son directamente comparables. Resultados exploratorios de una sola trayectoria de optimización.\n\n"
        + markdown_table(summary)
        + "\n",
        encoding="utf-8",
    )
    signed_records(study)
    analysis_session(config, destination)
    if (
        any(source_digest(study / name) != value for name, value in signatures.items())
        or data_signature(config.data_dir) != state["dataset_signature"]
    ):
        raise ValueError("El estudio o los datos cambiaron durante sensibilidad")
    write_json(
        destination / "provenance.json",
        {
            "schema_version": SENSITIVITY_SCHEMA_VERSION,
            "study": str(study.resolve()),
            "state_signature": signatures["study.json"],
            "freeze_signature": signatures["freeze.json"],
            "dataset_signature": state["dataset_signature"],
            "analysis_source_signature": source_signature(),
            "analysis": asdict(config.analysis),
            "trained_catalog_size": config.catalog_size,
            "retraining": bool(config.analysis.generalization_seeds),
            "report_signatures": {
                path.relative_to(destination).as_posix(): source_digest(path)
                for path in destination.rglob("*")
                if path.is_file()
            },
        },
    )
    return destination


def cached_catalog(
    config: SelectionConfig, policies: tuple[PolicyReference, ...]
) -> pd.DataFrame:
    from truss_research.evaluacion.trabajos_analisis import preserve_partial

    marker = config.output_dir / "completion.json"
    if marker.is_file():
        saved = read_artifact(marker)
        measured = {
            name: digest
            for name, digest in artifact_signatures(config.output_dir).items()
            if name != marker.name
        }
        if saved.get("signatures") != measured:
            raise ValueError("El catálogo completo cambió durante la reanudación")
        return pd.read_csv(config.output_dir / "summary.csv", low_memory=False)
    preserve_partial(config.output_dir)
    result = evaluate_selection_suite(config, policies)
    write_json(marker, {"signatures": artifact_signatures(config.output_dir)})
    return result


def latest_sensitivity(study: Path) -> Path | None:
    from truss_research.evaluacion.registro import results_root

    root = results_root(study)
    for candidate in sorted(root.glob("sensibilidad_*"), reverse=True):
        path = candidate / "provenance.json"
        if not path.is_file():
            continue
        provenance = read_artifact(path)
        if provenance.get("schema_version") != SENSITIVITY_SCHEMA_VERSION:
            continue
        signatures = cast(dict[str, str], provenance["report_signatures"])
        if (
            provenance["state_signature"] != source_digest(study / "study.json")
            or provenance["freeze_signature"] != source_digest(study / "freeze.json")
            or Path(str(provenance["study"])).resolve() != study.resolve()
        ):
            continue
        measured = {
            path.relative_to(candidate).as_posix(): source_digest(path)
            for path in candidate.rglob("*")
            if path.is_file() and path != candidate / "provenance.json"
        }
        if not measured or signatures != measured:
            raise ValueError("Cambió una tabla, traza o figura de sensibilidad")
        return candidate
    return None
