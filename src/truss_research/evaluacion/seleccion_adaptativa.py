"""Amplía un estudio cerrado con selección incierta, reutilizando sus ajustes."""

import logging
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.costos import machine_context
from truss_research.evaluacion.incertidumbre_seleccion import (
    paired_selection_differences,
    selection_intervals,
)
from truss_research.evaluacion.informe_estudio import markdown_table, signed_records
from truss_research.evaluacion.recursos import ResourceMonitor, timed
from truss_research.evaluacion.sensibilidad import sensitivity_inputs
from truss_research.evaluacion.telemetria_cpu import CampaignMonitor
from truss_research.optimizacion.adaptativa_paralela import (
    AdaptiveRun,
    PreparedAdaptiveBank,
    adaptive_commands,
    aggregate_adaptive_blocks,
    execute_adaptive_blocks,
    export_worker_tables,
    prepare_adaptive_bank,
)
from truss_research.optimizacion.catalogos import (
    SelectionBank,
    bank_manifest,
    comparison_signature,
    load_bank,
)
from truss_research.optimizacion.contratos import EvaluationSplit, SelectionConfig
from truss_research.optimizacion.ejecucion_cpu import ExecutionParameters, cpu_pool
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    artifact_signatures,
    environment_versions,
    load_study_config,
    source_signature,
    study_lock,
)
from truss_research.optimizacion.seleccion_incierta import (
    DEFAULT_EXPLORATION_BETA,
    ENSEMBLE_STRATEGIES,
    MIN_ENSEMBLE_MEMBERS,
    ONLINE_STRATEGIES,
    ensemble_disagreement,
)
from truss_research.surrogates.configuracion import FloatArray
from truss_research.surrogates.experimentos import (
    data_signature,
    predict,
    read_run,
    write_json,
)
from truss_research.surrogates.persistencia import load_model, model_signatures

ADAPTIVE_SCHEMA_VERSION = 1
PREDICTION_RTOL = 1e-10
PREDICTION_ATOL = 1e-12
EVIDENCE_NOTE = (
    "Extensión exploratoria con parámetros congelados, sin selección con prueba. "
    "Los miembros proceden de pliegues externos; su desacuerdo no equivale a intervalos "
    "calibrados ni a probabilidades conjuntas. Las variantes online corrigen residuos "
    "con respuestas reveladas del episodio, sin cambiar predictores persistidos. "
    "La prueba ya fue observada; la cobertura tras selección adaptativa se evalúa descriptivamente."
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveParameters:
    beta: float = DEFAULT_EXPLORATION_BETA
    splits: tuple[EvaluationSplit, ...] = ("validation", "test")
    cpu_workers: int | None = None
    online_residual: bool = False

    def __post_init__(self) -> None:
        if type(self.online_residual) is not bool:
            raise ValueError("La corrección residual debe ser booleana")
        if self.cpu_workers is not None:
            ExecutionParameters(cpu_workers=self.cpu_workers)
        if (
            isinstance(self.beta, bool)
            or not isinstance(self.beta, (int, float))
            or not np.isfinite(self.beta)
            or self.beta < 0
        ):
            raise ValueError("Beta debe ser finito y no negativo")
        if (
            not self.splits
            or not isinstance(self.splits, tuple)
            or any(split not in ("validation", "test") for split in self.splits)
            or len(set(self.splits)) != len(self.splits)
        ):
            raise ValueError(
                "Las particiones deben ser validation/test, sin repeticiones"
            )


def load_adaptive_parameters(path: Path) -> AdaptiveParameters:
    raw = read_artifact(path)
    if (
        set(raw) - {"beta", "splits", "online_residual"}
        or not {"beta", "splits"} <= set(raw)
        or not isinstance(raw["splits"], list)
    ):
        raise ValueError("El JSON adaptativo requiere beta y una lista splits")
    return AdaptiveParameters(
        beta=cast(float, raw["beta"]),
        online_residual=cast(bool, raw.get("online_residual", False)),
        splits=tuple(cast(list[EvaluationSplit], raw["splits"])),
    )


@dataclass(frozen=True)
class AdaptiveInputs:
    config: StudyConfig
    tasks: dict[str, Path]
    winner: Path
    family: str
    members: dict[int, Path]


def adaptive_inputs(study: Path) -> AdaptiveInputs:
    _, tasks = sensitivity_inputs(study)
    config = load_study_config(study)
    winner = Path(str(read_artifact(study / "freeze.json")["winner"]))
    settings = cast(dict[str, JsonValue], read_run(winner)["config"])
    family = str(settings["model"])
    signatures = cast(
        dict[str, dict[str, str]], read_run(winner).get("fold_model_signatures", {})
    )
    if len(signatures) < MIN_ENSEMBLE_MEMBERS:
        raise ValueError(
            "Faltan modelos externos entrenados; no usar semillas como incertidumbre"
        )
    members = {i: winner / name for i, name in enumerate(sorted(signatures))}
    if any(
        not path.is_dir() or model_signatures(path) != signatures[path.name]
        for path in members.values()
    ):
        raise ValueError("Cambió un miembro externo del ensemble")
    return AdaptiveInputs(config, tasks, winner, family, members)


def verify_saved_predictions(predicted: pd.DataFrame, path: Path) -> None:
    columns = [f"pred_{name}" for name in OUTPUT_COLUMNS]
    saved = pd.read_csv(path, usecols=["iter_id", *columns]).set_index(
        "iter_id", verify_integrity=True
    )
    if set(saved.index) != set(predicted.index):
        raise ValueError(
            "Las predicciones guardadas no corresponden a los candidatos actuales"
        )
    if not np.allclose(
        predicted.to_numpy(dtype=float),
        saved.loc[predicted.index, columns].to_numpy(dtype=float),
        rtol=PREDICTION_RTOL,
        atol=PREDICTION_ATOL,
    ):
        raise ValueError(
            "Las predicciones actuales difieren de las guardadas; no mezclar protocolos"
        )


def member_predictions(
    bank: SelectionBank, selection: SelectionConfig, inputs: AdaptiveInputs
) -> tuple[FloatArray, list[dict[str, JsonValue]]]:
    features = bank.table.loc[:, list(INPUT_COLUMNS)]
    values: list[FloatArray] = []
    costs: list[dict[str, JsonValue]] = []
    root = (
        inputs.winner
        if selection.split == "validation"
        else inputs.tasks[
            f"test/predictor/{inputs.family}/{inputs.config.representative_seed}"
        ]
    )
    verify_saved_predictions(
        bank.predictions, root / f"{selection.split}_predictions.csv"
    )
    for member_id, path in inputs.members.items():
        model, loading = timed(lambda: load_model(path))
        predicted, inference = timed(lambda: predict(model, features))
        costs.extend(
            {
                "split": selection.split,
                "member_id": member_id,
                "operation": operation,
                **timing,
            }
            for operation, timing in (
                ("member_load", loading),
                ("member_prediction", inference),
            )
        )
        values.append(predicted.to_numpy(dtype=float))
    return np.stack(values), costs


def export_public_predictions(
    bank: SelectionBank,
    samples: FloatArray,
    context: tuple[SelectionConfig, AdaptiveInputs],
) -> pd.DataFrame:
    selection, inputs = context
    public = bank.table.loc[:, ["iter_id", *INPUT_COLUMNS]]
    members = pd.concat(
        [
            public.join(
                pd.DataFrame(
                    values, index=public.index, columns=OUTPUT_COLUMNS
                ).add_prefix("pred_")
            ).assign(member_id=seed, member_type="outer_fold", split=selection.split)
            for seed, values in zip(inputs.members, samples, strict=True)
        ],
        ignore_index=True,
    )
    members.to_csv(
        selection.output_dir / f"public_members_{selection.split}.csv", index=False
    )
    mean, std = samples.mean(axis=0), ensemble_disagreement(samples)
    estimates = public.join(
        pd.DataFrame(mean, index=public.index, columns=OUTPUT_COLUMNS).add_prefix(
            "mean_"
        )
    ).join(
        pd.DataFrame(std, index=public.index, columns=OUTPUT_COLUMNS).add_prefix("std_")
    )
    estimates.to_csv(
        selection.output_dir / f"public_estimates_{selection.split}.csv", index=False
    )
    errors = np.abs(bank.table.loc[:, list(OUTPUT_COLUMNS)].to_numpy() - mean)
    rows: list[dict[str, JsonValue]] = []
    for column, name in enumerate(OUTPUT_COLUMNS):
        defined = np.ptp(std[:, column]) > 0 and np.ptp(errors[:, column]) > 0
        rows.append(
            {
                "split": selection.split,
                "target": name,
                "designs": len(public),
                "members": len(samples),
                "distinct_member_predictions": len(
                    np.unique(samples[:, :, column], axis=0)
                ),
                "zero_std_count": int(np.sum(std[:, column] == 0)),
                "mean_std": float(std[:, column].mean()),
                "std_absolute_error_spearman": float(
                    spearmanr(std[:, column], errors[:, column]).statistic
                )
                if defined
                else None,
                "undefined_reason": None
                if defined
                else "constant_disagreement_or_error",
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class AdaptiveSplitResult:
    summary: pd.DataFrame
    traces: pd.DataFrame
    diagnostics: pd.DataFrame
    costs: list[dict[str, JsonValue]]


def evaluate_adaptive_split(
    selection: SelectionConfig, inputs: AdaptiveInputs, run: AdaptiveRun
) -> AdaptiveSplitResult:
    bank, bank_cost = timed(lambda: load_bank(selection))
    key = "test/decision" if selection.split == "test" else "decision/validation"
    original = read_run(inputs.tasks[key])
    if comparison_signature(bank, selection) != original["comparison_signature"]:
        raise ValueError("El banco adaptativo no coincide con los episodios originales")
    write_json(
        selection.output_dir / f"episodes_{selection.split}.json",
        bank_manifest(bank, selection),
    )
    samples, costs = member_predictions(bank, selection, inputs)
    costs.insert(
        0,
        {
            "split": selection.split,
            "seed": inputs.config.representative_seed,
            "operation": "bank_preparation_including_representative_load_and_prediction",
            "representative_prediction_seconds_included": bank.prediction_seconds,
            **bank_cost,
        },
    )
    diagnostics = export_public_predictions(bank, samples, (selection, inputs))
    prepared = PreparedAdaptiveBank(
        bank,
        samples,
        run.beta,
        selection.split,
        comparison_signature(bank, selection),
        run.online_residual,
    )
    root = run.output_dir / "preparation" / selection.split
    _, serialization = timed(lambda: prepare_adaptive_bank(prepared, root))
    costs.append(
        {
            "split": selection.split,
            "seed": None,
            "operation": "private_bank_serialization",
            **serialization,
        }
    )
    commands = adaptive_commands(prepared, root, run)
    logger.info(
        "Selección adaptativa %s: %s episodios, %s bloques, hasta %s procesos",
        selection.split,
        len(bank.episodes),
        len(commands),
        run.execution.effective_workers,
    )
    execute_adaptive_blocks(commands, run)
    rows, traces = aggregate_adaptive_blocks(prepared, commands)
    return AdaptiveSplitResult(rows, traces, diagnostics, costs)


def export_adaptive_comparison(
    inputs: AdaptiveInputs, result: pd.DataFrame, destination: Path
) -> None:
    references = pd.concat(
        [
            pd.read_csv(
                inputs.tasks[
                    "test/decision" if split == "test" else "decision/validation"
                ]
                / "summary.csv"
            ).assign(split=split)
            for split in result["split"].unique()
        ],
        ignore_index=True,
    )
    combined = pd.concat(
        [references.loc[references.strategy.isin(("random", "greedy"))], result],
        ignore_index=True,
    )
    paired_selection_differences(combined).loc[
        lambda frame: frame.strategy.isin((*ENSEMBLE_STRATEGIES, *ONLINE_STRATEGIES))
    ].to_csv(destination / "paired_differences.csv", index=False)
    selection_intervals(combined, inputs.config.analysis).loc[
        lambda frame: frame.strategy.isin((*ENSEMBLE_STRATEGIES, *ONLINE_STRATEGIES))
    ].to_csv(destination / "paired_intervals.csv", index=False)
    summary = combined.groupby(["split", "strategy", "budget"], as_index=False)[
        ["normalized_auc", "relative_hypervolume"]
    ].mean()
    summary.to_csv(destination / "comparison.csv", index=False)
    decision_costs = result.groupby(["split", "strategy", "budget"], as_index=False)[
        [
            "decision_seconds",
            "decision_cpu_seconds",
            "preparation_seconds",
            "strategy_seconds",
        ]
    ].mean()
    decision_costs.to_csv(destination / "decision_costs.csv", index=False)
    (destination / "ADAPTIVE.md").write_text(
        "# Selección adaptativa con incertidumbre\n\n"
        + EVIDENCE_NOTE
        + f"\n\nFamilia congelada: {inputs.family}. Pliegues externos: {tuple(inputs.members)}.\n\n"
        + markdown_table(result[["strategy", "beta"]].drop_duplicates())
        + "\n\n"
        + "Para cada candidato se calcula la ganancia marginal de hipervolumen con cada miembro. "
        "ensemble_mean usa su media; ensemble_ucb suma beta por desviación muestral de esas ganancias. "
        "Se emplean los mismos catálogos, inicios, presupuestos y escalas del estudio. "
        "Los empates conservan el primer candidato disponible.\n\n"
        + markdown_table(summary)
        + "\n\n## Costos de las estrategias añadidas\n\n"
        + markdown_table(decision_costs)
        + "\n\nEstos tiempos son observaciones de esta ejecución, bajo la concurrencia registrada en execution.json, sin repeticiones de benchmark. "
        "Los tiempos históricos de greedy y random se recogieron en otro contexto de concurrencia y no se comparan aquí. "
        "La CPU de llamadas breves puede quedar bajo resolución del reloj; estas observaciones no reemplazan el benchmark agrupado. "
        "shared_costs.csv separa preparación del banco y cargas/predicciones de los miembros adicionales; "
        "la preparación del banco incluye al miembro representativo. La verificación contra predicciones guardadas se mide aparte. "
        "resources.json mide únicamente el coordinador durante evaluación y exportación. "
        "worker_jobs.csv y worker_resources.csv registran cada bloque; CPU incluye los hilos de monitorización. "
        "execution.json distingue tiempo total y trabajo acumulado. La suma de duraciones no equivale a tiempo transcurrido; "
        "el mayor RSS de un trabajador no es el pico global. telemetry.csv observa el árbol de procesos y la suma de RSS puede contar memoria compartida. "
        "El costo de entrenamiento ya realizado no se vuelve a imputar como trabajo ejecutado. "
        "Preparar un ensemble desde cero requeriría ajustar todos sus miembros; esos ajustes ya están registrados en el estudio.\n",
        encoding="utf-8",
    )


def adaptive_execution_parameters(
    config: StudyConfig, override: int | None
) -> tuple[ExecutionParameters, int]:
    requested = config.execution.cpu_workers if override is None else override
    execution = replace(config.execution, cpu_workers=requested)
    episodes = len(config.seeds) * config.episodes_per_seed * len(config.budgets)
    return replace(
        execution, cpu_workers=min(execution.effective_workers, episodes)
    ), requested


def run_adaptive_selection(
    study: Path,
    destination: Path,
    parameters: AdaptiveParameters | None = None,
) -> Path:
    if destination.exists():
        raise ValueError("La evaluación adaptativa ya existe; elegir otro destino")
    if not (study / "study.json").is_file():
        raise ValueError("Falta un estudio cerrado; ejecutar make study y make final")
    if destination.resolve().is_relative_to(study.resolve()):
        raise ValueError("Guardar la extensión fuera del estudio firmado, en reports")
    with study_lock(study):
        inputs = adaptive_inputs(study)
        if parameters is None:
            parameters = AdaptiveParameters(
                beta=inputs.config.adaptive.beta,
                online_residual=inputs.config.adaptive.online_residual,
            )
        if (parameters.beta, parameters.online_residual) != (
            inputs.config.adaptive.beta,
            inputs.config.adaptive.online_residual,
        ):
            raise ValueError(
                "La configuración adaptativa difiere del protocolo congelado"
            )
        execution, requested = adaptive_execution_parameters(
            inputs.config, parameters.cpu_workers
        )
        run = AdaptiveRun(
            execution,
            destination,
            inputs.config.data_dir,
            inputs.config.measurement,
            parameters.beta,
            online_residual=parameters.online_residual,
        )
        signatures = {
            name: source_digest(study / name) for name in ("study.json", "freeze.json")
        }
        destination.mkdir(parents=True, exist_ok=False)
        provenance: dict[str, JsonValue] = {
            "schema_version": ADAPTIVE_SCHEMA_VERSION,
            "status": "running",
            "study": str(study.resolve()),
            "state_signature": signatures["study.json"],
            "freeze_signature": signatures["freeze.json"],
            "dataset_signature": data_signature(inputs.config.data_dir),
            "training_source_signature": read_artifact(study / "study.json")[
                "source_signature"
            ],
            "analysis_source_signature": source_signature(),
            "parameters": asdict(parameters),
            "family": inputs.family,
            "members": [
                {
                    "member_id": seed,
                    "member_file": str(root),
                    "model_signatures": model_signatures(root),
                }
                for seed, root in inputs.members.items()
            ],
            "machine": machine_context(),
            "versions": environment_versions(),
            "retraining": False,
            "resource_scope": "coordinator_only; workers_in_worker_resources; tree_in_telemetry",
            "execution": {
                "cpu_workers_requested": requested,
                "cpu_workers_effective": execution.effective_workers,
                **asdict(execution),
            },
            "interpretation": "post_hoc_exploratory",
        }
        write_json(destination / "provenance.json", provenance)
        started = perf_counter()
        monitor: ResourceMonitor | None = None
        try:
            with ExitStack() as stack:
                monitor = stack.enter_context(
                    ResourceMonitor(inputs.config.measurement.memory_interval_ms)
                )
                run.telemetry = stack.enter_context(
                    CampaignMonitor(
                        destination / "telemetry.csv",
                        execution.telemetry_interval_seconds,
                        execution.effective_workers,
                    )
                )
                stack.enter_context(threadpool_limits(limits=1))
                if execution.effective_workers > 1:
                    run.executor = stack.enter_context(cpu_pool(execution))
                outputs = [
                    evaluate_adaptive_split(
                        SelectionConfig(
                            inputs.config.data_dir,
                            inputs.winner,
                            destination,
                            split=split,
                            catalog_size=inputs.config.catalog_size,
                            budgets=inputs.config.budgets,
                            seeds=inputs.config.seeds,
                            episodes_per_seed=inputs.config.episodes_per_seed,
                            evolution=inputs.config.evolution,
                        ),
                        inputs,
                        run,
                    )
                    for split in parameters.splits
                ]
                result = pd.concat(
                    [part.summary for part in outputs], ignore_index=True
                )
                result.to_csv(destination / "summary.csv", index=False)
                pd.concat([part.traces for part in outputs], ignore_index=True).to_csv(
                    destination / "traces.csv", index=False
                )
                pd.concat(
                    [part.diagnostics for part in outputs], ignore_index=True
                ).to_csv(destination / "uncertainty_diagnostics.csv", index=False)
                pd.DataFrame([row for part in outputs for row in part.costs]).to_csv(
                    destination / "shared_costs.csv", index=False
                )
                export_adaptive_comparison(inputs, result, destination)
            signed_records(study)
            if (
                any(
                    source_digest(study / name) != signature
                    for name, signature in signatures.items()
                )
                or data_signature(inputs.config.data_dir)
                != provenance["dataset_signature"]
            ):
                raise ValueError(
                    "El estudio o los datos cambiaron durante la evaluación"
                )
            provenance["status"] = "complete"
        except BaseException as error:
            provenance["status"] = "failed"
            provenance["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            run.executor = None
            totals = export_worker_tables(run)
            write_json(
                destination / "execution.json",
                {
                    "status": provenance["status"],
                    "wall_seconds": perf_counter() - started,
                    "cpu_workers_requested": requested,
                    "cpu_workers_effective": execution.effective_workers,
                    "scope": "post_startup_validation_through_final_integrity_checks; includes_pool_startup_and_shutdown",
                    "telemetry_errors": run.telemetry.errors
                    if run.telemetry is not None
                    else [],
                    **totals,
                },
            )
            if monitor is not None:
                write_json(
                    destination / "resources.json",
                    {
                        **monitor.snapshot(),
                        "cpu_scope": "coordinator_all_threads_excluding_worker_CPU",
                        "scope": "coordinator_only",
                    },
                )
            if provenance["status"] == "complete":
                provenance["report_signatures"] = {
                    name: value
                    for name, value in artifact_signatures(destination).items()
                    if name != "provenance.json"
                }
            write_json(destination / "provenance.json", provenance)
    return destination


def adaptive_export_root(study: Path) -> Path:
    from truss_research.evaluacion.registro import results_root

    return results_root(study)


def latest_adaptive_selection(study: Path) -> Path | None:
    for candidate in sorted(
        adaptive_export_root(study).glob("adaptativa_*"), reverse=True
    ):
        path = candidate / "provenance.json"
        if not path.is_file():
            continue
        provenance = read_artifact(path)
        if (
            provenance.get("status") != "complete"
            or provenance.get("schema_version") != ADAPTIVE_SCHEMA_VERSION
            or provenance.get("state_signature") != source_digest(study / "study.json")
            or provenance.get("freeze_signature")
            != source_digest(study / "freeze.json")
            or Path(str(provenance.get("study"))).resolve() != study.resolve()
        ):
            continue
        actual = {
            name: value
            for name, value in artifact_signatures(candidate).items()
            if name != "provenance.json"
        }
        if not actual or actual != provenance.get("report_signatures"):
            raise ValueError(
                "Cambió una tabla, traza o medición de selección adaptativa"
            )
        return candidate
    return None
