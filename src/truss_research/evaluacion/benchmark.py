"""Benchmark explícito sobre artefactos congelados, sin entrenamiento."""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue, load_design_split
from truss_research.evaluacion.benchmark_seleccion import selection_benchmark
from truss_research.evaluacion.costos import machine_context
from truss_research.evaluacion.precision_cpu import (
    cpu_batches,
    probe_cpu_clock,
)
from truss_research.evaluacion.recursos import ResourceMonitor, timed
from truss_research.evaluacion.sensibilidad import sensitivity_inputs
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    artifact_signatures,
    environment_versions,
    load_study_config,
    source_signature,
    study_lock,
)
from truss_research.surrogates.experimentos import load_predictor, write_json
from truss_research.surrogates.persistencia import (
    model_signatures,
    prediction_device,
    prediction_devices,
)

BENCHMARK_SCHEMA_VERSION = 6
logger = logging.getLogger(__name__)


def inference_benchmark(
    config: StudyConfig, tasks: dict[str, Path], destination: Path
) -> None:
    split = load_design_split(config.data_dir)
    rows: list[dict[str, JsonValue]] = []
    memory: list[dict[str, JsonValue]] = []
    cpu_samples: list[dict[str, JsonValue]] = []
    clock = probe_cpu_clock(config.measurement.cpu_probe_seconds)
    artifacts: list[dict[str, JsonValue]] = []
    rng = np.random.default_rng(config.measurement.seed)
    identities = [(name, seed) for name in config.models for seed in config.seeds]
    for index in rng.permutation(len(identities)):
        name, seed = identities[int(index)]
        logger.info("Inferencia: %s, semilla %s", name, seed)
        root = tasks[f"predictor/{name}/{seed}"]
        model, loading = timed(lambda: load_predictor(root))
        identity: dict[str, JsonValue] = {
            "model": name,
            "seed": seed,
            "inference_device": prediction_device(model),
            "inference_components": prediction_devices(model),
        }
        rows.append(
            {
                **identity,
                "split": "shared",
                "operation": "load",
                "batch_size": 0,
                "repetition": 0,
                **loading,
            }
        )
        artifacts.append(
            {
                **identity,
                "artifact": "predictor",
                "size_mib": sum(
                    (root / name).stat().st_size for name in model_signatures(root)
                )
                / 1024**2,
            }
        )
        if name == "tabpfn" and seed == config.representative_seed:
            artifacts.append(
                {
                    **identity,
                    "artifact": "shared_foundation_weights",
                    "size_mib": config.tabpfn.checkpoint.stat().st_size / 1024**2,
                }
            )
        for split_name, table in (
            ("validation", split.validation),
            ("test", split.testing),
        ):
            indices = np.random.default_rng(config.measurement.seed).permutation(
                len(table)
            )
            features = table.iloc[indices].loc[:, list(INPUT_COLUMNS)]
            for size in config.measurement.batch_sizes:
                for _ in range(config.measurement.warmup):
                    model.predict(features.iloc[:size])
                for repetition in range(config.measurement.inference_repeats):
                    rotated = np.roll(np.arange(len(features)), -repetition)
                    batch = features.iloc[rotated[:size]]
                    _, timing = timed(lambda: model.predict(batch))
                    rows.append(
                        {
                            **identity,
                            "split": split_name,
                            "operation": "predict",
                            "batch_size": size,
                            "repetition": repetition,
                            "design_ids": ";".join(
                                table.iloc[indices[rotated[:size]]].iter_id.astype(str)
                            ),
                            **timing,
                        }
                    )
                cpu_features = features.iloc[:size]

                def cpu_prediction() -> None:
                    model.predict(cpu_features)

                cpu_samples.extend(
                    {
                        **identity,
                        "split": split_name,
                        "batch_size": size,
                        "operation": "predict",
                        "scope": "complete_predict_call",
                        **sample,
                    }
                    for sample in cpu_batches(cpu_prediction, config.measurement, clock)
                )
                with ResourceMonitor(config.measurement.memory_interval_ms) as monitor:
                    model.predict(features.iloc[:size])
                memory.append(
                    {
                        **identity,
                        "split": split_name,
                        "batch_size": size,
                        **monitor.snapshot(),
                    }
                )
    for variant in config.rl.variants:
        for seed in config.seeds:
            root = tasks[f"policy/{variant}/{seed}"]
            artifacts.append(
                {
                    "model": variant,
                    "seed": seed,
                    "artifact": "policy_and_encoder",
                    "size_mib": sum(
                        (root / filename).stat().st_size
                        for filename in ("policy.pt", "encoder.joblib")
                    )
                    / 1024**2,
                }
            )
    pd.DataFrame(rows).to_csv(destination / "inference_samples.csv", index=False)
    pd.DataFrame(memory).to_csv(destination / "inference_memory.csv", index=False)
    pd.DataFrame(cpu_samples).to_csv(
        destination / "inference_cpu_batches.csv", index=False
    )
    pd.DataFrame(artifacts).to_csv(destination / "artifact_sizes.csv", index=False)


def benchmark_export_root(study: Path) -> Path:
    from truss_research.evaluacion.registro import results_root

    return results_root(study)


def verify_benchmark(study: Path, destination: Path) -> dict[str, JsonValue]:
    provenance = read_artifact(destination / "provenance.json")
    if (
        provenance.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or provenance.get("status") != "complete"
    ):
        raise ValueError("El benchmark está incompleto o tiene un formato incompatible")
    if (
        any(
            provenance[key] != source_digest(study / name)
            for key, name in (
                ("state_signature", "study.json"),
                ("freeze_signature", "freeze.json"),
            )
        )
        or Path(str(provenance["study"])).resolve() != study.resolve()
    ):
        raise ValueError("El benchmark no corresponde al estudio congelado")
    actual = {
        name: value
        for name, value in artifact_signatures(destination).items()
        if name != "provenance.json"
    }
    if actual != provenance.get("report_signatures"):
        raise ValueError("Cambió una medición, tabla o figura del benchmark")
    return provenance


def latest_benchmark(study: Path) -> Path | None:
    for candidate in sorted(
        benchmark_export_root(study).glob("benchmark_*"), reverse=True
    ):
        path = candidate / "provenance.json"
        if not path.is_file():
            continue
        provenance = read_artifact(path)
        if provenance.get("status") != "complete":
            continue
        if provenance.get("state_signature") != source_digest(study / "study.json"):
            continue
        verify_benchmark(study, candidate)
        return candidate
    return None


def run_benchmark(study: Path, destination: Path) -> Path:
    from truss_research.evaluacion.informe_costos import export_benchmark_report

    if destination.exists():
        raise ValueError("El benchmark ya existe; elegir un destino nuevo")
    if not (study / "study.json").is_file():
        raise ValueError(
            "Falta el estudio; ejecutar make study y make final antes del benchmark"
        )
    with study_lock(study):
        state, tasks = sensitivity_inputs(study)
        config = load_study_config(study)
        split = load_design_split(config.data_dir)
        if max(config.measurement.batch_sizes) > min(
            len(split.validation), len(split.testing)
        ):
            raise ValueError("Los lotes del benchmark superan los diseños disponibles")
        signatures = {
            name: source_digest(study / name) for name in ("study.json", "freeze.json")
        }
        frozen = read_artifact(study / "freeze.json")
        winner = Path(str(frozen["winner"]))
        destination.mkdir(parents=True, exist_ok=False)
        provenance: dict[str, JsonValue] = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "status": "running",
            "study": str(study.resolve()),
            "state_signature": signatures["study.json"],
            "freeze_signature": signatures["freeze.json"],
            "dataset_signature": state["dataset_signature"],
            "source_signature": source_signature(),
            "training_source_signature": state["source_signature"],
            "measurement": cast(dict[str, JsonValue], asdict(config.measurement)),
            "machine": machine_context(),
            "versions": environment_versions(),
            "retraining": False,
            "prediction_values": "batch_predictions_verified_against_reference; reference_values_preserve_trajectories",
            "memory_scope": "sampled_python_worker_RSS; baseline_includes_loaded_libraries_and_models",
        }
        write_json(destination / "provenance.json", provenance)
        try:
            torch.set_num_threads(1)
            with threadpool_limits(limits=1):
                logger.info("Benchmark de inferencia: %s", destination)
                inference_benchmark(config, tasks, destination)
                logger.info("Benchmark de selección")
                selection_benchmark(config, (winner, tasks), destination)
            memory_files = (
                "inference_memory.csv",
                "selection_memory.csv",
            )
            for name in memory_files:
                memory = pd.read_csv(destination / name)
                if (
                    not memory.cpu_coverage.eq("complete").all()
                    or not memory.memory_coverage.eq("sampled").all()
                ):
                    raise ValueError(
                        "El monitor registró cobertura incompleta en el benchmark"
                    )
            export_benchmark_report(study, destination)
            sensitivity_inputs(study)
            if any(
                source_digest(study / name) != value
                for name, value in signatures.items()
            ):
                raise ValueError("El estudio cambió durante el benchmark")
            provenance["status"] = "complete"
            provenance["report_signatures"] = artifact_signatures(destination)
            provenance["report_signatures"].pop("provenance.json", None)
        except BaseException as error:
            provenance["status"], provenance["error"] = (
                "failed",
                f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            write_json(destination / "provenance.json", provenance)
    return destination
