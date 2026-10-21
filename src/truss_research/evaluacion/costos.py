"""Costos efectivos por fase, sin duplicar entrenamiento al evaluar prueba."""

import os
import platform
from pathlib import Path

import pandas as pd
import psutil
import torch

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.procesados import JsonValue


def machine_context() -> dict[str, str | int | float]:
    cuda_available = torch.cuda.is_available()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "no disponible",
        "logical_cpus": os.cpu_count() or 1,
        "physical_cpus": psutil.cpu_count(logical=False) or 1,
        "ram_mib": psutil.virtual_memory().total / 1024**2,
        "numeric_threads": 1,
        "device": os.environ.get("TRUSS_WORKER_DEVICE", "cpu"),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda or "not_available",
        "gpu_name": torch.cuda.get_device_name(0)
        if cuda_available
        else "not_available",
    }


def export_effective_costs(
    records: tuple[dict[str, JsonValue], ...],
    destination: Path,
    preparation: tuple[dict[str, JsonValue], ...] = (),
) -> None:
    rows: list[dict[str, str | float]] = []
    for record in records:
        key = str(record["key"])
        run = read_artifact(Path(str(record["directory"])) / "run.json")
        phases: dict[str, float] = {}
        if key.startswith(("predictor/", "comparator/")):
            phases = {
                name: float(str(run.get(name, 0)))
                for name in (
                    "tuning_seconds",
                    "crossfit_seconds",
                    "fit_seconds",
                    "reuse_seconds",
                    "prediction_seconds",
                )
            }
        elif key.startswith(("test/predictor/", "test/comparator/")):
            phases = {
                "prediction_seconds": float(str(run.get("prediction_seconds", 0)))
            }
        elif key.startswith("policy/"):
            phases = {
                "policy_optimization_seconds": float(str(run["optimization_seconds"])),
                "policy_validation_seconds": float(str(run["validation_wall_seconds"])),
            }
        else:
            phases = {"task_wall_seconds": float(str(record["seconds"]))}
        rows.extend(
            {"task": key, "phase": name, "seconds": value}
            for name, value in phases.items()
        )
        if key.startswith(
            (
                "predictor/",
                "test/predictor/",
                "comparator/",
                "test/comparator/",
                "policy/",
            )
        ):
            rows.append(
                {
                    "task": key,
                    "phase": "task_wall_seconds",
                    "seconds": float(str(record["seconds"])),
                }
            )
    rows.extend(
        {
            "task": str(record["key"]),
            "phase": "task_wall_seconds",
            "seconds": float(str(record["seconds"])),
        }
        for record in preparation
    )
    table = pd.DataFrame(rows, columns=["task", "phase", "seconds"])
    table.to_csv(destination / "effective_costs.csv", index=False)
    table.groupby("phase", as_index=False)["seconds"].sum().to_csv(
        destination / "cost_totals.csv", index=False
    )
