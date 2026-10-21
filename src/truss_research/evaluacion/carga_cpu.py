"""Carga sintética pequeña con los estimadores reales y geometría neuronal fija."""

import json
import os
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from time import monotonic, perf_counter, process_time, sleep
from typing import cast

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from truss_research.cerchas.schemas import DESIGN_BOUNDS, INPUT_COLUMNS, INTEGER_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.evaluacion.pruebas_neuronales import (
    NeuralCase,
    learner_for,
    synthetic_replay,
    update_sequence,
)
from truss_research.surrogates.configuracion import (
    FloatArray,
    ModelName,
    ModelParameters,
    PredictorConfig,
)
from truss_research.surrogates.modelos import build_predictor

CALIBRATION_KINDS = ("random_forest", "catboost", "gpr", "mlp", "dqn")
WARMUP_ROWS = 32
PREDICTION_ROWS = 16
WARMUP_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class CpuCalibration:
    workers: tuple[int, ...] = (4, 6, 8, 12)
    screen_rows: int = 256
    confirmation_rows: int = 512
    confirmation_repeats: int = 3
    copies_per_kind: int = 6
    trees: int = 30
    epochs: int = 20
    updates: int = 100
    seed: int = 42
    budget_seconds: int = 300
    screen_seconds: int = 90
    selection_tolerance: float = 0.05
    memory_limit_fraction: float = 0.80

    def __post_init__(self) -> None:
        values = (
            self.screen_rows,
            self.confirmation_rows,
            self.confirmation_repeats,
            self.copies_per_kind,
            self.trees,
            self.epochs,
            self.updates,
            self.budget_seconds,
            self.screen_seconds,
            *self.workers,
        )
        if not self.workers or any(
            type(value) is not int or value < 1 for value in values
        ):
            raise ValueError(
                "La calibración requiere tamaños y cantidades enteros positivos"
            )
        if (
            min(self.screen_rows, self.confirmation_rows) < WARMUP_ROWS
            or self.screen_seconds >= self.budget_seconds
            or not 0 <= self.selection_tolerance < 1
            or not 0 < self.memory_limit_fraction < 1
        ):
            raise ValueError("Filas, presupuesto, tolerancia o memoria inválidos")


@dataclass(frozen=True)
class CpuWork:
    kind: str
    rows: int
    seed: int
    protocol: CpuCalibration


@dataclass(frozen=True)
class CpuWorkResult:
    kind: str
    seed: int
    pid: int
    wall_seconds: float
    cpu_seconds: float
    values: tuple[float, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class WarmupCommand:
    directory: Path
    workers: int
    deadline: float
    protocol: CpuCalibration


def load_cpu_protocol(path: Path | None = None) -> CpuCalibration:
    if path is None:
        return CpuCalibration()
    payload = read_artifact(path)
    if set(payload) - {field.name for field in fields(CpuCalibration)}:
        raise ValueError("Campos desconocidos en calibración CPU")
    if "workers" in payload:
        if not isinstance(payload["workers"], list):
            raise ValueError("workers requiere una lista JSON")
        payload["workers"] = tuple(payload["workers"])
    return CpuCalibration(**payload)


def synthetic_features(work: CpuWork) -> tuple[pd.DataFrame, FloatArray]:
    rng = np.random.default_rng(work.seed)
    columns = {
        name: rng.integers(
            int(DESIGN_BOUNDS[name][0]), int(DESIGN_BOUNDS[name][1]) + 1, work.rows
        )
        if name in INTEGER_COLUMNS
        else rng.uniform(*DESIGN_BOUNDS[name], work.rows)
        for name in INPUT_COLUMNS
    }
    table = pd.DataFrame(columns, columns=INPUT_COLUMNS)
    x = table.lr_height_m.to_numpy()
    y = table.lmrm_height_m.to_numpy()
    responses = np.column_stack(
        (
            1000 + 100 * x + 50 * np.sin(y),
            0.5 + 0.1 * np.cos(x) + 0.01 * table.web_config.to_numpy(),
            0.02 + 0.005 * np.sin(x + y),
        )
    )
    return table, responses


def execute_cpu_work(work: CpuWork) -> CpuWorkResult:
    wall, cpu = perf_counter(), process_time()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if work.kind == "dqn":
            case = NeuralCase("cpu_calibration", width=64, batch=64, catalog_size=32)
            batch = synthetic_replay(case, work.seed)
            learner = learner_for(case, (work.seed, "cpu"))
            values = (
                update_sequence(learner, (batch, "cpu", True), work.protocol.updates),
            )
        else:
            features, responses = synthetic_features(work)
            settings = ModelParameters(
                trees=work.protocol.trees,
                epochs=work.protocol.epochs,
                hidden_width=64,
                batch_size=64,
            )
            model = build_predictor(
                PredictorConfig(
                    model=cast(ModelName, work.kind),
                    seed=work.seed,
                    parameters=settings,
                )
            )
            model.fit(features, responses)
            values = tuple(
                float(value)
                for value in np.asarray(
                    model.predict(features.iloc[:PREDICTION_ROWS])
                ).ravel()
            )
    return CpuWorkResult(
        work.kind,
        work.seed,
        os.getpid(),
        perf_counter() - wall,
        process_time() - cpu,
        values,
        tuple(str(warning.message) for warning in captured),
    )


def warm_cpu_worker(command: WarmupCommand) -> int:
    from dataclasses import replace

    tiny = replace(command.protocol, trees=1, epochs=1, updates=1)
    diagnostics = [
        execute_cpu_work(CpuWork(kind, WARMUP_ROWS, tiny.seed, tiny))
        for kind in CALIBRATION_KINDS
    ]
    pid = os.getpid()
    (command.directory / f"{pid}.ready").write_text(
        json.dumps(
            {
                "pid": pid,
                "warnings": [
                    message for result in diagnostics for message in result.warnings
                ],
            }
        ),
        encoding="utf-8",
    )
    while len(list(command.directory.glob("*.ready"))) < command.workers:
        if monotonic() >= command.deadline:
            raise TimeoutError("Se agotó el presupuesto al calentar trabajadores")
        sleep(WARMUP_POLL_SECONDS)
    return pid


def work_batch(
    protocol: CpuCalibration, rows: int, repetition: int
) -> tuple[CpuWork, ...]:
    jobs = [
        CpuWork(kind, rows, protocol.seed + number, protocol)
        for kind in CALIBRATION_KINDS
        for number in range(protocol.copies_per_kind)
    ]
    order = np.random.default_rng(protocol.seed + repetition).permutation(len(jobs))
    return tuple(jobs[int(index)] for index in order)
