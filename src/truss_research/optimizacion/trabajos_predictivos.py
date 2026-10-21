"""Trabajos serializables: un proceso ejecuta una búsqueda, pliegue o ajuste."""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter, time
from typing import Literal

from threadpoolctl import threadpool_limits

from truss_research.evaluacion.recursos import MeasurementParameters, resource_session
from truss_research.evaluacion.registro import execution_log
from truss_research.surrogates.configuracion import ModelParameters, PredictorConfig
from truss_research.surrogates.experimentos import (
    data_signature,
    read_run,
    train_predictor,
)
from truss_research.surrogates.preparacion import (
    assemble_preparation,
    prepare_outer_fold,
    prepare_search,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictorJob:
    key: str
    kind: Literal["search", "fold", "predictor"]
    config: PredictorConfig
    measurement: MeasurementParameters
    fold: int = 0
    search: Path | None = None
    folds: tuple[Path, ...] = ()


@dataclass(frozen=True)
class JobOutcome:
    success: bool
    seconds: float
    pid: int
    started_at: float
    finished_at: float
    error: str | None = None


@dataclass(frozen=True)
class JobResources:
    key: str
    destination: Path
    data_dir: Path
    measurement: MeasurementParameters


def execute_job(job: PredictorJob) -> None:
    config = job.config
    if job.kind == "search":
        prepare_search(config, config.output_dir)
        return
    if job.kind == "fold":
        if config.nested_folds is None and job.search is not None:
            config = replace(
                config, parameters=ModelParameters(**read_run(job.search)["parameters"])
            )
        prepare_outer_fold(config, config.output_dir, job.fold)
        return
    if job.search is not None:
        preparation = config.output_dir.with_suffix(".prepared")
        assemble_preparation(config, (job.search, job.folds), preparation)
        config = replace(config, preparation_dir=preparation)
    train_predictor(config)


def run_predictor_job(job: PredictorJob) -> JobOutcome:
    return run_measured_job(
        JobResources(
            job.key, job.config.output_dir, job.config.data_dir, job.measurement
        ),
        lambda: execute_job(job),
    )


def run_measured_job(job: JobResources, operation: Callable[[], None]) -> JobOutcome:
    started, clock = time(), perf_counter()
    logger.info("Iniciando %s", job.key)
    try:
        with (
            execution_log(
                job.destination.with_suffix(".log"),
                job.key,
                {"data_dir": str(job.data_dir)},
            ),
            threadpool_limits(limits=1),
            resource_session(
                job.destination.with_suffix(".resources.json"), job.measurement
            ),
        ):
            signature = data_signature(job.data_dir)
            operation()
            if signature != data_signature(job.data_dir):
                raise ValueError("Los datos cambiaron durante el trabajo paralelo")
    except Exception as error:
        logger.exception("Falló %s", job.key)
        return JobOutcome(
            False,
            perf_counter() - clock,
            os.getpid(),
            started,
            time(),
            f"{type(error).__name__}: {error}",
        )
    logger.info("Completado %s", job.key)
    return JobOutcome(True, perf_counter() - clock, os.getpid(), started, time())
