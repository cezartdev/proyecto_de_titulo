"""Mediciones inclusivas por fase y checkpoints de recursos del proceso de trabajo."""

import json
import logging
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from math import isfinite
from pathlib import Path
from time import perf_counter, process_time
from types import TracebackType
from typing import ParamSpec, TypeVar

import psutil

from truss_research.datos.artefactos import replace_artifact
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.dispositivos import device_context, synchronize_cuda

MIB = 1024**2
RESOURCE_SCHEMA_VERSION = 1
CHECKPOINT_SECONDS = 5.0
logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


MEMORY_INTERVAL_MS = 50


@dataclass(frozen=True)
class MeasurementParameters:
    memory_interval_ms: int = MEMORY_INTERVAL_MS

    def __post_init__(self) -> None:
        if type(self.memory_interval_ms) is not int or self.memory_interval_ms < 1:
            raise ValueError("El intervalo de memoria debe ser un entero positivo")


class ResourceMonitor:
    """RSS muestreada; CPU incluye todos los hilos del proceso, también el monitor."""

    def __init__(self, interval_ms: int = MEMORY_INTERVAL_MS) -> None:
        if type(interval_ms) is not int or interval_ms < 1:
            raise ValueError("El intervalo de memoria debe ser positivo")
        self.interval_ms = interval_ms
        self.process = psutil.Process(os.getpid())
        self.start_wall = 0.0
        self.start_cpu = 0.0
        self.wall_seconds = 0.0
        self.cpu_seconds = 0.0
        self.initial_rss_mib: float | None = None
        self.peak_rss_mib: float | None = None
        self.samples = 0
        self.children_seen: set[int] = set()
        self.child_names: dict[int, str] = {}
        self.errors: list[str] = []
        self.finished = False
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.callback: Callable[[], None] | None = None

    def sample(self) -> None:
        try:
            rss = self.process.memory_info().rss / MIB
            self.initial_rss_mib = (
                rss if self.initial_rss_mib is None else self.initial_rss_mib
            )
            self.peak_rss_mib = max(self.peak_rss_mib or rss, rss)
            for child in self.process.children(recursive=True):
                self.children_seen.add(child.pid)
                self.record_child(child)
            self.samples += 1
        except (psutil.Error, OSError) as error:
            message = f"{type(error).__name__}: {error}"
            if message not in self.errors:
                logger.warning("Medición de recursos incompleta: %s", message)
                self.errors.append(message)

    def record_child(self, child: psutil.Process) -> None:
        """La desaparición de un hijo no invalida la medición del proceso padre."""
        try:
            self.child_names[child.pid] = child.name()
        except psutil.Error:
            self.child_names.setdefault(child.pid, "identity_unavailable")

    def poll(self) -> None:
        while not self.stop.wait(self.interval_ms / 1000):
            self.sample()
            if self.callback is not None:
                try:
                    self.callback()
                except OSError as error:
                    message = f"Checkpoint de recursos: {error}"
                    logger.warning(message)
                    self.errors.append(message)

    def __enter__(self) -> "ResourceMonitor":
        self.sample()
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        self.reset_clock()
        return self

    def reset_clock(self) -> None:
        synchronize_cuda()
        self.start_wall, self.start_cpu = perf_counter(), process_time()

    def __exit__(
        self, *exception: type[BaseException] | BaseException | TracebackType | None
    ) -> None:
        synchronize_cuda()
        self.wall_seconds = perf_counter() - self.start_wall
        self.cpu_seconds = process_time() - self.start_cpu
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        self.sample()
        self.finished = True

    def snapshot(self) -> dict[str, JsonValue]:
        wall = self.wall_seconds if self.finished else perf_counter() - self.start_wall
        cpu = self.cpu_seconds if self.finished else process_time() - self.start_cpu
        return {
            **device_context(),
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "initial_rss_mib": self.initial_rss_mib,
            "peak_rss_mib": self.peak_rss_mib,
            "peak_increment_mib": self.peak_rss_mib - self.initial_rss_mib
            if self.peak_rss_mib is not None and self.initial_rss_mib is not None
            else None,
            "samples": self.samples,
            "memory_interval_ms": self.interval_ms,
            "cpu_scope": "python_worker_all_threads_including_monitor",
            "cpu_coverage": "complete" if not self.errors else "incomplete",
            "cpu_tree_coverage": "unmeasured_children"
            if self.children_seen
            else "no_children_observed",
            "coverage_policy_version": 2,
            "memory_coverage": "sampled" if not self.errors else "incomplete",
            "children_seen": sorted(self.children_seen),
            "child_processes": [
                {"pid": pid, "name": self.child_names[pid]}
                for pid in sorted(self.child_names)
            ],
            "errors": self.errors.copy(),
            "finished": self.finished,
        }


@dataclass
class PhaseRecord:
    phase: str
    monitor: ResourceMonitor
    status: str = "running"


@dataclass
class ResourceLedger:
    path: Path
    parameters: MeasurementParameters
    records: list[PhaseRecord] = field(default_factory=list)
    status: str = "running"
    lock: threading.RLock = field(default_factory=threading.RLock)
    last_save: float = 0.0

    def save(self) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": RESOURCE_SCHEMA_VERSION,
                "status": self.status,
                "phases": [
                    {
                        "phase": record.phase,
                        "status": record.status,
                        **record.monitor.snapshot(),
                    }
                    for record in self.records
                ],
            }
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            replace_artifact(temporary, self.path)
            self.last_save = perf_counter()

    def checkpoint(self) -> None:
        if perf_counter() - self.last_save >= CHECKPOINT_SECONDS:
            self.save()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        with ResourceMonitor(self.parameters.memory_interval_ms) as monitor:
            record = PhaseRecord(name, monitor)
            self.records.append(record)
            self.save()
            monitor.reset_clock()
            try:
                yield
            except BaseException:
                record.status = "failed"
                raise
            else:
                record.status = "complete"
        self.save()


CURRENT_LEDGER: ContextVar[ResourceLedger | None] = ContextVar(
    "resource_ledger", default=None
)


@contextmanager
def measure_phase(name: str) -> Iterator[None]:
    ledger = CURRENT_LEDGER.get()
    if ledger is None:
        yield
        return
    with ledger.phase(name):
        yield


def measured(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def invoke(*args: P.args, **kwargs: P.kwargs) -> R:
            with measure_phase(name):
                return function(*args, **kwargs)

        return invoke

    return decorate


@contextmanager
def resource_session(path: Path, parameters: MeasurementParameters) -> Iterator[None]:
    ledger = ResourceLedger(path, parameters)
    token = CURRENT_LEDGER.set(ledger)
    try:
        with ResourceMonitor(parameters.memory_interval_ms) as monitor:
            ledger.records.append(PhaseRecord("task_total", monitor))
            ledger.save()
            monitor.callback = ledger.checkpoint
            monitor.reset_clock()
            yield
        ledger.status = "complete"
    except BaseException:
        ledger.status = "failed"
        raise
    finally:
        ledger.records[0].status = ledger.status
        ledger.save()
        CURRENT_LEDGER.reset(token)


def valid_seconds(value: float) -> bool:
    return isfinite(value) and value >= 0


def timed[T](function: Callable[[], T]) -> tuple[T, dict[str, float]]:
    synchronize_cuda()
    start_wall, start_cpu = perf_counter(), process_time()
    result = function()
    synchronize_cuda()
    return result, {
        "wall_seconds": perf_counter() - start_wall,
        "cpu_seconds": process_time() - start_cpu,
    }
