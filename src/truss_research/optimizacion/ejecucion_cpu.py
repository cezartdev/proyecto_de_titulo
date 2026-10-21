"""Pool único de procesos CPU; cierre explícito de trabajadores al interrumpir."""

import gc
import logging
import multiprocessing
import os
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from threading import Lock
from time import monotonic, time

import psutil
import torch

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
WORKER_SHUTDOWN_SECONDS = 3
MEMORY_WAIT_SECONDS = 30.0
MEMORY_RECOVERY_DIVISOR = 2
CAMPAIGN_CPU_WORKERS = 12
CAMPAIGN_GPU_WORKERS = 1
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionParameters:
    cpu_workers: int = 1
    gpu_workers: int = 1
    gpu_memory_limit_fraction: float = 0.80
    rl_device: str = "cpu"
    gpu_models: tuple[str, ...] = ()
    memory_limit_fraction: float = 0.80
    telemetry_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.gpu_memory_limit_fraction, bool)
            or not isfinite(self.gpu_memory_limit_fraction)
            or not 0 < self.gpu_memory_limit_fraction < 1
        ):
            raise ValueError("El límite de VRAM debe estar entre cero y uno")
        object.__setattr__(self, "gpu_models", tuple(self.gpu_models))
        if self.rl_device not in ("cpu", "cuda") or set(self.gpu_models) - {
            "mlp",
            "catboost",
            "xgboost",
        }:
            raise ValueError("Dispositivo RL o familias GPU no compatibles")
        if type(self.gpu_workers) is not int or self.gpu_workers not in (1, 2, 4):
            raise ValueError("El perfil GPU admite 1, 2 o 4 trabajadores calibrados")
        if type(self.cpu_workers) is not int or self.cpu_workers < 1:
            raise ValueError("cpu_workers debe ser un entero positivo")
        if (
            isinstance(self.memory_limit_fraction, bool)
            or not isfinite(self.memory_limit_fraction)
            or not 0 < self.memory_limit_fraction < 1
        ):
            raise ValueError("El límite de memoria debe estar entre cero y uno")
        if (
            isinstance(self.telemetry_interval_seconds, bool)
            or not isfinite(self.telemetry_interval_seconds)
            or self.telemetry_interval_seconds <= 0
        ):
            raise ValueError("El intervalo de telemetría debe ser positivo")

    @property
    def effective_workers(self) -> int:
        return min(self.cpu_workers, os.cpu_count() or 1)

    def validate_campaign(self) -> None:
        """Restringe campañas; los parámetros generales siguen disponibles en pilotos."""
        if self.rl_device != "cpu" or self.gpu_models:
            raise ValueError(
                "La campaña permite GPU exclusivamente para TabPFN; "
                "usar rl_device=cpu y gpu_models=[]"
            )
        if self.gpu_workers != CAMPAIGN_GPU_WORKERS:
            raise ValueError("La campaña admite un único trabajador GPU para TabPFN")
        if self.cpu_workers > CAMPAIGN_CPU_WORKERS:
            raise ValueError("La campaña admite como máximo doce trabajadores CPU")


@dataclass
class MemoryAdmission:
    parameters: ExecutionParameters
    blocked_since: float | None = None

    def allow(self, active: int) -> bool:
        if (
            psutil.virtual_memory().percent
            <= self.parameters.memory_limit_fraction * 100
        ):
            self.blocked_since = None
            return True
        if active:
            self.blocked_since = None
            return False
        if self.blocked_since is None:
            self.blocked_since = monotonic()
        if monotonic() - self.blocked_since >= MEMORY_WAIT_SECONDS:
            raise RuntimeError(
                "Memoria insuficiente: el sistema supera el límite durante 30 segundos sin tareas activas; conservar el estudio y reanudar con memoria disponible"
            )
        return False


def gpu_admission(parameters: ExecutionParameters, active: int) -> bool:
    free, total = torch.cuda.mem_get_info()
    available = free > total * (1 - parameters.gpu_memory_limit_fraction)
    if not available and not active:
        raise RuntimeError(
            "VRAM insuficiente para admitir un trabajo GPU; liberar memoria y reanudar"
        )
    return available


def cpu_operation[Argument, Result](
    command: tuple[Callable[[Argument], Result], Argument],
) -> Result:
    operation, argument = command
    try:
        return operation(argument)
    finally:
        gc.collect()


class CpuPool:
    """Recicla procesos únicamente al quedar libres y reduce la concurrencia efectiva."""

    def __init__(self, parameters: ExecutionParameters, device: str = "cpu") -> None:
        self.device = device
        self.capacity = (
            parameters.gpu_workers if device == "cuda" else parameters.effective_workers
        )
        self.executor = self.create_executor()
        self.pending = 0
        self.lock = Lock()
        self.pressure_recycled = False
        self.recoveries: list[dict[str, int | float]] = []

    def create_executor(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self.capacity,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_cpu_worker,
            initargs=(self.device,),
        )

    def release(self) -> None:
        with self.lock:
            self.pending -= 1

    def submit[Argument, Result](
        self, operation: Callable[[Argument], Result], argument: Argument
    ) -> Future[Result]:
        with self.lock:
            self.pending += 1
        try:
            future = self.executor.submit(cpu_operation, (operation, argument))
        except BaseException:
            self.release()
            raise
        future.add_done_callback(lambda result: self.release())
        return future

    def recycle(self) -> bool:
        with self.lock:
            if self.pending:
                return False
            before = psutil.virtual_memory().percent
            previous = self.capacity
            self.executor.shutdown(wait=True)
            gc.collect()
            self.capacity = max(1, self.capacity // MEMORY_RECOVERY_DIVISOR)
            self.executor = self.create_executor()
        after = psutil.virtual_memory().percent
        self.recoveries.append(
            {
                "timestamp": time(),
                "previous_capacity": previous,
                "capacity": self.capacity,
                "ram_before_percent": before,
                "ram_after_percent": after,
            }
        )
        logger.warning(
            "Recuperación de memoria: pool inactivo reciclado, límite efectivo %s -> %s; RAM %.1f%% -> %.1f%%. Las tareas completas se conservan",
            previous,
            self.capacity,
            before,
            after,
        )
        return True

    def admit(self, gate: MemoryAdmission, active: int) -> bool:
        if gate.allow(active):
            self.pressure_recycled = False
            return True
        if active or self.pressure_recycled:
            return False
        if not self.recycle():
            return False
        self.pressure_recycled = True
        gate.blocked_since = None
        admitted = gate.allow(0)
        if admitted:
            self.pressure_recycled = False
        return admitted

    def shutdown(self, cancel_futures: bool = False) -> None:
        self.executor.shutdown(wait=True, cancel_futures=cancel_futures)


def initialize_cpu_worker(device: str = "cpu") -> None:
    os.environ.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if device == "cuda" else ""
    os.environ["TRUSS_WORKER_DEVICE"] = device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("El trabajador GPU requiere PyTorch CUDA disponible")
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO, format="%(processName)s: %(message)s")


def is_pool_worker(process: psutil.Process, previous: set[int]) -> bool:
    try:
        return process.pid not in previous and "multiprocessing.spawn" in " ".join(
            process.cmdline()
        )
    except psutil.NoSuchProcess:
        return False


def stop_pool_workers(previous: set[int]) -> None:
    candidates = psutil.Process().children(recursive=True)
    workers = list(
        reversed(
            [process for process in candidates if is_pool_worker(process, previous)]
        )
    )
    for process in workers:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            logger.info("Trabajador %s ya terminado", process.pid)
    _, alive = psutil.wait_procs(workers, timeout=WORKER_SHUTDOWN_SECONDS)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            logger.info("Trabajador %s ya terminado", process.pid)
    psutil.wait_procs(alive, timeout=WORKER_SHUTDOWN_SECONDS)


@contextmanager
def cpu_pool(parameters: ExecutionParameters, device: str = "cpu") -> Iterator[CpuPool]:
    previous = {process.pid for process in psutil.Process().children(recursive=True)}
    environment = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    executor = CpuPool(parameters, device)
    try:
        yield executor
    except BaseException:
        stop_pool_workers(previous)
        executor.shutdown(cancel_futures=True)
        raise
    else:
        executor.shutdown()
    finally:
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
