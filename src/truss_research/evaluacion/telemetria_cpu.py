"""Muestreo del sistema y árbol de trabajadores, separado de los costos por tarea."""

import csv
import logging
import os
import threading
from pathlib import Path
from time import perf_counter, time
from types import TracebackType

import psutil

logger = logging.getLogger(__name__)
MIB = 1024**2
TELEMETRY_COLUMNS = (
    "timestamp",
    "elapsed_seconds",
    "system_cpu_percent",
    "system_ram_percent",
    "available_ram_mib",
    "tree_cpu_seconds",
    "tree_rss_sum_mib",
    "active_workers",
    "ready_tasks",
    "blocked_tasks",
    "memory_paused",
)


class CampaignMonitor:
    def __init__(
        self, path: Path, interval: float = 1.0, worker_limit: int = 1
    ) -> None:
        self.path, self.interval = path, interval
        self.worker_limit = worker_limit
        self.process = psutil.Process(os.getpid())
        self.previous = {child.pid for child in self.process.children(recursive=True)}
        self.cpu_by_pid: dict[int, float] = {}
        cpu = self.process.cpu_times()
        self.cpu_baseline = cpu.user + cpu.system
        self.counts = (0, 0, 0, False)
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.start = perf_counter()
        self.errors: list[str] = []

    def update(self, counts: tuple[int, int, int, bool]) -> None:
        self.counts = counts

    def sample(self) -> None:
        try:
            processes = [
                self.process,
                *(
                    child
                    for child in self.process.children(recursive=True)
                    if child.pid not in self.previous
                ),
            ]
            rss = 0.0
            for process in processes:
                try:
                    cpu = process.cpu_times()
                    self.cpu_by_pid[process.pid] = cpu.user + cpu.system
                    rss += process.memory_info().rss / MIB
                except psutil.NoSuchProcess:
                    continue
            memory = psutil.virtual_memory()
            row = (
                time(),
                perf_counter() - self.start,
                psutil.cpu_percent(),
                memory.percent,
                memory.available / MIB,
                max(0.0, sum(self.cpu_by_pid.values()) - self.cpu_baseline),
                rss,
                *self.counts,
            )
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(row)
        except (psutil.Error, OSError) as error:
            logger.warning("Telemetría de campaña incompleta: %s", error)
            self.errors.append(str(error))

    def poll(self) -> None:
        psutil.cpu_percent()
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self) -> "CampaignMonitor":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(TELEMETRY_COLUMNS)
        psutil.cpu_percent()
        self.start = perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(
        self, *exception: type[BaseException] | BaseException | TracebackType | None
    ) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        self.sample()
