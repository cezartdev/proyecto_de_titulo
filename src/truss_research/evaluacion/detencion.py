"""Detiene únicamente árboles del recorrido pertenecientes a este repositorio."""

import logging
from pathlib import Path

import psutil

STOP_TIMEOUT_SECONDS = 5
WORKSPACE = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)


def stop_campaign() -> None:
    roots = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or []).replace("\\", "/")
            if (
                "evaluacion/recorrido.py" in command
                and Path(process.cwd()).resolve() == WORKSPACE
            ):
                roots.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
            logger.debug("Proceso no inspeccionable: %s", error)
    captured: dict[int, psutil.Process] = {}
    for root in roots:
        try:
            captured.update(
                {
                    process.pid: process
                    for process in [root, *root.children(recursive=True)]
                }
            )
        except psutil.NoSuchProcess:
            logger.info("El recorrido ya terminó")
    for process in reversed(list(captured.values())):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            logger.info("El proceso %s ya terminó", process.pid)
    _, alive = psutil.wait_procs(list(captured.values()), timeout=STOP_TIMEOUT_SECONDS)
    for process in alive:
        process.kill()
    _, remaining = psutil.wait_procs(alive, timeout=STOP_TIMEOUT_SECONDS)
    if remaining:
        raise RuntimeError(
            f"No se pudieron detener los procesos {[p.pid for p in remaining]}"
        )
    print(
        f"Recorrido detenido: {len(captured)} procesos; resultados completos conservados"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stop_campaign()
