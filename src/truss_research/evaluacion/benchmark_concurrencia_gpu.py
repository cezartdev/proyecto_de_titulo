"""Rendimiento de cuatro explicaciones idénticas con uno o dos procesos GPU."""

import argparse
import logging
import os
from pathlib import Path
from time import perf_counter

import torch

from truss_research.evaluacion.benchmark_tabpfn import predictor_directory
from truss_research.evaluacion.dispositivos import device_context
from truss_research.optimizacion.ejecucion_cpu import ExecutionParameters, cpu_pool
from truss_research.surrogates.experimentos import write_json
from truss_research.surrogates.explicabilidad import ExplanationConfig, explain_model

DEFAULT_DESTINATION = Path("reports/piloto_concurrencia_gpu_001")
PILOT_JOBS = 4
PILOT_WORKERS = (1, 2)


def warm_gpu_worker(unused: int) -> int:
    torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    return os.getpid()


def explanation_job(config: ExplanationConfig) -> dict[str, str | float | int | None]:
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    explain_model(config)
    return {"pid": os.getpid(), "seconds": perf_counter() - started, **device_context()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estudio", type=Path, default=Path("artifacts/estudio"))
    parser.add_argument("--destino", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.destino.mkdir(parents=True, exist_ok=False)
    source = predictor_directory(args.estudio, "predictor/tabpfn/42")
    results = []
    for workers in PILOT_WORKERS:
        parameters = ExecutionParameters(cpu_workers=6, gpu_workers=workers)
        with cpu_pool(parameters, "cuda") as executor:
            warmups = [
                executor.submit(warm_gpu_worker, number) for number in range(workers)
            ]
            for future in warmups:
                future.result()
            started = perf_counter()
            jobs = [
                ExplanationConfig(
                    Path("data/processed_designs"),
                    source,
                    args.destino / f"workers_{workers}" / f"job_{number}",
                    device="cuda",
                    evaluation_limit=1,
                )
                for number in range(PILOT_JOBS)
            ]
            futures = [executor.submit(explanation_job, job) for job in jobs]
            measurements = [future.result() for future in futures]
            results.append(
                {
                    "gpu_workers": workers,
                    "jobs": PILOT_JOBS,
                    "wall_seconds": perf_counter() - started,
                    "measurements": measurements,
                }
            )
            write_json(
                args.destino / "comparison.json",
                {"status": "running", "profiles": results},
            )
    write_json(
        args.destino / "comparison.json", {"status": "complete", "profiles": results}
    )


if __name__ == "__main__":
    main()
