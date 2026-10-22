"""Comparación CPU/CUDA aislada; no utiliza datos ni artefactos de la investigación."""

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.informe_costos import markdown_table
from truss_research.evaluacion.pruebas_neuronales import (
    NeuralBenchParameters,
    NeuralCase,
    dqn_samples,
    mlp_samples,
)
from truss_research.optimizacion.estudio_estado import source_signature
from truss_research.surrogates.experimentos import write_json

DEFAULT_EXPORTS = Path("reports/pruebas_gpu")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeuralProtocol:
    parameters: NeuralBenchParameters
    mlp_cases: tuple[NeuralCase, ...]
    dqn_cases: tuple[NeuralCase, ...]


DEFAULT_NEURAL_PROTOCOL = NeuralProtocol(
    parameters=NeuralBenchParameters(),
    mlp_cases=(
        NeuralCase(name="study", rows=1018, width=64, batch=64),
        NeuralCase(name="search_width128", rows=1018, width=128, batch=64),
    ),
    dqn_cases=(
        NeuralCase(name="study", catalog_size=32, width=64, batch=64),
        NeuralCase(name="larger_batch", catalog_size=32, width=64, batch=256),
    ),
)


def load_neural_protocol(path: Path | None = None) -> NeuralProtocol:
    if path is None:
        return DEFAULT_NEURAL_PROTOCOL
    payload = read_artifact(path)
    allowed = {field.name for field in fields(NeuralBenchParameters)}
    if set(payload) - (allowed | {"mlp_cases", "dqn_cases"}):
        raise ValueError("Campos desconocidos en el protocolo GPU")
    settings = {key: value for key, value in payload.items() if key in allowed}
    for name in ("seeds", "inference_batches"):
        if name in settings:
            if not isinstance(settings[name], list):
                raise ValueError(f"{name} requiere una lista JSON")
            settings[name] = tuple(settings[name])
    parameters = NeuralBenchParameters(**settings)
    cases: dict[str, tuple[NeuralCase, ...]] = {}
    for name in ("mlp_cases", "dqn_cases"):
        entries = payload.get(name)
        if (
            not isinstance(entries, list)
            or not entries
            or any(not isinstance(entry, dict) for entry in entries)
        ):
            raise ValueError(f"{name} requiere casos JSON no vacíos")
        cases[name] = tuple(
            NeuralCase(**cast(dict[str, str | int], entry)) for entry in entries
        )
        if len({case.name for case in cases[name]}) != len(cases[name]):
            raise ValueError("Los nombres de casos deben ser únicos por red")
    if any(
        case.rows < max(parameters.inference_batches) for case in cases["mlp_cases"]
    ):
        raise ValueError("Los lotes de inferencia exceden las filas sintéticas")
    return NeuralProtocol(parameters, cases["mlp_cases"], cases["dqn_cases"])


def paired_comparison(samples: pd.DataFrame) -> pd.DataFrame:
    if (
        set(samples.device) != {"cpu", "cuda"}
        or not np.isfinite(samples.wall_seconds).all()
        or not samples.wall_seconds.gt(0).all()
    ):
        raise ValueError(
            "La comparación requiere CPU y CUDA con tiempos positivos y finitos"
        )
    keys = ["network", "case", "operation", "batch", "seed"]
    grouped = samples.groupby([*keys, "device"], as_index=False).wall_seconds.median()
    paired = grouped.pivot(
        index=keys, columns="device", values="wall_seconds"
    ).reset_index()
    if paired[["cpu", "cuda"]].isna().any().any():
        raise ValueError("Faltan pares CPU/CUDA para algunas semillas u operaciones")
    paired["cpu_over_gpu"] = paired.cpu / paired.cuda
    summary = paired.groupby(keys[:-1], as_index=False).agg(
        cpu_seconds=("cpu", "median"),
        gpu_seconds=("cuda", "median"),
        paired_speedup_median=("cpu_over_gpu", "median"),
        paired_speedup_q25=("cpu_over_gpu", lambda values: values.quantile(0.25)),
        paired_speedup_q75=("cpu_over_gpu", lambda values: values.quantile(0.75)),
        paired_seeds=("seed", "count"),
    )
    summary["faster"] = summary.paired_speedup_median.map(
        lambda ratio: "gpu" if ratio > 1 else "cpu"
    )
    return summary


def configure_cuda() -> float:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise ValueError(
            "PyTorch CUDA no está disponible. Ejecutar make gpu-test para utilizar el entorno CUDA aislado."
        )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = perf_counter()
    torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    return perf_counter() - start


def next_destination() -> Path:
    numbers = [
        int(path.name.removeprefix("benchmark_"))
        for path in DEFAULT_EXPORTS.glob("benchmark_*")
        if path.name.removeprefix("benchmark_").isdigit()
    ]
    return DEFAULT_EXPORTS / f"benchmark_{max(numbers, default=0) + 1:03d}"


def hardware_snapshot() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,temperature.gpu,utilization.gpu",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"No se pudo registrar nvidia-smi: {result.stderr}")
    return result.stdout.strip()


def export_neural_report(destination: Path, samples: pd.DataFrame) -> None:
    summary = paired_comparison(samples)
    summary.to_csv(destination / "comparison.csv", index=False)
    samples.loc[samples.operation.eq("fit_end_to_end")].to_csv(
        destination / "fit_quality.csv", index=False
    )
    sections = [
        "# Pruebas neuronales CPU y GPU",
        "Entradas sintéticas: no son resultados estructurales ni entrenamiento de la tesis. Mismas entradas, semillas, FP32 y versiones para CPU y CUDA; un hilo numérico CPU y GPU exclusiva para estas pruebas. El equipo puede tener aplicaciones gráficas residentes.",
        "cpu_over_gpu y paired_speedup_median mayores que 1 favorecen GPU. Se calcula primero la mediana por semilla y dispositivo y después el cociente pareado. Los cuartiles describen variación, no intervalos de confianza. Los tiempos de update son por actualización, calculados sobre una secuencia de pasos.",
        markdown_table(summary),
        "El MLP utiliza el regresor real: reserva interna, escalado de respuestas, Adam, parada temprana y restauración del mejor estado. fit_quality.csv conserva épocas ejecutadas, error sintético de entrenamiento y diferencias entre predicciones pareadas; una parada distinta cambia el trabajo realizado. La inferencia incluye transferencias y retorno a NumPy.",
        "DQN utiliza CandidateQNetwork, DQNLoss y update_learner reales sobre replay sintético. update_with_transfer incluye mover cada lote desde CPU; update_resident mantiene el lote en el dispositivo. Cada variante empieza con pesos y optimizador nuevos. choose_scores_with_transfer incluye ida del estado y retorno de puntuaciones. Se comprueba equivalencia de puntuaciones iniciales entre dispositivos.",
        "DQN no mide generación de episodios, hipervolumen, replay completo, validación ni checkpoints; estos resultados no acreditan aceleración del entrenamiento RL completo. larger_batch cambia el lote solo para explorar escalamiento y no modifica el protocolo de estudio. search_width128 corresponde a una anchura del espacio de búsqueda del MLP.",
        "Los tiempos CUDA sincronizan antes y después de cada operación. El arranque CUDA se registra aparte; importación, instalación y arranque de Python quedan fuera. gpu_peak_allocated_mib mide memoria de tensores administrada por PyTorch; no VRAM total de la tarjeta. CPU y GPU se ejecutan secuencialmente, alternando el orden por semilla, sin competir con el pool de entrenamiento.",
        "LightGBM permanece en CPU. La campaña principal conserva CPU para todas las familias y políticas; este comando no activa GPU ni cambia configuración o artefactos científicos.",
        "Referencias: [sincronización CUDA de PyTorch](https://docs.pytorch.org/docs/main/notes/cuda.html#asynchronous-execution) y [entornos aislados de uv](https://docs.astral.sh/uv/guides/scripts/).",
    ]
    (destination / "RESULTS.md").write_text(
        "\n\n".join(sections) + "\n", encoding="utf-8"
    )


def protocol_digest(protocol: NeuralProtocol, path: Path | None) -> str:
    if path is not None:
        return source_digest(path)
    return hashlib.sha256(
        json.dumps(asdict(protocol), sort_keys=True).encode("utf-8")
    ).hexdigest()


def run_neural_benchmark(protocol_path: Path | None, destination: Path) -> Path:
    if destination.exists():
        raise ValueError("Elegir un destino nuevo para las pruebas GPU")
    protocol = load_neural_protocol(protocol_path)
    startup = configure_cuda()
    before = hardware_snapshot()
    destination.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, JsonValue] = {
        "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_signature": source_signature(),
        "protocol_signature": protocol_digest(protocol, protocol_path),
        "protocol": json.loads(json.dumps(asdict(protocol))),
        "data": "synthetic_only",
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "numeric_threads": torch.get_num_threads(),
        "cold_cuda_start_seconds": startup,
        "versions": {
            name: version(name)
            for name in (
                "torch",
                "torchrl",
                "tensordict",
                "numpy",
                "pandas",
                "scikit-learn",
            )
        },
        "hardware_before": before,
        "deterministic_algorithms": True,
        "tf32": False,
        "lightgbm_device": "cpu",
        "campaign_device": "cpu",
    }
    write_json(destination / "provenance.json", metadata)
    rows: list[dict[str, JsonValue]] = []
    try:
        with threadpool_limits(limits=1):
            for function, cases in (
                (mlp_samples, protocol.mlp_cases),
                (dqn_samples, protocol.dqn_cases),
            ):
                for case in cases:
                    for seed in protocol.parameters.seeds:
                        rows.extend(function(protocol.parameters, case, seed))
                        pd.DataFrame(rows).to_csv(
                            destination / "samples.csv", index=False
                        )
        export_neural_report(destination, pd.DataFrame(rows))
        if (
            source_signature() != metadata["source_signature"]
            or protocol_digest(protocol, protocol_path)
            != metadata["protocol_signature"]
        ):
            raise ValueError(
                "El código o protocolo cambió durante la comparación CPU/CUDA"
            )
        metadata["status"] = "complete"
        metadata["hardware_after"] = hardware_snapshot()
        metadata["signatures"] = {
            path.name: source_digest(path)
            for path in destination.iterdir()
            if path.name != "provenance.json"
        }
    except BaseException as error:
        metadata["status"], metadata["error"] = "failed", str(error)
        raise
    finally:
        write_json(destination / "provenance.json", metadata)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pruebas sintéticas CPU/CUDA; LightGBM conserva CPU"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--destino", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run_neural_benchmark(args.config, args.destino or next_destination()))


if __name__ == "__main__":
    main()
