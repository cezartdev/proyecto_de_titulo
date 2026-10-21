"""Piloto CPU/CUDA sobre predictores guardados, sin modificar el estudio de origen."""

import argparse
import gc
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.datos.procesados import JsonValue, load_design_split
from truss_research.evaluacion.dispositivos import device_context
from truss_research.evaluacion.recursos import timed
from truss_research.surrogates.experimentos import load_predictor, write_json
from truss_research.surrogates.explicabilidad import ExplanationConfig, explain_model

DEFAULT_PILOT_ROOT = Path("reports/piloto_tabpfn_gpu_001")
PILOT_BATCHES = (1, 32, 64)
PILOT_REPEATS = 3
PREDICTION_RTOL = 1e-4
PREDICTION_ATOL = 1e-6


def predictor_directory(study: Path, key: str) -> Path:
    state = json.loads((study / "study.json").read_text(encoding="utf-8"))
    task = next(row for row in state["tasks"] if row["key"] == key)
    if task["status"] != "complete":
        raise ValueError("El piloto requiere un predictor completo")
    directory = Path(task["directory"])
    return directory if directory.is_absolute() else study / directory


def run_device(config: ExplanationConfig) -> dict[str, JsonValue]:
    torch.set_num_threads(1)
    split = load_design_split(config.data_dir)
    features = split.validation.loc[:, list(INPUT_COLUMNS)]
    model, loading = timed(lambda: load_predictor(config.predictor_dir, config.device))
    reference, first_prediction = timed(lambda: model.predict(features))
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        config.output_dir.with_suffix(".predictions.npy"), reference, allow_pickle=False
    )
    samples = []
    for size in PILOT_BATCHES:
        model.predict(features.iloc[:size])
        for repeat in range(PILOT_REPEATS):
            _, timing = timed(lambda: model.predict(features.iloc[:size]))
            samples.append({"batch_size": size, "repeat": repeat, **timing})
    hardware = (
        device_context()
        if config.device == "cuda"
        else {"device": "cpu", "torch_version": torch.__version__}
    )
    model.steps.clear()
    gc.collect()
    if config.device == "cuda":
        torch.cuda.empty_cache()
    _, explanation = timed(lambda: explain_model(config))
    return {
        "config": asdict(config),
        "hardware": hardware,
        "load": loading,
        "first_prediction": first_prediction,
        "inference": samples,
        "explanation": explanation,
        "status": "complete",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estudio", type=Path, default=Path("artifacts/estudio"))
    parser.add_argument("--destino", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--modelo", choices=("tabpfn", "targetwise"), default="tabpfn")
    parser.add_argument(
        "--devices", nargs="+", choices=("cpu", "cuda"), default=["cuda", "cpu"]
    )
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    source = predictor_directory(args.estudio, f"predictor/{args.modelo}/42")
    for device in args.devices:
        config = ExplanationConfig(
            Path("data/processed_designs"),
            source,
            args.destino / f"{args.modelo}_{device}",
            device=device,
            evaluation_limit=args.rows,
        )
        receipt = config.output_dir.with_suffix(".pilot.json")
        if receipt.exists():
            continue
        write_json(receipt, run_device(config))
    paths = [
        args.destino / f"{args.modelo}_{device}.predictions.npy"
        for device in ("cpu", "cuda")
    ]
    if all(path.exists() for path in paths):
        cpu, gpu = (np.load(path, allow_pickle=False) for path in paths)
        write_json(
            args.destino / f"{args.modelo}_equivalence.json",
            {
                "rtol": PREDICTION_RTOL,
                "atol": PREDICTION_ATOL,
                "equivalent": bool(
                    np.allclose(cpu, gpu, rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL)
                ),
                "max_absolute_error_by_target": np.max(
                    np.abs(cpu - gpu), axis=0
                ).tolist(),
            },
        )


if __name__ == "__main__":
    main()
