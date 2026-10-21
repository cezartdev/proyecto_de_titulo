"""Comparación completa CPU/CUDA de RL sobre entrenamiento y validación existentes."""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import torch

from truss_research.evaluacion.benchmark_tabpfn import predictor_directory
from truss_research.evaluacion.dispositivos import device_context
from truss_research.evaluacion.recursos import timed
from truss_research.optimizacion.contratos import StudyTask
from truss_research.optimizacion.entrenamiento import train_policy
from truss_research.optimizacion.politica import configured_policy
from truss_research.optimizacion.protocolo import load_protocol
from truss_research.surrogates.experimentos import read_run, write_json

DEFAULT_PILOT = Path("artifacts/piloto_politicas_gpu_001")
PILOT_TRANSITIONS = 5000
PILOT_VARIANTS = ("rl", "rl_dueling", "rl_ppo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estudio", type=Path, default=Path("artifacts/estudio"))
    parser.add_argument("--destino", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--steps", type=int, default=PILOT_TRANSITIONS)
    parser.add_argument(
        "--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"]
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = load_protocol(
        Path("config/study.json"), (args.destino, Path("data/processed_designs"))
    )
    for variant in PILOT_VARIANTS:
        old = predictor_directory(args.estudio, f"policy/{variant}/42")
        predictor = Path(read_run(old)["config"]["predictor_dir"])
        for device in args.devices:
            destination = args.destino / f"{variant}_{device}"
            if (destination / "pilot.json").exists():
                continue
            policy = configured_policy(
                config,
                (predictor, destination, None),
                StudyTask(f"policy/{variant}/42", "policy", variant=variant),
            )
            policy = replace(
                policy,
                device=device,
                parameters=replace(
                    policy.parameters,
                    transitions=args.steps,
                    validation_interval=args.steps,
                ),
            )
            _, timing = timed(lambda: train_policy(policy))
            write_json(
                destination / "pilot.json",
                {
                    "device": device,
                    "hardware": device_context(),
                    "transitions": args.steps,
                    "timing": timing,
                    "status": "complete",
                },
            )
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
