"""Entrenamiento aislado y medido de una familia, sin iniciar una campaña."""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import torch

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.procesados import JsonValue
from truss_research.datos.schemas import DESIGN_DESTINATION
from truss_research.evaluacion.costos import machine_context
from truss_research.optimizacion.contratos import StudyTask
from truss_research.optimizacion.entrenamiento import train_policy
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    environment_versions,
    source_signature,
)
from truss_research.optimizacion.politica import configured_policy
from truss_research.optimizacion.protocolo import load_protocol
from truss_research.optimizacion.trabajos_predictivos import (
    JobResources,
    run_measured_job,
)
from truss_research.optimizacion.variantes_rl import VARIANT_REGISTRY, PolicyVariant
from truss_research.surrogates.configuracion import (
    MODEL_NAMES,
    ModelName,
    PredictorConfig,
)
from truss_research.surrogates.experimentos import (
    data_signature,
    read_run,
    train_predictor,
    write_json,
)
from truss_research.surrogates.poda import PruningParameters
from truss_research.surrogates.tabpfn import verify_tabpfn

TRAINING_ROOT = Path("artifacts/entrenamientos_individuales")
EVOLUTION_POLICIES = (
    "evolution_rl",
    "evolution_rl_assisted",
    "evolution_contextual_bandit",
    "evolution_contextual_bandit_assisted",
)
type TrainingModel = (
    ModelName
    | PolicyVariant
    | Literal[
        "evolution_rl",
        "evolution_rl_assisted",
        "evolution_contextual_bandit",
        "evolution_contextual_bandit_assisted",
    ]
)
TRAINING_MODELS = (*MODEL_NAMES, *VARIANT_REGISTRY, *EVOLUTION_POLICIES)


@dataclass(frozen=True)
class TrainingRequest:
    model: TrainingModel
    protocol: Path = Path("config/study.json")
    data_dir: Path = DESIGN_DESTINATION
    destination: Path | None = None
    predictor_dir: Path | None = None
    seed: int = 42
    accelerator: str = "cpu"
    mode: Literal["protocol", "fit"] = "protocol"
    pruning: bool | None = None
    early_stopping_rounds: int | None = None
    cheap_predictor_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.pruning is not None and type(self.pruning) is not bool:
            raise ValueError("La opción de poda debe ser booleana")
        if self.early_stopping_rounds is not None and (
            type(self.early_stopping_rounds) is not int
            or self.early_stopping_rounds < 0
        ):
            raise ValueError("La paciencia de boosting debe ser un entero no negativo")
        if self.model not in TRAINING_MODELS:
            raise ValueError("Familia o política desconocida")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("La semilla debe ser un entero no negativo")
        if self.accelerator not in ("cpu", "cuda"):
            raise ValueError("El acelerador debe ser cpu o cuda")
        if self.mode not in ("protocol", "fit"):
            raise ValueError("El modo debe ser protocol o fit")
        if self.mode == "fit" and (
            self.model == "targetwise"
            or self.model in (*VARIANT_REGISTRY, *EVOLUTION_POLICIES)
        ):
            raise ValueError(
                "El modo fit admite predictores fijos; targetwise y RL requieren protocol"
            )
        if (
            self.model in (*VARIANT_REGISTRY, *EVOLUTION_POLICIES)
            and self.predictor_dir is None
        ):
            raise ValueError("RL requiere --predictor con un predictor ya entrenado")
        if self.model in MODEL_NAMES and self.predictor_dir is not None:
            raise ValueError("--predictor solo se utiliza al entrenar políticas RL")
        if (
            self.model in EVOLUTION_POLICIES
            and self.model.endswith("_assisted")
            and self.cheap_predictor_dir is None
        ):
            raise ValueError("La política asistida requiere --cheap-predictor")


def next_training_destination(model: str) -> Path:
    numbers = [
        int(path.name.removeprefix(model + "_"))
        for path in TRAINING_ROOT.glob(model + "_*")
        if path.name.removeprefix(model + "_").isdigit()
    ]
    return TRAINING_ROOT / f"{model}_{max(numbers, default=0) + 1:03d}"


def training_config(request: TrainingRequest) -> StudyConfig:
    label = request.model if request.mode == "protocol" else f"{request.model}_fit"
    destination = request.destination or next_training_destination(label)
    config = load_protocol(
        request.protocol,
        (destination.resolve(), request.data_dir.resolve()),
        request.accelerator,
    )
    if request.pruning is not None:
        config = replace(
            config, pruning=replace(config.pruning, enabled=request.pruning)
        )
    if request.early_stopping_rounds is not None:
        config = replace(
            config,
            parameters=replace(
                config.parameters, early_stopping_rounds=request.early_stopping_rounds
            ),
        )
    return config


def training_plan(
    request: TrainingRequest, config: StudyConfig
) -> dict[str, JsonValue]:
    policy = request.model in (*VARIANT_REGISTRY, *EVOLUTION_POLICIES)
    uses_tabpfn = request.model == "tabpfn" or (
        request.model == "targetwise" and config.tabpfn_enabled
    )
    devices = {
        "tabpfn": config.tabpfn.device,
        "targetwise": "cpu+cuda"
        if uses_tabpfn and config.tabpfn.device == "cuda"
        else "cpu",
    }
    return {
        "mode": request.mode,
        "pruning": asdict(config.pruning),
        "boosting_early_stopping_rounds": config.parameters.early_stopping_rounds
        if request.mode == "protocol"
        else 0,
        "model": request.model,
        "seed": request.seed,
        "protocol": str(request.protocol.resolve()),
        "destination": str(config.output_dir),
        "kind": "policy" if policy else "predictor",
        "training_device": devices.get(request.model, "cpu"),
        "device_note": "Solo componentes TabPFN usan CUDA; RL y demás familias usan CPU",
        "predictor": str(request.predictor_dir.resolve())
        if request.predictor_dir
        else None,
        "trials": 0
        if request.mode == "fit" or request.model in ("tabpfn", "dummy") or policy
        else config.trials,
        "outer_folds": None if policy or request.mode == "fit" else config.folds,
        "inner_folds": None if policy or request.mode == "fit" else config.nested_folds,
        "fit_calls": 1 if request.mode == "fit" else None,
        "transitions": config.rl_steps if policy else None,
        "validation_interval": config.validation_interval if policy else None,
        "test_used": False,
        "shap": False,
        "single_training_process": True,
        "cheap_predictor_dir": str(request.cheap_predictor_dir)
        if request.cheap_predictor_dir
        else None,
        "experiment": config.experiment,
    }


def execute_training(request: TrainingRequest, config: StudyConfig) -> None:
    destination = config.output_dir / "run"
    if request.model in EVOLUTION_POLICIES:
        from truss_research.evaluacion.busqueda_evolutiva import (
            build_context,
            policy_configuration,
        )
        from truss_research.optimizacion.entrenamiento_propuestas import (
            train_proposal_policy,
        )

        assert request.predictor_dir is not None
        cheap = request.cheap_predictor_dir or request.predictor_dir
        if (
            request.model.endswith("_assisted")
            and cheap.resolve() == request.predictor_dir.resolve()
        ):
            raise ValueError("El predictor asistente debe ser distinto del evaluador")
        if read_run(cheap)["dataset_signature"] != data_signature(config.data_dir):
            raise ValueError("El predictor barato pertenece a otros datos")
        context = build_context(config, (request.predictor_dir, cheap))
        assisted = request.model.endswith("_assisted")
        strategy_name = request.model.removeprefix("evolution_")
        signature = hashlib.sha256(
            json.dumps(
                {
                    "config": asdict(config),
                    "source": source_signature(),
                    "predictor": read_run(request.predictor_dir),
                    "cheap": read_run(cheap),
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        train_proposal_policy(
            replace(context, assisted=assisted),
            policy_configuration(config, (signature, request.seed, strategy_name)),
            destination,
        )
        return
    if request.model in VARIANT_REGISTRY:
        assert request.predictor_dir is not None
        policy = configured_policy(
            config,
            (request.predictor_dir.resolve(), destination, None),
            StudyTask(
                f"policy/{request.model}/{request.seed}",
                "policy",
                seed=request.seed,
                variant=cast(PolicyVariant, request.model),
            ),
        )
        train_policy(policy)
        return
    predictor = PredictorConfig(
        config.data_dir,
        destination,
        model=cast(ModelName, request.model),
        seed=request.seed,
        folds=config.folds,
        trials=0
        if request.mode == "fit" or request.model == "tabpfn"
        else config.trials,
        tabpfn=config.tabpfn,
        pruning=PruningParameters() if request.mode == "fit" else config.pruning,
        device="cpu",
        parameters=config.parameters,
        cv_seed=config.cv_seed,
        nested_folds=None if request.mode == "fit" else config.nested_folds,
        final_tuning_folds=config.final_tuning_folds,
        candidate_models=tuple(name for name in config.models if name != "targetwise"),
        study_path=config.output_dir / "search.sqlite3",
    )
    if request.mode == "fit":
        from truss_research.surrogates.ajuste_directo import fit_diagnostic

        fit_diagnostic(predictor)
        return
    train_predictor(predictor)


def run_individual_training(request: TrainingRequest) -> Path:
    config = training_config(request)
    if config.output_dir.exists():
        raise ValueError(
            "El destino ya existe; elegir otro o dejar numeración automática"
        )
    plan = training_plan(request, config)
    signature = data_signature(config.data_dir)
    if request.predictor_dir is not None:
        metadata = read_run(request.predictor_dir)
        if metadata.get("dataset_signature") != signature:
            raise ValueError("El predictor no corresponde a los datos solicitados")
        if not (request.predictor_dir / "model.joblib").is_file():
            raise ValueError("Falta model.joblib en el directorio predictor")
    if request.model == "tabpfn" or (
        request.model == "targetwise" and config.tabpfn_enabled
    ):
        verify_tabpfn(config.tabpfn)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, JsonValue] = {
        **plan,
        "status": "running",
        "source_signature": source_signature(),
        "dataset_signature": signature,
        "versions": environment_versions(),
        "machine": machine_context(),
        "study_config": asdict(config),
    }
    write_json(config.output_dir / "summary.json", summary)
    torch.set_num_threads(1)
    try:
        outcome = run_measured_job(
            JobResources(
                request.model,
                config.output_dir / "run",
                config.data_dir,
                config.measurement,
            ),
            lambda: execute_training(request, config),
        )
        summary.update(
            asdict(outcome), status="complete" if outcome.success else "failed"
        )
        if not outcome.success:
            raise RuntimeError(outcome.error)
        if source_signature() != summary["source_signature"]:
            raise ValueError("El código cambió durante el entrenamiento individual")
        resources = read_artifact(config.output_dir / "run.resources.json")
        pd.DataFrame(cast(list[dict[str, JsonValue]], resources["phases"])).to_csv(
            config.output_dir / "phase_times.csv", index=False
        )
        summary["pruning_results"] = {
            path.relative_to(config.output_dir).as_posix(): read_artifact(path)
            for path in config.output_dir.rglob("*.pruning.json")
        }
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(config.output_dir / "summary.json", summary)
    return config.output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", required=True, choices=TRAINING_MODELS)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("config/study.json"))
    parser.add_argument("--datos", type=Path, default=DESIGN_DESTINATION)
    parser.add_argument("--destino", type=Path)
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--cheap-predictor", type=Path)
    parser.add_argument("--accelerator", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("protocol", "fit"), default="protocol")
    parser.add_argument("--pruning", type=int, choices=(0, 1))
    parser.add_argument("--early-stopping-rounds", type=int)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    request = TrainingRequest(
        cast(TrainingModel, args.modelo),
        args.config,
        args.datos,
        args.destino,
        args.predictor,
        args.semilla,
        args.accelerator,
        args.mode,
        pruning=None if args.pruning is None else bool(args.pruning),
        early_stopping_rounds=args.early_stopping_rounds,
        cheap_predictor_dir=args.cheap_predictor,
    )
    if args.plan:
        print(
            json.dumps(
                training_plan(request, training_config(request)),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(run_individual_training(request))


if __name__ == "__main__":
    main()
