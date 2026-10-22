"""Etapa SHAP manual y reanudable sobre predictores firmados de validación."""

import argparse
import gc
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import torch
from threadpoolctl import threadpool_limits

from truss_research import DEFAULT_STUDY
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue, manifest_protocol, read_manifest
from truss_research.evaluacion.busqueda_evolutiva import (
    predictor_records,
    verify_evaluation,
)
from truss_research.evaluacion.recursos import ResourceMonitor
from truss_research.evaluacion.registro import results_root
from truss_research.optimizacion.estudio_estado import (
    artifact_signatures,
    load_study_config,
    study_lock,
    verify_task,
)
from truss_research.surrogates.experimentos import data_signature, read_run, write_json
from truss_research.surrogates.explicabilidad import ExplanationConfig, explain_model
from truss_research.surrogates.persistencia import model_signatures

DEFAULT_CONFIG = Path("config/explain.json")

IMPLEMENTATION_FILES = (
    Path(__file__),
    Path(__file__).with_name("informe_explicabilidad.py"),
    Path(__file__).parents[1] / "surrogates/explicabilidad.py",
    Path(__file__).parents[1] / "surrogates/atribuciones.py",
    Path(__file__).parents[1] / "surrogates/permutaciones.py",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PilotParameters:
    background_size: int
    permutation_rounds: int
    evaluation_limit: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("Los parámetros del piloto deben ser enteros positivos")


@dataclass(frozen=True)
class ExplanationParameters:
    background_size: int
    permutation_rounds: int
    batch_size: int
    winner_seeds: bool
    pilot: PilotParameters

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.background_size,
                self.permutation_rounds,
                self.batch_size,
            )
        ):
            raise ValueError("Referencia, rondas y lotes deben ser enteros positivos")
        if type(self.winner_seeds) is not bool:
            raise ValueError("winner_seeds debe ser booleano")


@dataclass(frozen=True)
class ExplanationRequest:
    study: Path
    destination: Path | None = None
    config: Path = DEFAULT_CONFIG
    mode: Literal["full", "pilot"] = "full"
    accelerator: Literal["cpu", "cuda"] = "cuda"

    def __post_init__(self) -> None:
        if self.mode not in ("full", "pilot") or self.accelerator not in (
            "cpu",
            "cuda",
        ):
            raise ValueError("Modo o acelerador SHAP desconocido")


def read_parameters(path: Path) -> ExplanationParameters:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("pilot"), dict):
        raise ValueError("La configuración SHAP requiere un objeto pilot")
    try:
        return ExplanationParameters(
            **(raw | {"pilot": PilotParameters(**raw["pilot"])})
        )
    except TypeError as error:
        raise ValueError(
            "Configuración SHAP incompleta o con campos desconocidos"
        ) from error


def selected_families(study: Path) -> list[str]:
    path = study / "freeze.json"
    path = path if path.is_file() else study / "evolution/search.json"
    selected = read_artifact(path).get("selected")
    if (
        not isinstance(selected, list)
        or len(selected) != 3
        or not all(isinstance(name, str) for name in selected)
    ):
        raise ValueError(
            "Completar la selección de validación con make study antes de SHAP"
        )
    return cast(list[str], selected)


def explanation_plan(request: ExplanationRequest) -> dict[str, JsonValue]:
    """Describe el alcance sin cargar modelos, ejecutar SHAP ni crear directorios."""
    config = load_study_config(request.study)
    if config.experiment != "evolution":
        raise ValueError("SHAP requiere el protocolo evolution vigente")
    if manifest_protocol(read_manifest(config.data_dir)) != "first_appearance":
        raise ValueError("SHAP vigente requiere el dataset de primera aparición")
    parameters = read_parameters(request.config)
    selected = selected_families(request.study)
    destination = request.destination or results_root(request.study) / (
        "explicabilidad_001" if request.mode == "full" else "explicabilidad_piloto_001"
    )
    references = [
        row
        for row in predictor_records(request.study)
        if row.seed == config.representative_seed
        or (parameters.winner_seeds and row.model == selected[0])
    ]
    expected = {
        (family, seed)
        for family in config.models
        for seed in config.seeds
        if seed == config.representative_seed
        or (parameters.winner_seeds and family == selected[0])
    }
    if {(row.model, row.seed) for row in references} != expected:
        raise ValueError(
            "Faltan predictores representativos para explicar todas las familias"
        )
    jobs = [
        ExplanationConfig(
            config.data_dir,
            Path(str(row.record.directory)),
            destination / f"{row.model}_{row.seed}",
            seed=config.cv_seed,
            background_size=parameters.pilot.background_size
            if request.mode == "pilot"
            else parameters.background_size,
            permutation_rounds=parameters.pilot.permutation_rounds
            if request.mode == "pilot"
            else parameters.permutation_rounds,
            device=request.accelerator
            if row.model in ("tabpfn", "targetwise")
            else "cpu",
            evaluation_limit=parameters.pilot.evaluation_limit
            if request.mode == "pilot"
            else None,
            batch_size=parameters.batch_size,
        )
        for row in references
    ]
    return {
        "study": str(request.study.resolve()),
        "destination": str(destination.resolve()),
        "mode": request.mode,
        "selected": selected,
        "parameters": asdict(parameters),
        "jobs": [
            asdict(job) | {"model": row.model, "model_seed": row.seed}
            for row, job in zip(references, jobs, strict=True)
        ],
        "scope": "validation_only_training_background_no_refit_no_selection",
    }


def job_config(raw: dict[str, JsonValue]) -> ExplanationConfig:
    values = {
        key: value for key, value in raw.items() if key not in ("model", "model_seed")
    }
    values.update(
        {
            key: Path(str(values[key]))
            for key in ("data_dir", "predictor_dir", "output_dir")
        }
    )
    return ExplanationConfig(**values)


def implementation_signatures() -> dict[str, str]:
    return {
        f"{path.parent.name}/{path.name}": source_digest(path)
        for path in IMPLEMENTATION_FILES
    }


def stage_signatures(directory: Path) -> dict[str, str]:
    """El bloqueo operativo puede estar abierto en Windows y no es una salida científica."""
    return {
        path.relative_to(directory).as_posix(): source_digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.name != ".study.lock"
        and path != directory / "complete.json"
    }


def plan_identity(plan: dict[str, JsonValue]) -> str:
    jobs = cast(list[dict[str, JsonValue]], plan["jobs"])
    identity = {
        "plan": plan,
        "dataset": data_signature(job_config(jobs[0]).data_dir),
        "models": {
            str(row["model"]) + "/" + str(row["model_seed"]): model_signatures(
                job_config(row).predictor_dir
            )
            for row in jobs
        },
        "implementation": implementation_signatures(),
        "versions": {
            name: version(name)
            for name in ("shap", "numpy", "scikit-learn", "torch", "tabpfn")
        },
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()


def check_output_location(study: Path, destination: Path) -> None:
    """Mantiene los derivados SHAP fuera de los artefactos y datos científicos."""
    config = load_study_config(study)
    target = destination.resolve()
    protected = (study.resolve(), config.data_dir.resolve())
    if any(
        target == path or target.is_relative_to(path) or path.is_relative_to(target)
        for path in protected
    ):
        raise ValueError("El destino SHAP debe estar fuera del estudio y sus datos")


def verified_files(
    directory: Path, receipt: str = "complete.json"
) -> dict[str, JsonValue]:
    metadata = read_artifact(directory / receipt)
    files = metadata.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("La etapa SHAP no declara salidas completas")
    if any(
        not (directory / name).is_file() or source_digest(directory / name) != digest
        for name, digest in files.items()
    ):
        raise ValueError("Cambió una salida SHAP completa")
    return metadata


def execute_explanation(raw: dict[str, JsonValue]) -> None:
    """Reutiliza resultados íntegros o continúa checkpoints de una tarea incompleta."""
    config = job_config(raw)
    receipt = config.output_dir / "complete.json"
    if receipt.is_file():
        verified_files(config.output_dir)
        return
    if config.output_dir.exists():
        raise ValueError(
            "Hay una exportación SHAP parcial sin recibo; conservarla y elegir otro destino"
        )
    attempts = config.output_dir.with_suffix(".attempts")
    attempts.mkdir(parents=True, exist_ok=True)
    number = len(list(attempts.glob("attempt_*.json"))) + 1
    started = perf_counter()
    monitor = ResourceMonitor()
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    record: dict[str, JsonValue] = {
        "status": "running",
        "model": raw["model"],
        "seed": raw["model_seed"],
    }
    try:
        with monitor, threadpool_limits(limits=1):
            explain_model(config)
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(
            resources=monitor.snapshot(), wall_seconds=perf_counter() - started
        )
        write_json(attempts / f"attempt_{number:03d}.json", record)
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    write_json(receipt, {"files": artifact_signatures(config.output_dir)})


def pending_cuda(jobs: list[dict[str, JsonValue]]) -> bool:
    """Solo reserva CUDA cuando un trabajo pendiente contiene estados TabPFN."""
    return any(
        job_config(row).device == "cuda"
        and any(
            name.endswith(".tabpfn_fit")
            for name in read_run(job_config(row).predictor_dir).get(
                "model_artifacts", {}
            )
        )
        for row in jobs
        if not (job_config(row).output_dir / "complete.json").is_file()
    )


def run_explanations(request: ExplanationRequest) -> Path:
    from truss_research.evaluacion.informe_explicabilidad import aggregate_explanations

    plan = explanation_plan(request)
    destination = Path(str(plan["destination"]))
    check_output_location(request.study, destination)
    verify_evaluation(request.study / "evolution/validation")
    jobs = cast(list[dict[str, JsonValue]], plan["jobs"])
    references = {
        f"{row.model}/{row.seed}": row for row in predictor_records(request.study)
    }
    for row in jobs:
        metadata = read_run(
            verify_task(references[f"{row['model']}/{row['model_seed']}"].record)
        )
        if (
            metadata["config"]["model"] != row["model"]
            or metadata["config"]["seed"] != row["model_seed"]
        ):
            raise ValueError(
                "La identidad del predictor no corresponde a su tarea SHAP"
            )
    identity = plan_identity(plan)
    with study_lock(destination):
        path = destination / "request.json"
        if not path.is_file() and any(
            item.name != ".study.lock" for item in destination.iterdir()
        ):
            raise ValueError(
                "El destino contiene archivos ajenos a esta etapa SHAP; elegir otro EXPLAIN_DEST"
            )
        if path.is_file() and read_artifact(path)["identity"] != identity:
            raise ValueError(
                "Cambió el protocolo, código, datos o modelos SHAP; elegir otro EXPLAIN_DEST"
            )
        if (destination / "complete.json").is_file():
            verified_files(destination)
            return destination
        if (
            request.accelerator == "cuda"
            and pending_cuda(jobs)
            and not torch.cuda.is_available()
        ):
            raise ValueError(
                "SHAP solicita CUDA para TabPFN pero no está disponible; usar ACCELERATOR=cpu explícitamente"
            )
        write_json(path, {"identity": identity, "plan": plan})
        for row in jobs:
            execute_explanation(row)
        if identity != plan_identity(explanation_plan(request)):
            raise ValueError("Cambió el origen durante SHAP; no se firma la etapa")
        aggregate_explanations(plan, destination)
        write_json(
            destination / "complete.json",
            {
                "identity": identity,
                "study": str(request.study.resolve()),
                "mode": request.mode,
                "selected": plan["selected"],
                "dataset_signature": data_signature(job_config(jobs[0]).data_dir),
                "implementation": implementation_signatures(),
                "predictors": [
                    {
                        "model": row["model"],
                        "seed": row["model_seed"],
                        "directory": str(job_config(row).predictor_dir),
                        "files": model_signatures(job_config(row).predictor_dir),
                    }
                    for row in jobs
                ],
                "files": stage_signatures(destination),
            },
        )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estudio", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--destino", type=Path)
    parser.add_argument("--mode", choices=("full", "pilot"), default="full")
    parser.add_argument("--accelerator", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    request = ExplanationRequest(
        args.estudio, args.destino, args.config, args.mode, args.accelerator
    )
    if args.plan:
        print(
            json.dumps(
                explanation_plan(request), default=str, ensure_ascii=False, indent=2
            )
        )
        return
    print(run_explanations(request))


if __name__ == "__main__":
    main()
