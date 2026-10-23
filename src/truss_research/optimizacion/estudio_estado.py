"""Configuración congelada, bloqueo y registro atómico del estudio experimental."""

import hashlib
import json
import os
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from typing import Final, Literal, cast

from truss_research.cerchas.schemas import StructuralLimits
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.datos.schemas import DESIGN_DESTINATION
from truss_research.evaluacion.recursos import MeasurementParameters
from truss_research.evaluacion.telemetria_cpu import CampaignMonitor
from truss_research.optimizacion.configuracion_rl import (
    RLParameters,
    read_rl_parameters,
)
from truss_research.optimizacion.contratos import (
    AnalysisParameters,
    EvolutionParameters,
)
from truss_research.optimizacion.ejecucion_cpu import CpuPool, ExecutionParameters
from truss_research.optimizacion.estrategias_busqueda import (
    MIN_DIFFERENTIAL_POPULATION,
    SearchParameters,
    read_search_parameters,
)
from truss_research.optimizacion.propuestas import ProposalParameters
from truss_research.surrogates.configuracion import (
    EXPERIMENT_SEEDS,
    MODEL_NAMES,
    ModelName,
    ModelParameters,
)
from truss_research.surrogates.experimentos import (
    data_signature,
    write_json,
)
from truss_research.surrogates.poda import PruningParameters
from truss_research.surrogates.tabpfn import TabPFNParameters

type StudyPhase = Literal["predictores", "validacion", "final", "todo"]

STUDY_PHASES: Final[tuple[StudyPhase, ...]] = (
    "predictores",
    "validacion",
    "final",
    "todo",
)

STATE_FILE: Final[str] = "study.json"

STUDY_SCHEMA_VERSION: Final[int] = 20

DEPENDENCIES: Final[tuple[str, ...]] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "torchrl",
    "tensordict",
    "optuna",
    "shap",
    "pymoo",
    "catboost",
    "xgboost",
    "lightgbm",
    "joblib",
    "threadpoolctl",
    "psutil",
)


@dataclass(frozen=True)
class StudyConfig:
    output_dir: Path
    data_dir: Path = DESIGN_DESTINATION
    models: tuple[ModelName, ...] = MODEL_NAMES
    seeds: tuple[int, ...] = EXPERIMENT_SEEDS
    representative_seed: int = 42
    cv_seed: int = 42
    outer_folds: int = 5
    final_tuning_folds: int = 3
    trials: int = 30
    budgets: tuple[int, ...] = (32, 64, 96)
    rl_steps: int = 50_000
    rl_batch: int = 64
    rl_warmup: int = 1_000
    rl_hidden: int = 64
    validation_interval: int = 5_000
    parameters: ModelParameters = field(default_factory=ModelParameters)
    label: str = "posterior"
    inner_folds: int | None = None
    evolution: EvolutionParameters = field(default_factory=EvolutionParameters)
    analysis: AnalysisParameters = field(default_factory=AnalysisParameters)
    measurement: MeasurementParameters = field(default_factory=MeasurementParameters)
    execution: ExecutionParameters = field(default_factory=ExecutionParameters)
    rl: RLParameters = field(default_factory=RLParameters)
    calibration_required: bool = False
    calibration_fraction: float = 0.2
    tabpfn: TabPFNParameters = field(default_factory=TabPFNParameters)
    pruning: PruningParameters = field(default_factory=PruningParameters)
    limits: StructuralLimits = field(default_factory=StructuralLimits)
    proposals: ProposalParameters = field(default_factory=ProposalParameters)
    experiment: str = "evolution"
    split_protocol: str = "first_appearance"
    objective_mode: str = "three"
    search: SearchParameters = field(default_factory=SearchParameters)

    @property
    def tabpfn_enabled(self) -> bool:
        return "tabpfn" in self.models

    @property
    def folds(self) -> int:
        return self.outer_folds

    @property
    def nested_folds(self) -> int | None:
        return self.inner_folds

    def __post_init__(self) -> None:
        if (
            self.experiment == "evolution"
            and any(row.generator == "differential" for row in self.search.definitions)
            and self.evolution.population_size < MIN_DIFFERENTIAL_POPULATION
        ):
            raise ValueError("DE requiere una población compartida de al menos cuatro")
        if self.objective_mode != "three":
            raise ValueError("El protocolo vigente requiere los tres objetivos")
        if self.experiment != "evolution" or self.split_protocol != "first_appearance":
            raise ValueError(
                "El estudio requiere evolución y primera aparición temporal"
            )
        if type(self.calibration_required) is not bool:
            raise ValueError("La exigencia de calibración debe ser booleana")
        if (
            isinstance(self.calibration_fraction, bool)
            or not isinstance(self.calibration_fraction, (int, float))
            or not isfinite(self.calibration_fraction)
            or not 0 < self.calibration_fraction < 1
        ):
            raise ValueError("La proporción de calibración debe estar entre cero y uno")
        nonnegative = (
            self.trials,
            self.cv_seed,
            self.rl_warmup,
            self.representative_seed,
        )
        if any(type(value) is not int or value < 0 for value in nonnegative):
            raise ValueError(
                "Ensayos, semillas y warmup deben ser enteros no negativos"
            )
        if self.label not in ("comprobacion", "posterior"):
            raise ValueError("Etiqueta de estudio inválida")
        if self.nested_folds is not None and (
            type(self.nested_folds) is not int or self.nested_folds < 2
        ):
            raise ValueError(
                "La validación anidada requiere al menos dos pliegues internos"
            )
        if not self.models or not self.seeds:
            raise ValueError("El estudio requiere familias y semillas")
        if self.models == ("targetwise",):
            raise ValueError("El compuesto requiere al menos una familia candidata")
        if self.final_tuning_folds < 2:
            raise ValueError("La búsqueda final requiere al menos dos pliegues")
        if len(set(self.models)) != len(self.models) or any(
            model not in MODEL_NAMES for model in self.models
        ):
            raise ValueError("Familias inválidas o repetidas")
        if len(set(self.seeds)) != len(self.seeds) or any(
            type(seed) is not int or seed < 0 for seed in self.seeds
        ):
            raise ValueError("Semillas inválidas o repetidas")
        if self.representative_seed not in self.seeds:
            raise ValueError("La semilla representativa debe pertenecer al estudio")
        positive = (
            self.folds,
            self.final_tuning_folds,
            self.rl_steps,
            self.rl_batch,
            self.rl_hidden,
            self.validation_interval,
        )
        if any(type(value) is not int or value < 1 for value in positive):
            raise ValueError("Los tamaños y períodos deben ser enteros positivos")
        if self.rl_batch > self.rl.dqn.replay_size:
            raise ValueError("El replay DQN debe admitir el tamaño de lote compartido")
        if self.folds < 2 or self.trials < 0 or self.rl_warmup < 0 or self.cv_seed < 0:
            raise ValueError("Pliegues, ensayos, warmup o semilla CV inválidos")
        if (
            not self.budgets
            or len(set(self.budgets)) != len(self.budgets)
            or any(
                type(budget) is not int
                or not 2 <= budget <= self.proposals.max_evaluations
                for budget in self.budgets
            )
        ):
            raise ValueError("Presupuestos incompatibles con las evaluaciones máximas")
        if min(self.budgets) <= self.evolution.population_size:
            raise ValueError(
                "Cada presupuesto evolutivo debe superar la población inicial"
            )


@dataclass
class TaskRecord:
    key: str
    status: str = "pending"
    attempts: list[str] = field(default_factory=list)
    directory: str | None = None
    error: str | None = None
    seconds: float = 0.0
    signatures: dict[str, str] = field(default_factory=dict)
    measurement_files: list[str] = field(default_factory=list)
    measurement_signatures: dict[str, str] = field(default_factory=dict)
    worker_history: list[dict[str, JsonValue]] = field(default_factory=list)
    log_files: list[str] = field(default_factory=list)
    log_signatures: dict[str, str] = field(default_factory=dict)


@dataclass
class StudySession:
    config: StudyConfig
    dataset_signature: str
    source_signature: str
    versions: dict[str, str]
    tasks: dict[str, TaskRecord] = field(default_factory=dict)
    test_started: bool = False
    preparation_tasks: dict[str, TaskRecord] = field(default_factory=dict)
    execution_runs: list[dict[str, JsonValue]] = field(default_factory=list)
    predictor_import_signature: str | None = None
    source_compatibility: dict[str, JsonValue] | None = None
    executor: CpuPool | None = field(default=None, repr=False)
    gpu_executor: CpuPool | None = field(default=None, repr=False)
    telemetry: CampaignMonitor | None = field(default=None, repr=False)

    def save(self) -> None:
        atomic_json(
            self.config.output_dir / STATE_FILE,
            {
                "schema_version": STUDY_SCHEMA_VERSION,
                "config": asdict(self.config),
                "dataset_signature": self.dataset_signature,
                "source_signature": self.source_signature,
                "versions": self.versions,
                "test_started": self.test_started,
                "tasks": [asdict(task) for task in self.tasks.values()],
                "preparation_tasks": [
                    asdict(task) for task in self.preparation_tasks.values()
                ],
                "execution_runs": self.execution_runs,
                "predictor_import_signature": self.predictor_import_signature,
                "source_compatibility": self.source_compatibility,
            },
        )


def atomic_json(path: Path, payload: JsonValue) -> None:
    write_json(path, payload)


def source_signature() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = [
        *sorted((root / "src").rglob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    signatures = {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in paths
    }
    return hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest()


def study_source_signature(directory: Path) -> str:
    """Acepta únicamente una revisión operativa inventariada para este estudio."""
    current = source_signature()
    saved = read_study(directory)
    original = str(saved["source_signature"])
    if original == current:
        return original
    reference = saved.get("source_compatibility")
    if not isinstance(reference, dict):
        raise ValueError("Cambió el código del estudio; elegir un destino nuevo")
    receipt = (directory / str(reference["file"])).resolve()
    if not receipt.is_relative_to(directory.resolve()):
        raise ValueError("La revisión operativa debe estar dentro del estudio")
    if source_digest(receipt) != reference["signature"]:
        raise ValueError("Cambió la evidencia de compatibilidad del código")
    evidence = read_artifact(receipt)
    if (
        evidence.get("previous_source_signature") != original
        or evidence.get("current_source_signature") != current
        or evidence.get("scope") != "io_only"
    ):
        raise ValueError("El código no corresponde a la revisión operativa aprobada")
    return original


def environment_versions() -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{name: version(name) for name in DEPENDENCIES},
    }
    try:
        result["tabpfn"] = version("tabpfn")
    except PackageNotFoundError:
        pass
    return result


def read_study(directory: Path) -> dict[str, JsonValue]:
    value = read_artifact(directory / STATE_FILE)
    if value.get("schema_version") != STUDY_SCHEMA_VERSION:
        raise ValueError("El directorio no contiene un estudio compatible")
    return value


def load_study_config(directory: Path) -> StudyConfig:
    raw = read_study(directory).get("config")
    if not isinstance(raw, dict):
        raise ValueError("Falta configuración del estudio")
    if "limits" not in raw:
        raise ValueError(
            "El estudio usa supuestos anteriores; ejecutar make reset-study antes de entrenar"
        )
    values = dict(raw)
    values["output_dir"] = directory.resolve()
    values["data_dir"] = Path(str(values["data_dir"]))
    values["models"] = tuple(cast(list[ModelName], values["models"]))
    values["seeds"] = tuple(cast(list[int], values["seeds"]))
    values["budgets"] = tuple(cast(list[int], values["budgets"]))
    parameters = cast(dict[str, int | float], values["parameters"])
    values["parameters"] = ModelParameters(**parameters)
    values["evolution"] = EvolutionParameters(
        **cast(dict[str, int | float | bool], values.get("evolution", {}))
    )
    from truss_research.optimizacion.protocolo import (
        read_analysis,
        read_proposals,
        read_tabpfn,
    )

    values["analysis"] = read_analysis(values["analysis"])
    values["tabpfn"] = read_tabpfn(values.get("tabpfn", {}))
    values["pruning"] = PruningParameters(**values.get("pruning", {}))
    values["limits"] = StructuralLimits(**values["limits"])
    measurement = dict(cast(dict[str, JsonValue], values["measurement"]))
    values["measurement"] = MeasurementParameters(**measurement)
    values["execution"] = ExecutionParameters(
        **cast(dict[str, int | float], values["execution"])
    )
    values["rl"] = read_rl_parameters(values.get("rl", {}))
    values["proposals"] = read_proposals(values.get("proposals", {}))
    values["search"] = read_search_parameters(values.get("search", {}))
    return StudyConfig(**values)


def open_session(config: StudyConfig, resume: bool) -> StudySession:
    from truss_research.datos.procesados import load_structural_limits

    if load_structural_limits(config.data_dir) != config.limits:
        raise ValueError(
            "Los límites del dataset difieren del protocolo; ejecutar make reset-study y preparar de nuevo"
        )
    session = StudySession(
        config,
        data_signature(config.data_dir),
        source_signature(),
        environment_versions(),
    )
    state = config.output_dir / STATE_FILE
    if not resume:
        if state.exists() or any(
            path.name != ".study.lock" for path in config.output_dir.iterdir()
        ):
            raise ValueError(
                "El estudio ya existe; indicar --reanudar o un destino nuevo"
            )
        session.save()
        return session
    saved = read_study(config.output_dir)
    expected_config = json.loads(json.dumps(asdict(config), default=str))
    if saved.get("config") != expected_config or any(
        saved.get(name) != getattr(session, name)
        for name in ("dataset_signature", "versions")
    ):
        raise ValueError("Cambió la configuración, datos, código o entorno del estudio")
    if saved.get("source_signature") != session.source_signature:
        session.source_signature = study_source_signature(config.output_dir)
    session.test_started = saved.get("test_started") is True
    records = cast(list[dict[str, JsonValue]], saved.get("tasks", []))
    session.tasks = {str(record["key"]): TaskRecord(**record) for record in records}
    preparation = cast(list[dict[str, JsonValue]], saved.get("preparation_tasks", []))
    session.preparation_tasks = {
        str(record["key"]): TaskRecord(**record) for record in preparation
    }
    session.execution_runs = cast(
        list[dict[str, JsonValue]], saved.get("execution_runs", [])
    )
    session.predictor_import_signature = saved.get("predictor_import_signature")
    session.source_compatibility = saved.get("source_compatibility")
    return session


def artifact_signatures(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): source_digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def verify_task(task: TaskRecord) -> Path:
    if task.status != "complete" or task.directory is None:
        raise ValueError(f"La tarea {task.key} está incompleta")
    directory = Path(task.directory)
    if not task.signatures or artifact_signatures(directory) != task.signatures:
        raise ValueError(f"Los artefactos de {task.key} cambiaron o están incompletos")
    if any(
        not Path(path).is_file() or source_digest(Path(path)) != signature
        for path, signature in task.measurement_signatures.items()
    ):
        raise ValueError(f"Las mediciones de {task.key} cambiaron o están incompletas")
    if any(
        not Path(path).is_file() or source_digest(Path(path)) != signature
        for path, signature in task.log_signatures.items()
    ):
        raise ValueError(f"Los logs de {task.key} cambiaron o están incompletos")
    return directory


@contextmanager
def study_lock(directory: Path) -> Iterator[None]:
    """El sistema operativo libera el bloqueo incluso si termina el proceso."""
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".study.lock"
    handle = lock_path.open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        lock_handle(handle.fileno(), False)
    except OSError as error:
        handle.close()
        raise ValueError("Otro coordinador está usando este estudio") from error
    try:
        yield
    finally:
        lock_handle(handle.fileno(), True)
        handle.close()


def lock_handle(descriptor: int, release: bool) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB)
