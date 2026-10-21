"""Comandos inmutables para etapas independientes; ningún trabajador guarda estudio."""

from dataclasses import dataclass
from pathlib import Path

from truss_research.optimizacion.comparacion import PolicyReference, evaluate_ranking
from truss_research.optimizacion.comparacion_paralela import (
    ComparisonCommand,
    aggregate_comparison,
    comparison_block,
    prepare_bank,
)
from truss_research.optimizacion.contratos import SelectionConfig, StudyTask
from truss_research.optimizacion.estudio_cierre import evaluate_test_predictor
from truss_research.optimizacion.estudio_estado import StudyConfig
from truss_research.optimizacion.trabajos_predictivos import (
    JobOutcome,
    JobResources,
    run_measured_job,
)
from truss_research.surrogates.explicabilidad import ExplanationConfig, explain_model


@dataclass(frozen=True)
class StudyJob:
    task: StudyTask
    config: StudyConfig
    destination: Path
    predictor: Path
    kind: str
    resume: Path | None = None
    policies: tuple[PolicyReference, ...] = ()
    bank: Path | None = None
    indices: tuple[int, ...] = ()
    blocks: tuple[Path, ...] = ()


def comparison_command(job: StudyJob) -> ComparisonCommand:
    config = job.config
    return ComparisonCommand(
        SelectionConfig(
            config.data_dir,
            job.predictor,
            job.destination,
            split=job.task.split,
            catalog_size=config.catalog_size,
            budgets=config.budgets,
            seeds=config.seeds,
            episodes_per_seed=config.episodes_per_seed,
            evolution=config.evolution,
        ),
        job.policies,
        job.bank,
        job.indices,
        job.blocks,
    )


def policy_job(job: StudyJob) -> None:
    from truss_research.optimizacion.entrenamiento import train_policy
    from truss_research.optimizacion.politica import configured_policy

    config = job.config
    train_policy(
        configured_policy(
            config, (job.predictor, job.destination, job.resume), job.task
        )
    )


def execute_study_job(job: StudyJob) -> None:
    if job.kind in ("bank", "block", "decision"):
        {
            "bank": prepare_bank,
            "block": comparison_block,
            "decision": aggregate_comparison,
        }[job.kind](comparison_command(job))
        return
    if job.kind == "policy":
        policy_job(job)
        return
    if job.kind == "explain":
        explain_model(
            ExplanationConfig(
                job.config.data_dir,
                job.predictor,
                job.destination,
                seed=job.config.cv_seed,
                background_size=job.config.background_size,
                permutation_rounds=job.config.permutation_rounds,
                device=job.config.tabpfn.device
                if job.task.model in ("tabpfn", "targetwise")
                else None,
            )
        )
        return
    if job.kind == "test_predictor":
        evaluate_test_predictor(job.predictor, job.destination)
        return
    if job.kind == "ranking":
        evaluate_ranking(comparison_command(job).config)
        return
    raise ValueError(f"Trabajo CPU desconocido: {job.kind}")


def run_study_job(job: StudyJob) -> JobOutcome:
    return run_measured_job(
        JobResources(
            job.task.key, job.destination, job.config.data_dir, job.config.measurement
        ),
        lambda: execute_study_job(job),
    )
