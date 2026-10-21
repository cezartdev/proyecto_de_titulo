"""Dependencias de validación y prueba para el mismo pool global de predictores."""

from dataclasses import replace
from math import ceil
from pathlib import Path

from truss_research.optimizacion.comparacion import PolicyReference
from truss_research.optimizacion.contratos import StudyTask
from truss_research.optimizacion.estudio_cierre import family_selection
from truss_research.optimizacion.estudio_estado import StudySession, verify_task
from truss_research.optimizacion.paralelismo import (
    JobNode,
    begin_attempt,
    node_record,
    run_nodes,
)
from truss_research.optimizacion.trabajos_estudio import StudyJob
from truss_research.surrogates.experimentos import read_run

BLOCKS_PER_WORKER = 2


def stage_graph(
    session: StudySession, tasks: tuple[StudyTask, ...]
) -> tuple[JobNode, ...]:
    nodes: list[JobNode] = []
    config = session.config
    policies = tuple(task.key for task in tasks if task.kind == "policy")
    count = len(config.seeds) * config.episodes_per_seed * len(config.budgets)
    size = max(
        1, ceil(count / (BLOCKS_PER_WORKER * config.execution.effective_workers))
    )
    for task in tasks:
        if task.kind != "decision":
            nodes.append(JobNode(task.key, task.kind, task.model, task.seed, task=task))
            continue
        bank_key = f"{task.key}/bank"
        nodes.append(JobNode(bank_key, "bank", task.model, task.seed, task=task))
        blocks = tuple(
            f"{task.key}/block_{number:03d}"
            for number, _ in enumerate(range(0, count, size))
        )
        nodes.extend(
            JobNode(
                key,
                "block",
                task.model,
                task.seed,
                (bank_key, *policies),
                task=task,
                indices=tuple(range(start, min(start + size, count))),
            )
            for key, start in zip(blocks, range(0, count, size), strict=True)
        )
        nodes.append(
            JobNode(task.key, "decision", task.model, task.seed, blocks, task=task)
        )
    return tuple(nodes)


def policy_resume(session: StudySession, node: JobNode) -> Path | None:
    record = node_record(session, node)
    prior = [
        Path(path)
        for path in reversed(record.attempts[:-1])
        if (Path(path) / "run.json").is_file()
    ]
    if (
        not prior
        or read_run(prior[0]).get("completed_steps", 0) >= session.config.rl_steps
    ):
        return None
    return prior[0]


def start_study_node(session: StudySession, node: JobNode) -> StudyJob:
    if node.task is None:
        raise ValueError("Falta el comando de etapa")
    if session.test_started and node.task.split != "test":
        raise ValueError("No se permite revisar validación después del cierre")
    task = node.task
    destination = begin_attempt(session, node)
    if node.kind == "explain":
        from shutil import copyfile

        previous = next(
            (
                Path(path).with_suffix(".shap.npz")
                for path in reversed(node_record(session, node).attempts[:-1])
                if Path(path).with_suffix(".shap.npz").is_file()
            ),
            None,
        )
        if previous is not None:
            copyfile(previous, destination.with_suffix(".shap.npz"))
    if session.winner_dir is None:
        session.winner_dir = family_selection(session)[0]
    predictor = (
        verify_task(session.tasks[f"predictor/{task.model}/{task.seed}"])
        if node.kind in ("explain", "ranking", "test_predictor")
        else session.winner_dir
    )
    policies = (
        tuple(
            PolicyReference(
                verify_task(session.tasks[f"policy/{variant}/{seed}"]), seed, variant
            )
            for variant in session.config.rl.variants
            for seed in session.config.seeds
        )
        if node.kind in ("block", "decision")
        else ()
    )
    bank = (
        verify_task(session.preparation_tasks[f"{task.key}/bank"])
        if node.kind in ("block", "decision")
        else None
    )
    blocks = (
        tuple(verify_task(session.preparation_tasks[key]) for key in node.dependencies)
        if node.kind == "decision"
        else ()
    )
    job = StudyJob(
        replace(task, key=node.key),
        session.config,
        destination,
        predictor,
        node.kind,
        policy_resume(session, node) if node.kind == "policy" else None,
        policies,
        bank,
        node.indices,
        blocks,
    )
    session.save()
    return job


def run_parallel_tasks(session: StudySession, tasks: tuple[StudyTask, ...]) -> None:
    run_nodes(session, stage_graph(session, tasks))
