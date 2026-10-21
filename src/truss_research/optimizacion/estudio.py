"""Coordinación reanudable con búsquedas y pliegues paralelos en CPU."""

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from time import perf_counter, sleep

from truss_research.datos.preparacion import source_digest
from truss_research.evaluacion.recursos import resource_session
from truss_research.evaluacion.registro import attempt_logs, execution_log
from truss_research.evaluacion.telemetria_cpu import CampaignMonitor
from truss_research.optimizacion.comparacion import (
    PolicyReference,
    evaluate_ranking,
    evaluate_selection_suite,
)
from truss_research.optimizacion.contratos import SelectionConfig, StudyTask
from truss_research.optimizacion.ejecucion_cpu import MemoryAdmission, cpu_pool
from truss_research.optimizacion.estudio_cierre import (
    evaluate_test_predictor,
    family_selection,
    freeze_study,
)
from truss_research.optimizacion.estudio_estado import (
    StudyConfig,
    StudyPhase,
    StudySession,
    TaskRecord,
    artifact_signatures,
    open_session,
    study_lock,
    verify_task,
)
from truss_research.surrogates.configuracion import PredictorConfig
from truss_research.surrogates.experimentos import read_run, train_predictor
from truss_research.surrogates.explicabilidad import ExplanationConfig, explain_model

logger = logging.getLogger(__name__)


def predictor_path(session: StudySession, task: StudyTask) -> Path:
    return verify_task(session.tasks[f"predictor/{task.model}/{task.seed}"])


def policy_references(session: StudySession) -> tuple[PolicyReference, ...]:
    return tuple(
        PolicyReference(
            verify_task(session.tasks[f"policy/{variant}/{seed}"]), seed, variant
        )
        for variant in session.config.rl.variants
        for seed in session.config.seeds
    )


def selection_config(
    session: StudySession, context: tuple[Path, Path, StudyTask]
) -> SelectionConfig:
    predictor, destination, task = context
    config = session.config
    return SelectionConfig(
        config.data_dir,
        predictor,
        destination,
        split=task.split,
        catalog_size=config.catalog_size,
        budgets=config.budgets,
        seeds=config.seeds,
        episodes_per_seed=config.episodes_per_seed,
        evolution=config.evolution,
    )


def execute_policy(session: StudySession, task: StudyTask, destination: Path) -> None:
    from truss_research.optimizacion.entrenamiento import train_policy
    from truss_research.optimizacion.politica import configured_policy

    config = session.config
    record = session.tasks[task.key]
    prior = [
        Path(path)
        for path in reversed(record.attempts[:-1])
        if (Path(path) / "run.json").is_file()
    ]
    resume = prior[0] if prior else None
    if (
        resume is not None
        and read_run(resume).get("completed_steps", 0) >= config.rl_steps
    ):
        resume = None
    winner = family_selection(session)[0]
    train_policy(configured_policy(config, (winner, destination, resume), task))


def dispatch_task(session: StudySession, task: StudyTask, destination: Path) -> None:
    config = session.config
    if task.kind == "predictor":
        train_predictor(
            PredictorConfig(
                config.data_dir,
                destination,
                model=task.model,
                seed=task.seed,
                folds=config.folds,
                cv_seed=config.cv_seed,
                nested_folds=config.nested_folds,
                final_tuning_folds=config.final_tuning_folds,
                candidate_models=tuple(m for m in config.models if m != "targetwise"),
                trials=0 if task.model == "tabpfn" else config.trials,
                tabpfn=config.tabpfn,
                pruning=config.pruning,
                device="cuda" if task.model in config.execution.gpu_models else "cpu",
                parameters=config.parameters,
                reuse_dir=(
                    verify_task(
                        session.tasks[
                            f"predictor/{task.model}/{config.representative_seed}"
                        ]
                    )
                    if task.seed != config.representative_seed
                    else None
                ),
                study_path=config.output_dir
                / "studies"
                / f"{task.model}_{task.seed}.sqlite",
            )
        )
        return
    if task.kind == "policy":
        execute_policy(session, task, destination)
        return
    if task.kind == "explain":
        explain_model(
            ExplanationConfig(
                config.data_dir,
                predictor_path(session, task),
                destination,
                seed=config.cv_seed,
                background_size=config.background_size,
                permutation_rounds=config.permutation_rounds,
                device=config.tabpfn.device
                if task.model in ("tabpfn", "targetwise")
                else None,
            )
        )
        return
    if task.kind == "test_predictor":
        evaluate_test_predictor(predictor_path(session, task), destination)
        return
    predictor = (
        predictor_path(session, task)
        if task.kind == "ranking"
        else family_selection(session)[0]
    )
    selection = selection_config(session, (predictor, destination, task))
    if task.kind == "ranking":
        evaluate_ranking(selection)
        return
    evaluate_selection_suite(selection, policy_references(session))


def run_task(session: StudySession, task: StudyTask) -> bool:
    record = session.tasks[task.key]
    if record.status == "complete":
        verify_task(record)
        logger.info("Conservada: %s", task.key)
        return True
    if session.test_started and not task.key.startswith("test/"):
        raise ValueError(
            "No se permite entrenar ni revisar decisiones después del cierre"
        )
    destination = (
        session.config.output_dir
        / "tasks"
        / task.key
        / f"attempt_{len(record.attempts) + 1:03d}"
    )
    if (
        record.status == "running"
        and record.measurement_files
        and Path(record.measurement_files[-1]).is_file()
    ):
        from truss_research.datos.artefactos import read_artifact

        interrupted = read_artifact(Path(record.measurement_files[-1]))
        record.seconds += float(interrupted["phases"][0]["wall_seconds"])
        prior_path = record.measurement_files[-1]
        record.measurement_signatures[prior_path] = source_digest(Path(prior_path))
    record.attempts.append(str(destination))
    record.log_files.extend(str(path) for path in attempt_logs(destination))
    measurement_path = destination.with_suffix(".resources.json")
    record.measurement_files.append(str(measurement_path))
    record.status, record.error = "running", None
    session.save()
    logger.info("Iniciando %s (intento %s)", task.key, len(record.attempts))
    start = perf_counter()
    try:
        with (
            execution_log(destination.with_suffix(".log"), task.key),
            resource_session(measurement_path, session.config.measurement),
        ):
            dispatch_task(session, task, destination)
        record.signatures = artifact_signatures(destination)
        record.directory, record.status = str(destination), "complete"
    except Exception as error:
        record.status, record.error = "failed", f"{type(error).__name__}: {error}"
        logger.exception("Falló %s", task.key)
    except KeyboardInterrupt:
        record.status, record.error = (
            "failed",
            "Entrenamiento interrumpido por el usuario",
        )
        raise
    finally:
        record.log_signatures.update(
            {
                str(path): source_digest(path)
                for path in attempt_logs(destination)
                if path.is_file()
            }
        )
        record.seconds += perf_counter() - start
        if measurement_path.is_file():
            record.measurement_signatures[str(measurement_path)] = source_digest(
                measurement_path
            )
        session.save()
    logger.info("%s: %s (%.1f s acumulados)", task.key, record.status, record.seconds)
    return record.status == "complete"


def run_tasks(session: StudySession, tasks: tuple[StudyTask, ...]) -> None:
    if (
        tasks
        and all(task.kind == "predictor" for task in tasks)
        and session.config.execution.effective_workers > 1
    ):
        from truss_research.optimizacion.paralelismo import run_parallel_predictions

        run_parallel_predictions(session)
        return
    if tasks and session.config.execution.effective_workers > 1:
        from truss_research.optimizacion.etapas_cpu import run_parallel_tasks

        run_parallel_tasks(session, tasks)
        return
    for task in tasks:
        session.tasks.setdefault(task.key, TaskRecord(task.key))
    session.save()
    results: list[bool] = []
    gate = MemoryAdmission(session.config.execution)
    for number, task in enumerate(tasks):
        while session.tasks[task.key].status != "complete" and not gate.allow(0):
            if session.telemetry is not None:
                session.telemetry.update((0, len(tasks) - number, 0, True))
            sleep(session.config.execution.telemetry_interval_seconds)
        if session.telemetry is not None:
            session.telemetry.update((1, len(tasks) - number - 1, 0, False))
        results.append(run_task(session, task))
    if session.telemetry is not None:
        session.telemetry.update((0, 0, 0, False))
    if not all(results):
        raise RuntimeError(
            "Hay tareas fallidas; consultar study.json antes de reanudar"
        )


def prediction_tasks(config: StudyConfig) -> tuple[StudyTask, ...]:
    return tuple(
        StudyTask(f"predictor/{model}/{seed}", "predictor", model, seed)
        for model in config.models
        for seed in (
            config.representative_seed,
            *(seed for seed in config.seeds if seed != config.representative_seed),
        )
    )


def validation_tasks(session: StudySession) -> tuple[StudyTask, ...]:
    config = session.config
    winner, _ = family_selection(session)
    session.winner_dir = winner
    winner_config = read_run(winner)["config"]
    winner_name = str(winner_config["model"]) if isinstance(winner_config, dict) else ""
    policies = tuple(
        StudyTask(f"policy/{variant}/{seed}", "policy", seed=seed, variant=variant)
        for variant in config.rl.variants
        for seed in config.seeds
    )
    explanations = tuple(
        StudyTask(f"explain/{model}/{seed}", "explain", model, seed)
        for model in config.models
        for seed in config.seeds
        if seed == config.representative_seed or model == winner_name
    )
    rankings = tuple(
        StudyTask(f"ranking/{model}/{seed}", "ranking", model, seed)
        for model in config.models
        for seed in config.seeds
    )
    return (
        *policies,
        *explanations,
        *rankings,
        StudyTask("decision/validation", "decision"),
    )


def final_tasks(config: StudyConfig) -> tuple[StudyTask, ...]:
    predictions = tuple(
        StudyTask(
            f"test/predictor/{model}/{seed}", "test_predictor", model, seed, "test"
        )
        for model in config.models
        for seed in config.seeds
    )
    rankings = tuple(
        StudyTask(f"test/ranking/{model}/{seed}", "ranking", model, seed, "test")
        for model in config.models
        for seed in config.seeds
    )
    return (
        *predictions,
        *rankings,
        StudyTask("test/decision", "decision", split="test"),
    )


def requires_execution(session: StudySession, phase: StudyPhase) -> bool:
    config = session.config
    predictions = prediction_tasks(config)
    if phase != "final" and any(
        task.key not in session.tasks or session.tasks[task.key].status != "complete"
        for task in predictions
    ):
        return True
    tasks = (
        final_tasks(config)
        if phase == "final"
        else validation_tasks(session)
        if phase != "predictores"
        else ()
    )
    if phase == "todo":
        tasks = (*tasks, *final_tasks(config))
    return any(
        task.key not in session.tasks or session.tasks[task.key].status != "complete"
        for task in tasks
    )


@contextmanager
def execution_record(session: StudySession, phase: StudyPhase) -> Iterator[None]:
    if not requires_execution(session, phase):
        yield
        return
    record = {
        "phase": phase,
        "status": "running",
        "wall_seconds": 0.0,
        "cpu_workers": session.config.execution.effective_workers,
        "gpu_workers": session.config.execution.gpu_workers
        if session.config.tabpfn_enabled and session.config.tabpfn.device == "cuda"
        else 0,
        "tabpfn_device": session.config.tabpfn.device,
        "rl_device": session.config.execution.rl_device,
        "work_seconds": 0.0,
    }
    before = sum(
        task.seconds
        for task in (*session.tasks.values(), *session.preparation_tasks.values())
    )
    session.execution_runs.append(record)
    telemetry_path = (
        session.config.output_dir
        / "execution"
        / f"run_{len(session.execution_runs):03d}.telemetry.csv"
    )
    record["telemetry_file"] = str(telemetry_path)
    session.save()
    start = perf_counter()
    try:
        with ExitStack() as stack:
            session.telemetry = stack.enter_context(
                CampaignMonitor(
                    telemetry_path, session.config.execution.telemetry_interval_seconds
                )
            )
            if session.config.execution.effective_workers > 1:
                session.executor = stack.enter_context(
                    cpu_pool(session.config.execution)
                )
            if (
                (
                    session.config.tabpfn_enabled
                    and session.config.tabpfn.device == "cuda"
                )
                or session.config.execution.rl_device == "cuda"
                or session.config.execution.gpu_models
            ):
                session.gpu_executor = stack.enter_context(
                    cpu_pool(session.config.execution, "cuda")
                )
            yield
        record["status"] = "complete"
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        if session.executor is not None:
            record["memory_recoveries"] = session.executor.recoveries
            record["final_worker_capacity"] = session.executor.capacity
        session.executor = None
        session.gpu_executor = None
        record["telemetry_errors"] = (
            session.telemetry.errors if session.telemetry is not None else []
        )
        session.telemetry = None
        if telemetry_path.is_file():
            record["telemetry_signature"] = source_digest(telemetry_path)
        record["wall_seconds"] = perf_counter() - start
        record["work_seconds"] = (
            sum(
                task.seconds
                for task in (
                    *session.tasks.values(),
                    *session.preparation_tasks.values(),
                )
            )
            - before
        )
        session.save()


def run_study(
    config: StudyConfig, phase: StudyPhase = "todo", resume: bool = False
) -> Path:
    if config.experiment == "evolution" and phase != "predictores":
        from truss_research.evaluacion.busqueda_evolutiva import run_evolution_study

        return run_evolution_study(config, phase, resume)
    from truss_research.evaluacion.informe_estudio import (
        latest_study_report,
        summarize_study,
    )

    if phase not in ("predictores", "validacion", "final", "todo"):
        raise ValueError("Fase desconocida")
    config.execution.validate_campaign()
    logger.info(
        "Campaña: hasta %s trabajadores CPU, %s GPU; TabPFN=%s; RL y demás familias=cpu",
        config.execution.effective_workers,
        int(config.tabpfn_enabled and config.tabpfn.device == "cuda"),
        config.tabpfn.device if config.tabpfn_enabled else "inactivo",
    )
    validate_training_protocol(config)
    config = replace(
        config,
        output_dir=config.output_dir.resolve(),
        data_dir=config.data_dir.resolve(),
    )
    with study_lock(config.output_dir):
        session = open_session(config, resume)
        with execution_record(session, phase):
            if phase in ("predictores", "validacion", "todo"):
                run_tasks(session, prediction_tasks(config))
            if phase in ("validacion", "todo"):
                run_tasks(session, validation_tasks(session))
            if phase in ("final", "todo"):
                expected = (
                    *prediction_tasks(config),
                    *validation_tasks(session),
                )
                for task in expected:
                    if task.key not in session.tasks:
                        raise ValueError(f"Falta completar {task.key} antes del cierre")
                    verify_task(session.tasks[task.key])
                freeze_study(session, family_selection(session)[0])
                run_tasks(session, final_tasks(config))
        if config.experiment == "evolution":
            return config.output_dir
        reports = config.output_dir / "reports"
        reports.mkdir(exist_ok=True)
        if latest_study_report(config.output_dir) is None:
            numbers = [
                int(path.name.removeprefix("report_"))
                for path in reports.glob("report_*")
                if path.name.removeprefix("report_").isdigit()
            ]
            destination = reports / f"report_{max(numbers, default=0) + 1:03d}"
            summarize_study(config.output_dir, destination)
    return config.output_dir


def validate_training_protocol(config: StudyConfig) -> None:
    """Rechaza reservas o pliegues insuficientes antes de ejecutar búsquedas."""
    import numpy as np

    from truss_research.datos.particion import training_folds
    from truss_research.datos.procesados import load_design_split, read_manifest
    from truss_research.surrogates.calibracion import SplitConformal

    if config.tabpfn_enabled:
        from truss_research.evaluacion.piloto_tabpfn import require_pilot

        require_pilot(config)

    split = load_design_split(config.data_dir)
    from truss_research.datos.procesados import manifest_protocol
    from truss_research.datos.temporal import first_appearance_folds

    if manifest_protocol(read_manifest(config.data_dir)) != config.split_protocol:
        raise ValueError("La partición preparada difiere del protocolo solicitado")
    if config.experiment == "evolution":
        folds = first_appearance_folds(split.training, config.outer_folds)
        first_appearance_folds(split.training, config.final_tuning_folds)
        for fitting, _ in folds:
            first_appearance_folds(
                split.training.iloc[fitting], config.inner_folds or config.outer_folds
            )
        if config.calibration_required and (
            split.calibration is None or len(split.calibration) < 19
        ):
            raise ValueError(
                "La calibración temporal al 95 % requiere al menos 19 diseños"
            )
        return
    if (
        config.calibration_required
        and read_manifest(config.data_dir).get("calibration_fraction")
        != config.calibration_fraction
    ):
        raise ValueError(
            "La reserva de calibración difiere del protocolo; elegir una preparación compatible"
        )
    if config.calibration_required and split.calibration is None:
        raise ValueError(
            "El protocolo requiere calibración independiente; ejecutar make prepare con los datos v2"
        )
    if split.calibration is not None:
        SplitConformal(np.zeros((len(split.calibration), 3))).widths(0.95)
    folds = training_folds(split.training, config.outer_folds, config.cv_seed)
    training_folds(split.training, config.final_tuning_folds, config.cv_seed)
    if config.inner_folds is not None:
        for fitting, _ in folds:
            training_folds(
                split.training.iloc[fitting], config.inner_folds, config.cv_seed
            )
