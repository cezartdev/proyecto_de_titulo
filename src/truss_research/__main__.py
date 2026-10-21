"""Interfaz de las etapas reproducibles; Makefile es la entrada habitual."""

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from truss_research.optimizacion.estudio_estado import StudyConfig

DEFAULT_STUDY = Path("artifacts/evolucion")

DEFAULT_PROTOCOL = Path("config/study.json")
SMOKE_PROTOCOL = Path("config/smoke.json")

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def configure_workers(
    config: "StudyConfig", workers: int | None, resume: bool = False
) -> "StudyConfig":
    if workers is None:
        return config
    execution = replace(config.execution, cpu_workers=workers)
    if resume and execution != config.execution:
        raise ValueError(
            "El número de procesos difiere del estudio guardado; omitir CPU_WORKERS "
            "para conservarlo o elegir un destino nuevo para cambiarlo"
        )
    return replace(config, execution=execution)


def command_protocol(
    args: argparse.Namespace, path: Path, locations: tuple[Path, Path]
) -> "StudyConfig":
    """Carga el protocolo con la elección explícita de CPU o CUDA."""
    from truss_research.optimizacion.protocolo import load_protocol

    return load_protocol(path, locations, getattr(args, "accelerator", None))


def run_preparation(args: argparse.Namespace) -> None:
    from truss_research.datos.preparacion import ensure_dataset
    from truss_research.datos.schemas import (
        DESIGN_DESTINATION,
        PreparationConfig,
        SplitConfig,
    )

    protocol = command_protocol(args, args.config, (DEFAULT_STUDY, DESIGN_DESTINATION))

    ensure_dataset(
        PreparationConfig(
            source=args.origen,
            destination=args.destino or DESIGN_DESTINATION,
            split=SplitConfig(protocol=protocol.split_protocol),
            limits=protocol.limits,
            calibration_fraction=protocol.calibration_fraction
            if args.calibration_fraction is None
            else args.calibration_fraction,
        )
    )


def run_study_command(args: argparse.Namespace) -> None:
    from truss_research.datos.schemas import DESIGN_DESTINATION
    from truss_research.optimizacion.estudio import run_study
    from truss_research.optimizacion.estudio_estado import load_study_config
    from truss_research.optimizacion.protocolo import configure_accelerator, study_plan

    resume = args.reanudar or (args.destino / "study.json").is_file()
    if args.fase == "final" and not resume:
        raise ValueError(
            "Falta completar validación antes del cierre; ejecutar make study"
        )
    config = (
        load_study_config(args.destino)
        if resume
        else command_protocol(
            args,
            args.config or DEFAULT_PROTOCOL,
            (args.destino.resolve(), (args.datos or DESIGN_DESTINATION).resolve()),
        )
    )
    selected = configure_accelerator(config, getattr(args, "accelerator", None))
    if resume and selected != config:
        raise ValueError(
            "El acelerador difiere del estudio guardado; elegir un destino nuevo"
        )
    config = configure_workers(selected, args.cpu_workers, resume)
    if args.config is not None and resume:
        requested = configure_workers(
            command_protocol(
                args,
                args.config,
                (args.destino.resolve(), (args.datos or config.data_dir).resolve()),
            ),
            args.cpu_workers,
        )
        if requested != replace(
            config,
            output_dir=args.destino.resolve(),
            data_dir=config.data_dir.resolve(),
        ):
            raise ValueError("El protocolo JSON difiere del estudio guardado")
    if (
        resume
        and args.datos is not None
        and args.datos.resolve() != config.data_dir.resolve()
    ):
        raise ValueError("No cambiar el directorio de datos al reanudar un estudio")
    if args.plan:
        print(study_plan(config, args.fase))
        return
    run_study(config, args.fase, resume)


def next_export(study: Path, name: str) -> Path:
    root = results_root(study)
    if name in ("sensibilidad", "curvas"):
        incomplete = [
            path
            for path in sorted(root.glob(name + "_*"), reverse=True)
            if (path / "analysis_session.json").is_file()
            and not (path / "provenance.json").exists()
        ]
        if incomplete:
            return incomplete[0]
    numbers = [
        int(path.name.removeprefix(name + "_"))
        for path in root.glob(name + "_*")
        if path.name.removeprefix(name + "_").isdigit()
    ]
    return root / f"{name}_{max(numbers, default=0) + 1:03d}"


def results_root(study: Path) -> Path:
    """Ubica resultados junto al estudio y aísla los directorios temporales."""
    from truss_research.evaluacion.registro import results_root as resolve_results

    return resolve_results(study)


def smoke_config(
    root: Path,
    workers: int | None = None,
    selection: tuple[Path, str | None] = (SMOKE_PROTOCOL, None),
) -> "StudyConfig":
    """Reutiliza una comprobación compatible; conserva las anteriores al cambiar código."""
    from truss_research.datos.schemas import DESIGN_DESTINATION
    from truss_research.optimizacion.estudio_estado import open_session
    from truss_research.optimizacion.protocolo import load_protocol

    candidates = sorted(root.glob("comprobacion_[0-9][0-9][0-9]"), reverse=True)
    candidates.append(root / "comprobacion")
    for candidate in candidates:
        if not (candidate / "study.json").is_file():
            continue
        config = configure_workers(
            load_protocol(
                selection[0],
                (candidate.resolve(), DESIGN_DESTINATION.resolve()),
                selection[1],
            ),
            workers,
        )
        try:
            open_session(config, True)
        except ValueError as error:
            logging.getLogger(__name__).info("Se conserva %s: %s", candidate, error)
            continue
        return config
    numbers = [
        int(path.name.removeprefix("comprobacion_"))
        for path in candidates
        if path.name.removeprefix("comprobacion_").isdigit()
    ]
    destination = root / f"comprobacion_{max(numbers, default=0) + 1:03d}"
    return configure_workers(
        load_protocol(
            selection[0],
            (destination.resolve(), DESIGN_DESTINATION.resolve()),
            selection[1],
        ),
        workers,
    )


def run_smoke_command(args: argparse.Namespace) -> None:
    from truss_research.datos.schemas import DESIGN_DESTINATION
    from truss_research.optimizacion.estudio import run_study

    config = (
        configure_workers(
            command_protocol(
                args,
                args.config or SMOKE_PROTOCOL,
                (args.destino.resolve(), (args.datos or DESIGN_DESTINATION).resolve()),
            ),
            args.cpu_workers,
        )
        if args.destino is not None
        else smoke_config(
            Path("artifacts"),
            args.cpu_workers,
            (args.config or SMOKE_PROTOCOL, getattr(args, "accelerator", None)),
        )
    )
    if args.solo_dummy:
        config = replace(
            config,
            models=("dummy", "ridge")
            if config.experiment == "evolution"
            else ("dummy",),
        )
    logging.getLogger(__name__).info("Comprobación: %s", config.output_dir)
    run_study(
        config,
        "validacion" if config.experiment == "evolution" else "todo",
        (config.output_dir / "study.json").is_file(),
    )


def export_evolution_command(args: argparse.Namespace, kind: str) -> bool:
    from truss_research.evaluacion.informe_busqueda import export_search
    from truss_research.optimizacion.estudio_estado import load_study_config

    study = args.estudio or DEFAULT_STUDY
    if load_study_config(study).experiment != "evolution":
        return False
    destination = args.destino or next_export(
        study, "presentacion" if kind == "notebooks" else kind
    )
    print(export_search(study, destination, kind))
    return True


def run_summary_command(args: argparse.Namespace) -> None:
    if export_evolution_command(args, "informe"):
        return
    from truss_research.evaluacion.informe_estudio import summarize_study

    summarize_study(args.estudio, args.destino or next_export(args.estudio, "informe"))


def run_audit_command(args: argparse.Namespace) -> None:
    if export_evolution_command(args, "auditoria"):
        return
    from truss_research.evaluacion.auditoria_estudio import AuditConfig, audit_study
    from truss_research.optimizacion.estudio_estado import load_study_config

    data_dir = args.datos or load_study_config(args.estudio).data_dir
    audit_study(
        AuditConfig(
            args.estudio,
            data_dir,
            args.destino or next_export(args.estudio, "auditoria"),
        )
    )


def run_notebooks_command(args: argparse.Namespace) -> None:
    if export_evolution_command(args, "notebooks"):
        return
    from truss_research.evaluacion.presentacion import export_notebooks, resolve_study

    study = resolve_study(args.estudio)
    destination = args.destino or next_export(study, "presentacion")
    print(export_notebooks(study, destination))


def run_sensitivity_command(args: argparse.Namespace) -> None:
    from truss_research.optimizacion.estudio_estado import load_study_config

    if load_study_config(args.estudio).experiment == "evolution":
        raise ValueError(
            "La busqueda A/B se ejecuta mediante make study y make final; esta etapa pertenece al protocolo de catalogo anterior"
        )
    from truss_research.evaluacion.sensibilidad import run_sensitivity

    print(
        run_sensitivity(
            args.estudio, args.destino or next_export(args.estudio, "sensibilidad")
        )
    )


def run_benchmark_command(args: argparse.Namespace) -> None:
    if export_evolution_command(args, "benchmark"):
        return
    from truss_research.evaluacion.benchmark import run_benchmark

    print(
        run_benchmark(
            args.estudio, args.destino or next_export(args.estudio, "benchmark")
        )
    )


def run_learning_command(args: argparse.Namespace) -> None:
    from truss_research.optimizacion.estudio_estado import load_study_config

    if load_study_config(args.estudio).experiment == "evolution":
        raise ValueError(
            "La busqueda A/B se ejecuta mediante make study y make final; esta etapa pertenece al protocolo de catalogo anterior"
        )
    from truss_research.evaluacion.curvas_aprendizaje import run_learning_curves

    print(
        run_learning_curves(
            args.estudio, args.destino or next_export(args.estudio, "curvas")
        )
    )


def run_tabpfn_command(args: argparse.Namespace) -> None:
    from truss_research.datos.schemas import DESIGN_DESTINATION
    from truss_research.evaluacion.piloto_tabpfn import (
        download_weights,
        pilot_accelerator,
        run_pilot,
    )

    if args.command == "probar-tabpfn":
        selected = pilot_accelerator(getattr(args, "accelerator", None))
        args = argparse.Namespace(**{**vars(args), "accelerator": selected})
        logging.getLogger(__name__).info("Dispositivo del piloto TabPFN: %s", selected)

    config = command_protocol(
        args,
        args.config,
        (DEFAULT_STUDY.resolve(), (args.datos or DESIGN_DESTINATION).resolve()),
    )
    if args.command == "preparar-tabpfn":
        print(download_weights(config.tabpfn))
        return
    root = config.tabpfn.checkpoint.parent / "pilots"
    numbers = [
        int(path.name.removeprefix("pilot_"))
        for path in root.glob("pilot_*")
        if path.name.removeprefix("pilot_").isdigit()
    ]
    print(run_pilot(config, root / f"pilot_{max(numbers, default=0) + 1:03d}"))


def run_adaptive_command(args: argparse.Namespace) -> None:
    from truss_research.optimizacion.estudio_estado import load_study_config

    if load_study_config(args.estudio).experiment == "evolution":
        raise ValueError(
            "La busqueda A/B se ejecuta mediante make study y make final; esta etapa pertenece al protocolo de catalogo anterior"
        )
    from truss_research.evaluacion.seleccion_adaptativa import (
        AdaptiveParameters,
        run_adaptive_selection,
    )
    from truss_research.optimizacion.estudio_estado import load_study_config

    config = load_study_config(args.estudio)

    print(
        run_adaptive_selection(
            args.estudio,
            args.destino or next_export(args.estudio, "adaptativa"),
            AdaptiveParameters(
                beta=config.adaptive.beta,
                online_residual=config.adaptive.online_residual,
                cpu_workers=args.cpu_workers,
            ),
        )
    )


def run_jupyter_command(args: argparse.Namespace) -> None:
    from truss_research.evaluacion.presentacion import resolve_study
    from truss_research.optimizacion.estudio_estado import load_study_config

    study = resolve_study(args.estudio or DEFAULT_STUDY)
    notebook_directory = Path("notebooks")
    if load_study_config(study).experiment == "evolution":
        candidates = sorted(results_root(study).glob("presentacion_[0-9][0-9][0-9]"))
        if not candidates:
            raise ValueError(
                "Ejecutar make notebooks antes de abrir la presentación evolutiva"
            )
        notebook_directory = candidates[-1]
    environment = {**os.environ, "TRUSS_STUDY": str(study)}
    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "lab", str(notebook_directory)],
        env=environment,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("JupyterLab terminó con un error")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experimentos de predicción y selección de cerchas. Usar make para el recorrido habitual."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preparar-tabpfn", "probar-tabpfn"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL)
        command.add_argument("--datos", type=Path)
        command.set_defaults(handler=run_tabpfn_command)
    prepare = commands.add_parser(
        "preparar-datos",
        help="Preparar diseños únicos o reutilizar una preparación compatible",
    )
    prepare.add_argument(
        "--origen", type=Path, default=Path("data/raw/240214-C1-20m.xlsx")
    )
    prepare.add_argument("--destino", type=Path)
    prepare.add_argument("--calibration-fraction", type=float)
    prepare.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL)
    prepare.set_defaults(handler=run_preparation)
    smoke = commands.add_parser(
        "comprobar-estudio",
        help="Comprobación completa en un destino compatible o nuevo",
    )
    smoke.add_argument("--datos", type=Path)
    smoke.add_argument("--destino", type=Path)
    smoke.add_argument("--config", type=Path)
    smoke.add_argument(
        "--solo-dummy",
        action="store_true",
        help="Control preliminar mínimo; el estudio posterior conserva su perfil completo",
    )
    smoke.set_defaults(handler=run_smoke_command)
    study = commands.add_parser(
        "ejecutar-estudio",
        help="Entrenar y comparar con validación; reanudar automáticamente",
    )
    study.add_argument("--destino", type=Path, default=DEFAULT_STUDY)
    study.add_argument("--datos", type=Path)
    study.add_argument("--config", type=Path, help="Ruta a un protocolo JSON")
    study.add_argument(
        "--fase",
        choices=("predictores", "validacion", "final", "todo"),
        default="validacion",
    )
    study.add_argument("--reanudar", action="store_true")
    study.add_argument("--plan", action="store_true")
    study.set_defaults(handler=run_study_command)
    final = commands.add_parser(
        "evaluar-final",
        help="Congelar decisiones y evaluar prueba; requiere validación completa",
    )
    final.add_argument("--estudio", dest="destino", type=Path, default=DEFAULT_STUDY)
    final.set_defaults(
        handler=run_study_command,
        fase="final",
        reanudar=True,
        datos=None,
        config=None,
        plan=False,
    )
    for command in (study, smoke, final):
        command.add_argument(
            "--cpu-workers",
            type=int,
            help="Máximo de procesos CPU (entero positivo); omitido conserva JSON o estudio guardado",
        )
    handlers = {
        "resumir-estudio": run_summary_command,
        "auditar-estudio": run_audit_command,
        "presentar-notebooks": run_notebooks_command,
        "analizar-sensibilidad": run_sensitivity_command,
        "evaluar-curvas": run_learning_command,
        "medir-costos": run_benchmark_command,
        "evaluar-seleccion-adaptativa": run_adaptive_command,
        "abrir-notebooks": run_jupyter_command,
    }
    for name, handler in handlers.items():
        command = commands.add_parser(name)
        command.add_argument(
            "--estudio",
            type=Path,
            default=None
            if name in ("presentar-notebooks", "abrir-notebooks")
            else DEFAULT_STUDY,
        )
        command.set_defaults(handler=handler)
    for name in (
        "resumir-estudio",
        "auditar-estudio",
        "presentar-notebooks",
        "analizar-sensibilidad",
        "evaluar-curvas",
        "medir-costos",
        "evaluar-seleccion-adaptativa",
    ):
        commands.choices[name].add_argument("--destino", type=Path)
    commands.choices["auditar-estudio"].add_argument("--datos", type=Path)
    commands.choices["evaluar-seleccion-adaptativa"].add_argument(
        "--cpu-workers",
        type=int,
        help="Máximo de procesos para adaptive; omitido hereda el estudio, 1 ejecuta secuencialmente",
    )
    for command in (
        prepare,
        smoke,
        study,
        final,
        commands.choices["preparar-tabpfn"],
    ):
        command.add_argument(
            "--accelerator",
            choices=("cpu", "cuda"),
            help="Dispositivo de TabPFN; omitido conserva el JSON o el estudio guardado",
        )
    commands.choices["probar-tabpfn"].add_argument(
        "--accelerator",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto prefiere CUDA disponible y usa CPU en caso contrario",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    try:
        from truss_research.evaluacion.registro import execution_log, next_log

        if getattr(args, "plan", False):
            args.handler(args)
            return 0
        selected = getattr(args, "estudio", None) or getattr(args, "destino", None)
        label = selected.name if isinstance(selected, Path) else args.command
        root = (
            results_root(selected)
            if isinstance(selected, Path)
            else Path("reports") / label
        )
        log = next_log(root / "logs", args.command)
        with execution_log(
            log, args.command, {"arguments": sys.argv[1:], "cwd": str(Path.cwd())}
        ):
            args.handler(args)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        logging.getLogger(__name__).error(
            "Preparación o experimento interrumpido: %s", error
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
