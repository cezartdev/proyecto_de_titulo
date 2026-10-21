"""Coordina etapas existentes y verifica sus recibos antes de reutilizarlas."""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.schemas import DEFAULT_SOURCE, DESIGN_DESTINATION
from truss_research.evaluacion.recorrido import (
    pipeline_stages,
    report_root,
    save_record,
)
from truss_research.optimizacion.compatibilidad_ejecucion import compatible_sources
from truss_research.optimizacion.estudio_estado import (
    artifact_signatures,
    environment_versions,
    load_study_config,
    open_session,
    source_signature,
    verify_task,
)
from truss_research.optimizacion.protocolo import load_protocol

CLI_STAGES = {
    "adaptive": "evaluar-seleccion-adaptativa",
    "sensitivity": "analizar-sensibilidad",
    "learning-curves": "evaluar-curvas",
    "benchmark": "medir-costos",
    "report": "resumir-estudio",
    "audit": "auditar-estudio",
    "notebooks": "presentar-notebooks",
}
EXPORT_NAMES = {
    "adaptive": "adaptativa",
    "sensitivity": "sensibilidad",
    "learning-curves": "curvas",
    "benchmark": "benchmark",
    "report": "informe",
    "audit": "auditoria",
    "notebooks": "presentacion",
}
CHECK_TESTS = (
    "tests/test_run_all.py::test_synthetic_flow_source_matches_contract",
    "tests/test_tabpfn.py::test_adapter_and_signed_states",
    "tests/test_tabpfn.py::test_fixed_profile_and_no_hpo",
    "tests/test_run_all.py::test_deadline_stops_descendants",
)


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stages = pipeline_stages(getattr(args, "interpolation", False))
        self.interpolation_destination = (
            getattr(args, "interpolation_dest", None)
            or args.estudio.with_name(args.estudio.name + "_interpolacion")
        ).resolve()
        if "interpolation" in self.stages and args.mode != "full":
            raise ValueError("La interpolación de run-all requiere RUN_MODE=full")
        self.root = report_root(args.estudio)
        self.receipt = self.root / "run_all.json"
        self.records = (
            read_artifact(self.receipt) if self.receipt.exists() else {"stages": {}}
        )
        if not isinstance(self.records.get("stages"), dict):
            raise ValueError("El recibo del recorrido tiene un formato inválido")
        if self.records.get("mode", args.mode) != args.mode:
            raise ValueError(
                "No mezclar modos de comprobación y campaña en el mismo destino"
            )
        self.data = (
            self.root / "inputs/processed"
            if args.mode == "check"
            else DESIGN_DESTINATION.resolve()
        )
        self.source = (
            self.root / "inputs/synthetic.xlsx"
            if args.mode == "check"
            else DEFAULT_SOURCE.resolve()
        )
        self.config = load_protocol(
            args.config, (args.estudio, self.data), getattr(args, "accelerator", None)
        )
        if "interpolation" in self.stages:
            self.check_interpolation_destination()
        if args.mode == "full" and not self.source.is_file():
            raise FileNotFoundError(f"Falta el Excel autorizado: {self.source}")
        if args.mode == "check" and self.config.label != "comprobacion":
            raise ValueError("RUN_MODE=check requiere un perfil de comprobación")
        if args.mode == "full" and self.config.label != "posterior":
            raise ValueError("RUN_MODE=full requiere un perfil de campaña completa")
        if args.cpu_workers is not None:
            from dataclasses import replace

            self.config = replace(
                self.config,
                execution=replace(self.config.execution, cpu_workers=args.cpu_workers),
            )
        if (args.estudio / "study.json").is_file():
            if args.cpu_workers is None:
                from dataclasses import replace

                saved = load_study_config(args.estudio)
                self.config = replace(
                    self.config,
                    execution=replace(
                        self.config.execution, cpu_workers=saved.execution.cpu_workers
                    ),
                )
            open_session(self.config, True)

    def check_interpolation_destination(self) -> None:
        destination = self.interpolation_destination
        protected = (self.args.estudio.resolve(), self.data.resolve())
        if any(
            destination.is_relative_to(path) or path.is_relative_to(destination)
            for path in protected
        ):
            raise ValueError(
                "La interpolación requiere un destino separado del estudio y sus datos"
            )
        if destination.exists():
            from truss_research.evaluacion.interpolacion import verify_interpolation

            config = load_protocol(
                self.args.config,
                (destination, destination / "data"),
                self.config.tabpfn.device,
            )
            verify_interpolation(config, self.source)

    def inputs(self, stage: str, source: str | None = None) -> str:
        from truss_research.datos.procedencia import confirmations_digest

        extra = {}
        if stage == "test":
            extra = {
                p.as_posix(): source_digest(p) for p in Path("tests").rglob("*.py")
            }
        if stage == "interpolation":
            extra["destination"] = str(self.interpolation_destination)
        if stage in ("report", "audit", "notebooks"):
            extra["physical"] = confirmations_digest()
        if stage == "notebooks":
            extra.update(
                {
                    p.as_posix(): source_digest(p)
                    for p in Path("notebooks").glob("*.ipynb")
                }
            )
        upstream = {
            name: row.get("outputs", {})
            for name, row in self.records["stages"].items()
            if name in self.stages[1 : self.stages.index(stage)]
            and name not in ("check", "test", "plan")
        }
        payload = {
            "source": source if source is not None else source_signature(),
            "versions": environment_versions(),
            "config": asdict(self.config),
            "mode": self.args.mode,
            "stage": stage,
            "upstream": upstream,
            "extra": extra,
        }
        if self.source.is_file():
            payload["raw_sha256"] = source_digest(self.source)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    def valid(self, stage: str) -> bool:
        saved = self.records["stages"].get(stage)
        if not saved or saved.get("status") != "complete":
            return False
        expected = saved.get("outputs", {})
        if any(
            not Path(path).is_file() or source_digest(Path(path)) != digest
            for path, digest in expected.items()
        ):
            raise ValueError(
                f"Cambió una salida completa de {stage}; se conserva para diagnóstico"
            )
        acceptable = (self.inputs(stage),)
        if saved.get("inputs") not in acceptable:
            acceptable += tuple(
                self.inputs(stage, previous)
                for previous in compatible_sources(source_signature())
            )
        if saved.get("inputs") not in acceptable:
            return False
        if stage in ("study", "final"):
            if self.config.experiment == "evolution":
                from truss_research.evaluacion.busqueda_evolutiva import (
                    verify_evaluation,
                )

                phase = "validation" if stage == "study" else "test"
                verify_evaluation(self.args.estudio / "evolution" / phase)
                return True
            from truss_research.optimizacion.estudio import requires_execution

            session = open_session(self.config, True)
            if requires_execution(
                session, "validacion" if stage == "study" else "final"
            ):
                return False
            for task in (*session.tasks.values(), *session.preparation_tasks.values()):
                verify_task(task)
        return True

    def command(self, stage: str) -> list[str]:
        if stage == "interpolation":
            return [
                sys.executable,
                "-m",
                "truss_research.evaluacion.interpolacion",
                "--config",
                str(self.args.config),
                "--destino",
                str(self.interpolation_destination),
                "--source",
                str(self.source),
                "--accelerator",
                self.config.tabpfn.device,
            ]
        base = [sys.executable, "-m", "truss_research"]
        config = [
            "--config",
            str(self.args.config),
            "--accelerator",
            self.config.tabpfn.device,
        ]
        workers = ["--cpu-workers", str(self.config.execution.cpu_workers)]
        if stage == "check":
            return [sys.executable, "-m", "ruff", "check", "."]
        if stage == "test":
            return [
                sys.executable,
                "tests/ejecutar.py",
                *(CHECK_TESTS if self.args.mode == "check" else ()),
            ]
        if stage == "prepare":
            return [
                *base,
                "preparar-datos",
                *config,
                "--origen",
                str(self.source),
                "--destino",
                str(self.data),
            ]
        if stage in ("tabpfn-weights", "tabpfn-pilot"):
            return [
                *base,
                "preparar-tabpfn" if stage.endswith("weights") else "probar-tabpfn",
                *config,
                "--datos",
                str(self.data),
            ]
        if stage in ("plan", "study"):
            return [
                *base,
                "ejecutar-estudio",
                "--destino",
                str(self.args.estudio),
                "--datos",
                str(self.data),
                *config,
                *workers,
                *(["--plan"] if stage == "plan" else ()),
            ]
        if stage == "smoke":
            return [
                *base,
                "comprobar-estudio",
                *workers,
                "--datos",
                str(self.data),
                "--config",
                str(
                    self.args.config
                    if self.args.mode == "check"
                    else Path("config/smoke.json")
                ),
                "--accelerator",
                self.config.tabpfn.device,
                "--destino",
                str(self.args.estudio.with_name(self.args.estudio.name + "_smoke")),
                *(["--solo-dummy"] if self.args.mode == "check" else ()),
            ]
        if stage == "final":
            return [
                *base,
                "evaluar-final",
                "--estudio",
                str(self.args.estudio),
                *workers,
            ]
        return [
            *base,
            CLI_STAGES[stage],
            "--estudio",
            str(self.args.estudio),
            *(workers if stage == "adaptive" else ()),
        ]

    def outputs(self, stage: str) -> dict[str, str]:
        if stage == "interpolation":
            from truss_research.evaluacion.interpolacion import verify_interpolation

            destination = self.interpolation_destination
            config = load_protocol(
                self.args.config,
                (destination, destination / "data"),
                self.config.tabpfn.device,
            )
            verify_interpolation(config, self.source)
            return {
                str(self.interpolation_destination / name): digest
                for name, digest in artifact_signatures(
                    self.interpolation_destination
                ).items()
            }
        if stage in ("check", "test", "plan"):
            return {}
        if stage == "prepare":
            return {
                str(self.data / name): digest
                for name, digest in artifact_signatures(self.data).items()
            }
        if stage == "tabpfn-weights":
            return {
                str(self.config.tabpfn.checkpoint): source_digest(
                    self.config.tabpfn.checkpoint
                )
            }
        if stage == "tabpfn-pilot":
            from truss_research.evaluacion.piloto_tabpfn import require_pilot

            require_pilot(self.config)
            return {
                str(path): source_digest(path)
                for path in self.config.tabpfn.checkpoint.parent.glob(
                    "pilots/pilot_*/pilot.json"
                )
            }
        if stage in ("study", "final", "smoke"):
            from truss_research.optimizacion.estudio_estado import read_study

            root = (
                self.args.estudio.with_name(self.args.estudio.name + "_smoke")
                if stage == "smoke"
                else self.args.estudio
            )
            if self.config.experiment == "evolution":
                evolution = root / "evolution"
                return {
                    str(evolution / name): digest
                    for name, digest in artifact_signatures(evolution).items()
                    if stage not in ("study", "smoke")
                    or (not name.startswith("test/") and name != "closed.json")
                } | (
                    {str(root / "freeze.json"): source_digest(root / "freeze.json")}
                    if stage not in ("study", "smoke")
                    else {}
                )
            state = read_study(root)
            rows = (*state["tasks"], *state.get("preparation_tasks", []))
            outputs = {
                str(Path(row["directory"]) / name): digest
                for row in rows
                if stage != "study" or not row["key"].startswith("test/")
                for name, digest in row["signatures"].items()
            }
            if stage != "study":
                outputs[str(root / "freeze.json")] = source_digest(root / "freeze.json")
            return outputs
        candidates = sorted(self.root.glob(EXPORT_NAMES[stage] + "_[0-9][0-9][0-9]"))
        if not candidates:
            raise ValueError(f"No se produjo la exportación de {stage}")
        root = candidates[-1]
        if (
            stage == "audit"
            and read_artifact(root / "technical_closure.json").get("technical_complete")
            is not True
        ):
            raise ValueError("La auditoría no completó el cierre técnico")
        if stage == "notebooks" and (
            len(list(root.glob("*.html"))) != 4 or len(list(root.glob("*.ipynb"))) != 4
        ):
            raise ValueError(
                "El cierre requiere cuatro notebooks de resultados y cuatro HTML"
            )
        return {
            str(root / name): digest
            for name, digest in artifact_signatures(root).items()
        }

    def execute(self, stage: str) -> None:
        if (
            stage in ("tabpfn-weights", "tabpfn-pilot")
            and not self.config.tabpfn_enabled
        ):
            return
        if stage != "check" and self.valid(stage):
            print(f"run-all: {stage} reutilizado", flush=True)
            return
        if stage == "interpolation" and self.interpolation_destination.exists():
            self.check_interpolation_destination()
            self.complete(stage, 0.0)
            print("run-all: interpolación completa compatible reutilizada", flush=True)
            return
        if stage == "tabpfn-pilot":
            from truss_research.evaluacion.piloto_tabpfn import require_pilot

            try:
                require_pilot(self.config)
            except ValueError:
                pass
            else:
                self.complete(stage, 0.0)
                print("run-all: piloto compatible reutilizado", flush=True)
                return
        if stage == "prepare" and self.args.mode == "check":
            from truss_research.evaluacion.datos_comprobacion import ensure_flow_source

            ensure_flow_source(self.source)
        print(f"run-all: iniciando {stage}", flush=True)
        self.records.update(status="running", current_stage=stage, mode=self.args.mode)
        self.records["stages"][stage] = {"status": "running"}
        save_record(self.receipt, self.records)
        start = perf_counter()
        try:
            subprocess.run(self.command(stage), check=True)
            if stage == "check":
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ruff",
                        "format",
                        "--check",
                        ".",
                    ],
                    check=True,
                )
            self.complete(stage, perf_counter() - start)
        except BaseException as error:
            self.records["stages"][stage].update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                wall_seconds=perf_counter() - start,
            )
            self.records["status"] = "failed"
            save_record(self.receipt, self.records)
            raise

    def complete(self, stage: str, seconds: float) -> None:
        self.records["stages"][stage] = {
            "status": "complete",
            "wall_seconds": seconds,
            "inputs": self.inputs(stage),
            "outputs": self.outputs(stage),
        }
        save_record(self.receipt, self.records)

    def run(self) -> None:
        for stage in self.stages[1:]:
            if self.config.experiment == "evolution" and stage in (
                "adaptive",
                "sensitivity",
                "learning-curves",
            ):
                continue
            self.execute(stage)
        self.records.update(
            status="complete",
            current_stage=None,
            scientific_classification="exploratory",
            technical_complete=True,
        )
        save_record(self.receipt, self.records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "check"), required=True)
    parser.add_argument("--estudio", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cpu-workers", type=int)
    parser.add_argument("--accelerator", choices=("cpu", "cuda"))
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--interpolation-dest", type=Path)
    Campaign(parser.parse_args()).run()


if __name__ == "__main__":
    main()
