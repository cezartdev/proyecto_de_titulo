"""Recorrido público por etapas, sin consultar el Excel privado."""

import json
import sys
from pathlib import Path

import pytest

from truss_research.__main__ import main
from truss_research.optimizacion.estudio_estado import StudyConfig


def test_cli_final_requires_existing_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = tmp_path / "missing"
    monkeypatch.setattr(
        sys, "argv", ["truss_research", "evaluar-final", "--estudio", str(study)]
    )
    assert main() == 1
    assert not study.exists()


@pytest.mark.parametrize("workers", [None, "1", "6", "12"])
def test_cli_cpu_workers_plan_limits_to_detected_cpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: str | None
) -> None:
    import truss_research.__main__ as cli
    from truss_research.optimizacion.protocolo import load_protocol

    output: list[str] = []
    monkeypatch.setattr(cli, "print", output.append, raising=False)
    target = tmp_path / "absent"
    expected = load_protocol(
        Path("config/study.json"), (target, tmp_path)
    ).execution.cpu_workers
    arguments = [
        "truss_research",
        "ejecutar-estudio",
        "--destino",
        str(target),
        "--plan",
    ]
    if workers is not None:
        arguments.extend(["--cpu-workers", workers])
        expected = int(workers)
    monkeypatch.setattr(
        "truss_research.optimizacion.ejecucion_cpu.os.cpu_count", lambda: 4
    )
    monkeypatch.setattr(sys, "argv", arguments)
    assert main() == 0
    plan = json.loads(output[0])
    assert plan["execution"]["cpu_workers"] == expected
    assert plan["execution"]["workers"] == min(expected, 4)
    assert plan["execution"]["numeric_threads_per_worker"] == 1
    assert not target.exists()


@pytest.mark.parametrize("workers", ["0", "-1", "13", "1000", "1.5", "texto"])
def test_cli_cpu_workers_rejects_invalid_values_before_creating_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: str
) -> None:
    target = tmp_path / "absent"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "truss_research",
            "ejecutar-estudio",
            "--destino",
            str(target),
            "--cpu-workers",
            workers,
        ],
    )
    if workers in ("1.5", "texto"):
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2
    else:
        assert main() == 1
    assert not target.exists()


@pytest.mark.parametrize(
    "selection", [(None, "cpu"), ("1", "cpu"), (None, "cuda"), ("1", "cuda")]
)
def test_smoke_accepts_cpu_workers_without_editing_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: tuple[str | None, str],
) -> None:
    import truss_research.__main__ as cli

    workers, accelerator = selection
    protocol = Path("config/smoke.json").resolve()
    original = protocol.read_bytes()
    received: list[StudyConfig] = []

    def capture(config: StudyConfig, phase: str, resume: bool) -> Path:
        received.append(config)
        assert phase == "validacion" and not resume
        return config.output_dir

    monkeypatch.setattr(cli, "SMOKE_PROTOCOL", protocol)
    monkeypatch.setattr("truss_research.optimizacion.estudio.run_study", capture)
    monkeypatch.chdir(tmp_path)
    arguments = ["truss_research", "comprobar-estudio", "--accelerator", accelerator]
    if workers is not None:
        arguments.extend(["--cpu-workers", workers])
    monkeypatch.setattr(sys, "argv", arguments)
    assert main() == 0
    assert received[0].execution.cpu_workers == (int(workers) if workers else 2)
    assert received[0].tabpfn.device == accelerator
    assert received[0].tabpfn_enabled
    assert protocol.read_bytes() == original
