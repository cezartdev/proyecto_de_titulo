"""Contención temporal y planificación del recorrido, sin campaña científica."""

import argparse
import json
import os
import sys
from pathlib import Path
from time import monotonic

import psutil
import pytest

from truss_research.evaluacion.recorrido import (
    STAGES,
    monitored,
    pipeline_stages,
    supervise,
)


@pytest.fixture
def campaign_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"synthetic-source-for-orchestration-only")
    monkeypatch.setattr("truss_research.evaluacion.campana.DEFAULT_SOURCE", source)
    return source


@pytest.mark.parametrize("accelerator", ["cpu", "cuda"])
def test_profiles_share_locked_root_environment(
    tmp_path: Path, accelerator: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def capture(command: list[str], context: tuple[float | None, Path]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr("truss_research.evaluacion.recorrido.monitored", capture)
    args = argparse.Namespace(
        mode="check",
        accelerator=accelerator,
        estudio=tmp_path / "study",
        config=Path("config/smoke.json"),
        cpu_workers=2,
        max_seconds=600,
        plan=False,
    )
    assert supervise(args) == 0
    assert commands[0] == ["uv", "sync", "--locked"]
    assert commands[1][:4] == ["uv", "run", "--locked", "python"]
    assert "--project" not in commands[1] and "--extra" not in commands[1]
    assert commands[1][commands[1].index("--accelerator") + 1] == accelerator


def test_deadline_stops_descendants(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.json"
    program = (
        "import json,subprocess,sys,time;from pathlib import Path;"
        f"p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session={os.name != 'nt'});"
        f"Path({str(pid_file)!r}).write_text(json.dumps({{'pid':p.pid}}));"
        "time.sleep(60)"
    )
    start = monotonic()
    with pytest.raises(TimeoutError, match="MAX_SECONDS"):
        monitored(
            [sys.executable, "-c", program], (start + 2.0, tmp_path / "process.log")
        )
    assert monotonic() - start < 7.0
    assert pid_file.is_file()
    pid = json.loads(pid_file.read_text())["pid"]
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )
    assert psutil.pid_exists(os.getpid())


@pytest.mark.parametrize("accelerator", ["cpu", "cuda"])
def test_campaign_propagates_device_with_shared_protocols(
    tmp_path: Path,
    accelerator: str,
) -> None:
    from truss_research.evaluacion.campana import Campaign

    campaign = Campaign(
        argparse.Namespace(
            mode="check",
            estudio=tmp_path / "study",
            config=Path("config/smoke.json"),
            cpu_workers=2,
            accelerator=accelerator,
        )
    )
    for stage in (
        "prepare",
        "tabpfn-weights",
        "tabpfn-pilot",
        "plan",
        "study",
        "smoke",
    ):
        command = campaign.command(stage)
        assert command[command.index("--accelerator") + 1] == accelerator
        assert Path(command[command.index("--config") + 1]) == Path("config/smoke.json")
    campaign.args.mode = "full"
    command = campaign.command("smoke")
    assert Path(command[command.index("--config") + 1]) == Path("config/smoke.json")
    assert command[command.index("--accelerator") + 1] == accelerator


def test_plan_creates_no_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        mode="check",
        estudio=tmp_path / "study",
        config=Path("config/smoke.json"),
        cpu_workers=2,
        max_seconds=600,
        plan=True,
    )
    assert supervise(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["stages"] == list(STAGES)
    assert result["max_seconds"] == 600
    assert not list(tmp_path.iterdir())


def test_synthetic_flow_source_matches_contract(tmp_path: Path) -> None:
    from truss_research.cerchas.schemas import INPUT_COLUMNS
    from truss_research.datos.preparacion import load_raw_table, normalize_columns
    from truss_research.datos.schemas import DatasetSpec
    from truss_research.datos.validacion import (
        add_identifiers,
        validate_structure,
        validate_values,
    )
    from truss_research.evaluacion.datos_comprobacion import ensure_flow_source

    source = ensure_flow_source(tmp_path / "synthetic.xlsx")
    table = add_identifiers(validate_values(normalize_columns(load_raw_table(source))))
    validate_structure(table, DatasetSpec())
    assert len(table) == 5000 and len(table.drop_duplicates(list(INPUT_COLUMNS))) == 500


def test_resume_checks_complete_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.datos.preparacion import source_digest
    from truss_research.evaluacion.campana import Campaign

    args = argparse.Namespace(
        mode="check",
        estudio=tmp_path / "study",
        config=Path("config/smoke.json"),
        cpu_workers=2,
    )
    flow = Campaign(args)
    flow.root.mkdir(parents=True)
    output = tmp_path / "proof.csv"
    output.write_text("complete")
    calls = []
    monkeypatch.setattr(
        "truss_research.evaluacion.campana.subprocess.run",
        lambda command, **options: calls.append(command),
    )
    monkeypatch.setattr(
        flow, "outputs", lambda stage: {str(output): source_digest(output)}
    )
    flow.execute("test")
    flow.execute("test")
    assert len(calls) == 1
    output.write_text("changed")
    with pytest.raises(ValueError, match="salida completa"):
        flow.execute("test")
    assert len(calls) == 1


def test_failed_stage_stops_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion.campana import Campaign

    flow = Campaign(
        argparse.Namespace(
            mode="check",
            estudio=tmp_path / "study",
            config=Path("config/smoke.json"),
            cpu_workers=2,
        )
    )
    called = []

    def fail(stage: str) -> None:
        called.append(stage)
        if stage == "final":
            raise RuntimeError("failure")

    monkeypatch.setattr(flow, "execute", fail)
    with pytest.raises(RuntimeError, match="failure"):
        flow.run()
    assert called == list(STAGES[1 : STAGES.index("final") + 1])


def test_timeout_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import truss_research.evaluacion.recorrido as runner

    def expire(command: list[str], context: tuple[float, Path]) -> int:
        raise TimeoutError("Se agotó MAX_SECONDS")

    monkeypatch.setattr(runner, "monitored", expire)
    args = argparse.Namespace(
        mode="check",
        estudio=tmp_path / "study",
        config=Path("config/smoke.json"),
        cpu_workers=2,
        max_seconds=1,
        plan=False,
    )
    assert runner.supervise(args) == 124
    saved = json.loads(
        next(
            (runner.report_root(args.estudio) / "logs").glob("run_all_*.json")
        ).read_text(encoding="utf-8")
    )
    assert saved["status"] == "timed_out" and saved["exit_code"] == 124


def test_posix_descendant_graph_preserves_other_trees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import truss_research.evaluacion.recorrido as runner

    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda command, **options: "1 0\n10 1\n20 10\n30 20\n40 1\n",
    )
    assert runner.posix_descendants(10, 1.0) == [20, 30]


def test_resume_inherits_workers_when_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import truss_research.evaluacion.campana as workflow
    from truss_research.optimizacion.estudio_estado import StudyConfig
    from truss_research.optimizacion.protocolo import load_protocol

    study = tmp_path / "study"
    study.mkdir()
    (study / "study.json").write_text("{}")
    config_path = Path("config/smoke.json")
    saved = load_protocol(config_path, (study, tmp_path / "data"))
    saved = replace(saved, execution=replace(saved.execution, cpu_workers=6))
    monkeypatch.setattr(workflow, "load_study_config", lambda path: saved)

    def validate(config: StudyConfig, resume: bool) -> None:
        if config.execution != saved.execution:
            raise ValueError("workers incompatibles")

    monkeypatch.setattr(workflow, "open_session", validate)
    args = argparse.Namespace(
        mode="check", estudio=study, config=config_path, cpu_workers=None
    )
    assert workflow.Campaign(args).config.execution.cpu_workers == 6
    args.cpu_workers = 2
    with pytest.raises(ValueError, match="workers incompatibles"):
        workflow.Campaign(args)


def test_interpolation_flag_reaches_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def capture(command: list[str], context: tuple[float | None, Path]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr("truss_research.evaluacion.recorrido.monitored", capture)
    args = argparse.Namespace(
        mode="full",
        accelerator="cuda",
        estudio=tmp_path / "study",
        config=Path("config/study.json"),
        cpu_workers=2,
        max_seconds=None,
        plan=False,
        interpolation=True,
        interpolation_dest=tmp_path / "sensitivity",
    )
    assert supervise(args) == 0
    command = commands[1]
    assert "--interpolation" in command
    assert command[command.index("--interpolation-dest") + 1] == str(
        args.interpolation_dest
    )


def test_interpolation_plan_has_separate_destination_and_creates_no_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        mode="full",
        estudio=tmp_path / "study",
        config=Path("config/study.json"),
        max_seconds=None,
        plan=True,
        interpolation=True,
        interpolation_dest=None,
    )
    assert supervise(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["stages"] == list(pipeline_stages(True))
    assert plan["stages"][-1] == "interpolation"
    assert Path(plan["interpolation_destination"]) == tmp_path / "study_interpolacion"
    assert not list(tmp_path.iterdir())
    args.mode = "check"
    with pytest.raises(ValueError, match="RUN_MODE=full"):
        supervise(args)


def test_campaign_interpolation_command_and_stage_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign_source: Path
) -> None:
    from truss_research.evaluacion.campana import Campaign

    args = argparse.Namespace(
        mode="full",
        estudio=tmp_path / "study",
        config=Path("config/study.json"),
        cpu_workers=2,
        accelerator="cuda",
        interpolation=True,
        interpolation_dest=None,
    )
    campaign = Campaign(args)
    command = campaign.command("interpolation")
    assert command[2] == "truss_research.evaluacion.interpolacion"
    assert command[command.index("--accelerator") + 1] == "cuda"
    assert Path(command[command.index("--source") + 1]) == campaign.source
    assert (
        Path(command[command.index("--destino") + 1])
        == tmp_path / "study_interpolacion"
    )
    called: list[str] = []
    campaign.root.mkdir(parents=True)
    monkeypatch.setattr(campaign, "execute", called.append)
    campaign.run()
    assert called == list(pipeline_stages(True)[1:])
    assert campaign.records["technical_complete"] is True


def test_interpolation_reuses_only_intact_compatible_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign_source: Path
) -> None:
    from dataclasses import asdict

    from truss_research.datos.preparacion import source_digest
    from truss_research.evaluacion.campana import Campaign
    from truss_research.optimizacion.estudio_estado import (
        artifact_signatures,
        source_signature,
    )
    from truss_research.optimizacion.protocolo import load_protocol
    from truss_research.surrogates.experimentos import write_json

    args = argparse.Namespace(
        mode="full",
        estudio=tmp_path / "study",
        config=Path("config/study.json"),
        cpu_workers=2,
        accelerator="cpu",
        interpolation=True,
        interpolation_dest=tmp_path / "sensitivity",
    )
    flow = Campaign(args)
    destination = flow.interpolation_destination
    (destination / "data").mkdir(parents=True)
    with pytest.raises(ValueError, match="incompleta"):
        Campaign(args)
    config = load_protocol(args.config, (destination, destination / "data"), "cpu")
    write_json(destination / "config.json", asdict(config))
    write_json(
        destination / "data/manifest.json",
        {"source_sha256": source_digest(flow.source)},
    )
    proof = destination / "report.csv"
    proof.write_text("complete")
    write_json(
        destination / "complete.json",
        {
            "files": artifact_signatures(destination),
            "source_signature": source_signature(),
        },
    )
    resumed = Campaign(args)
    resumed.root.mkdir(parents=True)

    def unexpected_execution(command: list[str], check: bool) -> None:
        raise AssertionError(
            "No se debe ejecutar un entrenamiento completo al reanudar"
        )

    monkeypatch.setattr(
        "truss_research.evaluacion.campana.subprocess.run", unexpected_execution
    )
    resumed.execute("interpolation")
    resumed.execute("interpolation")
    assert resumed.records["stages"]["interpolation"]["status"] == "complete"
    proof.write_text("changed")
    with pytest.raises(ValueError, match="salida completa"):
        Campaign(args)


def test_interpolation_destination_cannot_overlap_main_study(tmp_path: Path) -> None:
    from truss_research.evaluacion.campana import Campaign

    args = argparse.Namespace(
        mode="full",
        estudio=tmp_path / "study",
        config=Path("config/study.json"),
        cpu_workers=2,
        interpolation=True,
        interpolation_dest=tmp_path / "study/interpolation",
    )
    with pytest.raises(ValueError, match="destino separado"):
        Campaign(args)
