"""Integridad de bloques, caché y control de ejecución sin entrenar estimadores."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from truss_research.cerchas.schemas import OUTPUT_COLUMNS
from truss_research.datos.artefactos import read_artifact
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import load_design_split
from truss_research.evaluacion import seleccion_adaptativa
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.evaluacion.recursos import MeasurementParameters
from truss_research.evaluacion.seleccion_adaptativa import (
    AdaptiveParameters,
    adaptive_execution_parameters,
)
from truss_research.optimizacion import adaptativa_paralela
from truss_research.optimizacion.adaptativa_paralela import (
    BANK_FILENAME,
    AdaptiveRun,
    PreparedAdaptiveBank,
    adaptive_commands,
    aggregate_adaptive_blocks,
    execute_adaptive_blocks,
    export_worker_tables,
    prepare_adaptive_bank,
)
from truss_research.optimizacion.catalogos import SelectionBank, episode_specs
from truss_research.optimizacion.contratos import SelectionConfig
from truss_research.optimizacion.ejecucion_cpu import ExecutionParameters
from truss_research.optimizacion.estudio_estado import StudyConfig
from truss_research.surrogates.experimentos import data_signature, write_json


@pytest.fixture
def prepared_run(
    design_data: Path, tmp_path: Path
) -> tuple[PreparedAdaptiveBank, Path, AdaptiveRun]:
    split = load_design_split(design_data)
    table = split.validation.set_index("iter_id", drop=False)
    rng = np.random.default_rng(42)
    samples = table.loc[:, list(OUTPUT_COLUMNS)].to_numpy()[None, :, :] * rng.uniform(
        0.9, 1.1, size=(5, len(table), 3)
    )
    selection = SelectionConfig(
        design_data,
        tmp_path / "unused",
        tmp_path / "evaluation",
        catalog_size=4,
        budgets=(2, 3),
        seeds=(42, 43),
        episodes_per_seed=2,
    )
    bank = SelectionBank(
        table,
        pd.DataFrame(samples[0], columns=OUTPUT_COLUMNS, index=table.index),
        episode_specs(table, selection),
        ObjectiveSpace.fit(split.training.loc[:, list(OUTPUT_COLUMNS)].to_numpy()),
        data_signature(design_data),
        "synthetic_ensemble",
    )
    prepared = PreparedAdaptiveBank(
        bank, samples, 1.0, "validation", "synthetic_comparison"
    )
    root = tmp_path / "bank"
    prepare_adaptive_bank(prepared, root)
    selection.output_dir.mkdir()
    run = AdaptiveRun(
        ExecutionParameters(cpu_workers=1),
        selection.output_dir,
        design_data,
        MeasurementParameters(memory_interval_ms=5),
        1.0,
    )
    return prepared, root, run


def test_block_cache_order_and_exact_coverage(
    prepared_run: tuple[PreparedAdaptiveBank, Path, AdaptiveRun],
) -> None:
    prepared, root, run = prepared_run
    commands = adaptive_commands(prepared, root, run)
    execute_adaptive_blocks(commands, run)
    rows, traces = aggregate_adaptive_blocks(prepared, tuple(reversed(commands)))
    assert len(rows) == len(prepared.bank.episodes) * 2
    assert rows.episode_id.tolist() == [
        spec.episode_id for spec in prepared.bank.episodes for _ in range(2)
    ]
    assert (
        traces.groupby(["episode_id", "strategy"])["queries_remaining"]
        .last()
        .eq(0)
        .all()
    )
    manifests = [
        read_artifact(command.resources.destination / "run.json")
        for command in commands
    ]
    assert [manifest["bank_cache_miss"] for manifest in manifests] == [True, False]
    assert manifests[1]["bank_loading_seconds"] == 0
    assert export_worker_tables(run)["worker_coverage"] == "complete"


@pytest.mark.parametrize("damage", ["missing", "duplicate", "bank", "table", "budget"])
def test_corrupt_or_incomplete_blocks_are_rejected(
    prepared_run: tuple[PreparedAdaptiveBank, Path, AdaptiveRun], damage: str
) -> None:
    prepared, root, run = prepared_run
    commands = adaptive_commands(prepared, root, run)
    execute_adaptive_blocks(commands, run)
    if damage == "missing":
        commands = commands[1:]
    elif damage == "duplicate":
        commands = (commands[0], *commands)
    elif damage == "bank":
        with (root / BANK_FILENAME).open("ab") as stream:
            stream.write(b"alterado")
    elif damage == "table":
        with (commands[0].resources.destination / "summary.csv").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write("\n")
    else:
        output = commands[0].resources.destination
        table = pd.read_csv(output / "traces.csv").iloc[1:]
        table.to_csv(output / "traces.csv", index=False)
        manifest = read_artifact(output / "run.json")
        manifest["table_signatures"]["traces.csv"] = source_digest(
            output / "traces.csv"
        )
        write_json(output / "run.json", manifest)
    with pytest.raises(ValueError):
        aggregate_adaptive_blocks(prepared, commands)


def test_memory_gate_pauses_before_starting_work(
    prepared_run: tuple[PreparedAdaptiveBank, Path, AdaptiveRun],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, root, run = prepared_run
    commands = adaptive_commands(prepared, root, run)
    admissions = iter((False, True, True))
    sleeps: list[float] = []
    monkeypatch.setattr(
        adaptativa_paralela.MemoryAdmission,
        "allow",
        lambda self, active: next(admissions),
    )
    monkeypatch.setattr(adaptativa_paralela, "sleep", sleeps.append)
    execute_adaptive_blocks(commands, run)
    assert sleeps == [run.execution.telemetry_interval_seconds]
    assert all(record["status"] == "complete" for record in run.records)


def test_failed_block_preserves_diagnostics_and_stops_queue(
    prepared_run: tuple[PreparedAdaptiveBank, Path, AdaptiveRun],
) -> None:
    prepared, root, run = prepared_run
    commands = adaptive_commands(prepared, root, run)
    commands = (replace(commands[0], bank_signature="invalid"), *commands[1:])
    with pytest.raises(RuntimeError, match="Falló"):
        execute_adaptive_blocks(commands, run)
    assert run.records[0]["status"] == "failed"
    assert run.records[1]["status"] == "interrupted"
    assert commands[0].resources.destination.with_suffix(".log").is_file()
    assert export_worker_tables(run)["worker_coverage"] == "incomplete"


def test_single_episode_and_worker_limit(
    prepared_run: tuple[PreparedAdaptiveBank, Path, AdaptiveRun],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, root, run = prepared_run
    config = StudyConfig(
        run.output_dir,
        data_dir=run.data_dir,
        seeds=(42,),
        catalog_size=4,
        budgets=(2,),
        episodes_per_seed=1,
        execution=ExecutionParameters(cpu_workers=12),
    )
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    resolved, requested = adaptive_execution_parameters(config, None)
    assert requested == 12 and resolved.effective_workers == 1
    resolved, requested = adaptive_execution_parameters(
        replace(config, episodes_per_seed=10), 3
    )
    assert requested == 3 and resolved.effective_workers == 3
    single = replace(
        prepared, bank=replace(prepared.bank, episodes=prepared.bank.episodes[:1])
    )
    single_root = root.parent / "single_bank"
    prepare_adaptive_bank(single, single_root)
    commands = adaptive_commands(single, single_root, run)
    execute_adaptive_blocks(commands, run)
    assert len(aggregate_adaptive_blocks(single, commands)[0]) == 2


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_count_is_rejected(workers: int) -> None:
    with pytest.raises(ValueError, match="cpu_workers"):
        AdaptiveParameters(cpu_workers=workers)


def test_cli_forwards_adaptive_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.__main__ import run_adaptive_command

    captured: list[AdaptiveParameters] = []

    def capture(study: Path, destination: Path, parameters: AdaptiveParameters) -> Path:
        captured.append(parameters)
        return destination

    monkeypatch.setattr(seleccion_adaptativa, "run_adaptive_selection", capture)
    monkeypatch.setattr(
        "truss_research.optimizacion.estudio_estado.load_study_config",
        lambda path: StudyConfig(tmp_path / "study"),
    )
    run_adaptive_command(
        argparse.Namespace(
            estudio=tmp_path / "study",
            destino=tmp_path / "output",
            cpu_workers=12,
        )
    )
    assert captured[0].cpu_workers == 12
