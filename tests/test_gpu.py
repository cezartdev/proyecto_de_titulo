"""Contratos del benchmark CUDA y permanencia del entrenamiento principal en CPU."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone

from truss_research.evaluacion.benchmark_gpu import (
    load_neural_protocol,
    paired_comparison,
)
from truss_research.evaluacion.pruebas_neuronales import (
    NeuralBenchParameters,
    NeuralCase,
    learner_for,
    synthetic_replay,
    update_sequence,
)
from truss_research.surrogates.configuracion import ModelParameters, PredictorConfig
from truss_research.surrogates.experimentos import load_predictor
from truss_research.surrogates.mlp import TorchRegressor
from truss_research.surrogates.modelos import native_factory


def test_lightgbm_explicitly_uses_cpu() -> None:
    model = native_factory(PredictorConfig(model="lightgbm"))
    assert model.get_params()["device_type"] == "cpu"


def test_mlp_clone_keeps_device_and_cpu_default() -> None:
    cpu = TorchRegressor(ModelParameters())
    assert cpu.device == clone(cpu).device == "cpu"
    assert clone(TorchRegressor(ModelParameters(), device="cuda")).device == "cuda"


def test_mlp_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA no está disponible"):
        TorchRegressor(ModelParameters(), device="cuda").fit(
            np.ones((20, 11)), np.ones((20, 3))
        )


def test_synthetic_dqn_uses_real_loss_and_updates_weights() -> None:
    case = NeuralCase("technical", width=8, batch=2, catalog_size=4)
    batch = synthetic_replay(case, 42)
    assert batch["state", "values"].shape == (2, 4, 18)
    assert batch["state", "available"][:, 0].all()
    assert not batch["next", "state", "available"][:, 0].any()
    learner = learner_for(case, (42, "cpu"))
    before = [parameter.detach().clone() for parameter in learner.online.parameters()]
    loss = update_sequence(learner, (batch, "cpu", True), 2)
    assert np.isfinite(loss)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, learner.online.parameters(), strict=True)
    )


def test_comparison_uses_paired_ratios_after_median_per_seed() -> None:
    rows = [
        {
            "network": "mlp",
            "case": "study",
            "operation": "fit",
            "batch": 64,
            "seed": seed,
            "device": device,
            "wall_seconds": value,
        }
        for seed, values in (
            (42, {"cpu": (2, 4), "cuda": (1, 2)}),
            (43, {"cpu": (30, 50), "cuda": (10, 10)}),
        )
        for device, measurements in values.items()
        for value in measurements
    ]
    summary = paired_comparison(pd.DataFrame(rows)).iloc[0]
    assert summary.paired_speedup_median == 3
    assert summary.paired_seeds == 2 and summary.faster == "gpu"


@pytest.mark.parametrize(
    "values",
    [{"seeds": ()}, {"epochs": 0}, {"warmup": True}, {"inference_batches": ()}],
)
def test_invalid_neural_measurements(
    values: dict[str, tuple[int, ...] | int | bool],
) -> None:
    with pytest.raises(ValueError):
        NeuralBenchParameters(**values)


def test_gpu_protocol_matches_study_and_rejects_unknown_fields(tmp_path: Path) -> None:
    protocol = load_neural_protocol()
    assert protocol.mlp_cases[0].rows == 1018
    assert protocol.dqn_cases[0].batch == 64
    invalid = tmp_path / "bad.json"
    invalid.write_text(json.dumps({"unknown": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="desconocidos"):
        load_neural_protocol(invalid)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_comparison_rejects_missing_device(device: str) -> None:
    with pytest.raises(ValueError, match="CPU y CUDA"):
        paired_comparison(pd.DataFrame({"device": [device], "wall_seconds": [1.0]}))


def test_gpu_unavailable_does_not_create_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.evaluacion.benchmark_gpu import run_neural_benchmark

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    destination = tmp_path / "absent"
    with pytest.raises(ValueError, match="CUDA no está disponible"):
        run_neural_benchmark(None, destination)
    assert not destination.exists()


def test_predictor_without_device_contract_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(
        json.dumps({"schema_version": 3}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="formato"):
        load_predictor(tmp_path)
