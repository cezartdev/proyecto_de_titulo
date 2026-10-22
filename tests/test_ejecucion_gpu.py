"""Continuidad de SHAP y propagación de dispositivos sin alterar artefactos firmados."""

from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.surrogates.explicabilidad import (
    ExplanationConfig,
    permutation_values,
)


@pytest.mark.parametrize(
    "parameters",
    [
        {"rl_device": "cuda"},
        {"gpu_models": ("mlp",)},
        {"gpu_models": ("catboost", "xgboost")},
        {"gpu_workers": 2},
        {"cpu_workers": 13},
    ],
)
def test_campaign_rejects_experimental_devices(
    tmp_path: Path, parameters: dict[str, str | int | tuple[str, ...]]
) -> None:
    from truss_research.optimizacion.ejecucion_cpu import ExecutionParameters
    from truss_research.optimizacion.estudio import run_study
    from truss_research.optimizacion.estudio_estado import StudyConfig

    execution = ExecutionParameters(**parameters)
    with pytest.raises(ValueError, match="campaña"):
        run_study(StudyConfig(tmp_path / "untouched", execution=execution))
    assert not (tmp_path / "untouched").exists()


def test_campaign_profiles_and_plan(tmp_path: Path) -> None:
    import json

    from truss_research.optimizacion.protocolo import load_protocol, study_plan

    for name, accelerator in (
        ("study", "cpu"),
        ("study", "cuda"),
        ("smoke", "cpu"),
        ("smoke", "cuda"),
    ):
        config = load_protocol(
            Path(f"config/{name}.json"), (tmp_path, tmp_path), accelerator
        )
        plan = json.loads(study_plan(config, "todo"))["execution"]
        assert config.execution.rl_device == "cpu"
        assert not config.execution.gpu_models
        assert config.execution.gpu_workers == 1
        assert config.execution.cpu_workers <= 12
        assert plan["device"] == ("cpu" if accelerator == "cpu" else "cpu+cuda")
        assert plan["gpu_workers"] == (0 if accelerator == "cpu" else 1)

    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"execution":{"rl_device":"cuda"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="exclusivamente para TabPFN"):
        load_protocol(invalid, (tmp_path, tmp_path))


@pytest.mark.parametrize(
    "case",
    [
        ("search", "tabpfn", True),
        ("fold", "targetwise", True),
        ("predictor", "mlp", False),
        ("predictor", "catboost", False),
    ],
)
def test_campaign_routes_only_tabpfn_computation(
    tmp_path: Path, case: tuple[str, str, bool]
) -> None:
    from truss_research.optimizacion.ejecucion_cpu import CpuPool
    from truss_research.optimizacion.estudio_estado import StudySession
    from truss_research.optimizacion.paralelismo import JobNode, uses_gpu
    from truss_research.optimizacion.protocolo import load_protocol

    kind, name, expected = case
    config = load_protocol(Path("config/study.json"), (tmp_path, tmp_path), "cuda")
    session = StudySession(config, "dataset", "source", {})
    session.gpu_executor = CpuPool(config.execution, "cuda")
    try:
        assert uses_gpu(session, JobNode("job", kind, name, 42)) is expected
    finally:
        session.gpu_executor.shutdown()


def test_loading_mixed_components_moves_only_tabpfn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from sklearn.dummy import DummyRegressor

    from truss_research.surrogates.configuracion import PredictorConfig
    from truss_research.surrogates.especializacion import TargetwiseRegressor
    from truss_research.surrogates.persistencia import load_model, prediction_devices
    from truss_research.surrogates.tabpfn import TabPFNRegressor

    config = PredictorConfig(model="targetwise", trials=0)
    configs = [
        replace(config, model="dummy", target_index=0),
        replace(config, model="tabpfn", target_index=1),
        replace(config, model="mlp", target_index=2),
    ]
    from truss_research.surrogates.mlp import TorchRegressor

    components = [
        Pipeline([("regressor", DummyRegressor())]),
        Pipeline([("regressor", TabPFNRegressor(config.tabpfn, 42, 1))]),
        Pipeline([("regressor", TorchRegressor(config.parameters, 42))]),
    ]
    compound = TargetwiseRegressor(config)
    compound.component_configs_ = configs
    compound.component_signatures_ = {str(i): {} for i in range(len(components))}
    paths = {
        tmp_path / "components" / str(i) / "model.joblib": model
        for i, model in enumerate(components)
    }
    paths[tmp_path / "model.joblib"] = Pipeline([("regressor", compound)])
    monkeypatch.setattr("joblib.load", paths.__getitem__)
    monkeypatch.setattr(
        "truss_research.surrogates.persistencia.model_signatures", lambda path: {}
    )
    restored: list[str] = []
    monkeypatch.setattr(
        TabPFNRegressor,
        "restore_states",
        lambda model, path: restored.append(model.settings.device),
    )
    monkeypatch.setenv("TRUSS_WORKER_DEVICE", "cuda")
    model = load_model(tmp_path)
    assert restored == ["cuda"]
    assert prediction_devices(model) == [
        {"target": "0", "model": "dummy", "device": "cpu"},
        {"target": "1", "model": "tabpfn", "device": "cuda"},
        {"target": "2", "model": "mlp", "device": "cpu"},
    ]


def test_cuda_missing_fails_before_pilot_or_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.optimizacion.estudio import validate_training_protocol
    from truss_research.optimizacion.protocolo import load_protocol

    config = load_protocol(Path("config/study.json"), (tmp_path, tmp_path), "cuda")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA no está disponible"):
        validate_training_protocol(config)


def test_gpu_queue_does_not_occupy_cpu_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.optimizacion.ejecucion_cpu import CpuPool, ExecutionParameters
    from truss_research.optimizacion.estudio_estado import StudyConfig, StudySession
    from truss_research.optimizacion.paralelismo import JobNode, schedule_ready
    from truss_research.optimizacion.trabajos_predictivos import JobOutcome
    from truss_research.surrogates.tabpfn import TabPFNParameters

    config = StudyConfig(
        tmp_path,
        models=("dummy", "ridge", "tabpfn", "targetwise"),
        tabpfn=TabPFNParameters(device="cuda"),
        execution=ExecutionParameters(cpu_workers=2),
    )
    session = StudySession(config, "dataset", "source", {})
    cpu = CpuPool(config.execution)
    gpu = CpuPool(config.execution, "cuda")
    session.gpu_executor = gpu
    monkeypatch.setattr(
        "truss_research.optimizacion.paralelismo.gpu_admission",
        lambda parameters, active: True,
    )

    def capture[Argument, Result](
        operation: Callable[[Argument], Result], argument: Argument
    ) -> Future[Result]:
        return Future()

    monkeypatch.setattr(cpu, "submit", capture)
    monkeypatch.setattr(gpu, "submit", capture)
    pending = [
        JobNode(f"predictor/{name}/42/search", "search", name, 42)
        for name in config.models
    ]
    active: dict[Future[JobOutcome], JobNode] = {}
    try:
        schedule_ready(session, pending, (cpu, active))
        assert {node.model for node in active.values()} == {"dummy", "ridge", "tabpfn"}
        assert [node.model for node in pending] == ["targetwise"]
    finally:
        cpu.shutdown()
        gpu.shutdown()


def test_permutations_resume_matches_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        rng.normal(size=(20, len(INPUT_COLUMNS))), columns=INPUT_COLUMNS
    )
    features["web_config"] = rng.integers(0, 5, len(features))
    response = np.column_stack(
        (
            features.iloc[:, 0] * features.iloc[:, 1],
            features.iloc[:, 2] ** 2,
            features.iloc[:, 3] + features.iloc[:, 4],
        )
    )
    model = Pipeline(
        [("regressor", RandomForestRegressor(n_estimators=5, random_state=42))]
    ).fit(features, response)
    data = (features.iloc[:10], features.iloc[10:13])
    config = ExplanationConfig(
        tmp_path, tmp_path, tmp_path / "continuous", permutation_rounds=2
    )
    expected = permutation_values(model, data, config)
    factory = shap.Explainer

    def prediction(array: np.ndarray) -> np.ndarray:
        return model.predict(pd.DataFrame(array, columns=INPUT_COLUMNS))

    direct = factory(
        prediction,
        shap.maskers.Independent(data[0].to_numpy()),
        algorithm="permutation",
        seed=config.seed,
    )(
        data[1].to_numpy(),
        max_evals=config.permutation_rounds * (2 * len(INPUT_COLUMNS) + 1),
        silent=True,
    )
    np.testing.assert_array_equal(expected[0], direct.values)
    np.testing.assert_array_equal(expected[1], direct.base_values)

    class Interruptible:
        def __init__(
            self,
            model: Callable[[np.ndarray], np.ndarray],
            masker: shap.maskers.Independent,
            **kwargs: str | int,
        ) -> None:
            self.explainer = factory(model, masker, **kwargs)
            self.calls = 0

        def __call__(self, rows: np.ndarray, **kwargs: int | bool) -> shap.Explanation:
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return self.explainer(rows, **kwargs)

    interrupted = ExplanationConfig(
        tmp_path, tmp_path, tmp_path / "interrupted", permutation_rounds=2
    )
    monkeypatch.setattr(shap, "Explainer", Interruptible)
    with pytest.raises(KeyboardInterrupt):
        permutation_values(model, data, interrupted)
    monkeypatch.setattr(shap, "Explainer", factory)
    actual = permutation_values(model, data, interrupted)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)
    changed = (data[0], data[1].iloc[:2])
    with pytest.raises(ValueError, match="checkpoint SHAP"):
        permutation_values(model, changed, interrupted)
