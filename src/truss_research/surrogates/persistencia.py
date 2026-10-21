"""Persistencia de pipelines y estados TabPFN con pesos compartidos verificables."""

import os
from copy import copy
from dataclasses import replace
from pathlib import Path
from shutil import copytree

import joblib
from sklearn.pipeline import Pipeline

from truss_research.surrogates.tabpfn import TabPFNRegressor

MODEL_FILE = "model.joblib"


def save_model(model: Pipeline, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    regressor = model.named_steps["regressor"]
    from truss_research.surrogates.especializacion import TargetwiseRegressor

    if isinstance(regressor, TargetwiseRegressor):
        descriptor = copy(regressor)
        descriptor.component_configs_ = [config for config, _ in regressor.components_]
        for index, (_, component) in enumerate(regressor.components_):
            save_model(component, destination / "components" / str(index))
        descriptor.component_signatures_ = {
            str(index): model_signatures(destination / "components" / str(index))
            for index in range(len(regressor.components_))
        }
        del descriptor.components_
        joblib.dump(Pipeline([("regressor", descriptor)]), destination / MODEL_FILE)
        return
    if not isinstance(regressor, TabPFNRegressor):
        joblib.dump(model, destination / MODEL_FILE)
        return
    descriptor = copy(regressor)
    descriptor.state_signatures_ = regressor.save_states(destination)
    del descriptor.estimators_
    joblib.dump(Pipeline([("regressor", descriptor)]), destination / MODEL_FILE)


def load_model(directory: Path, device: str | None = None) -> Pipeline:
    device = device or os.environ.get("TRUSS_WORKER_DEVICE")
    model = joblib.load(directory / MODEL_FILE)
    if not isinstance(model, Pipeline):
        raise ValueError("El artefacto no contiene un pipeline predictor")
    regressor = model.named_steps["regressor"]
    from truss_research.surrogates.especializacion import TargetwiseRegressor

    if isinstance(regressor, TargetwiseRegressor):
        if any(
            model_signatures(directory / "components" / index) != expected
            for index, expected in regressor.component_signatures_.items()
        ):
            raise ValueError("Los componentes del predictor cambiaron")
        regressor.components_ = [
            (config, load_model(directory / "components" / str(index), device))
            for index, config in enumerate(regressor.component_configs_)
        ]
    if isinstance(regressor, TabPFNRegressor):
        if device is not None:
            regressor.settings = replace(regressor.settings, device=device)
        regressor.restore_states(directory)
    return model


def copy_model(source: Path, destination: Path) -> None:
    """Copia el descriptor y sus estados, excluyendo tablas y pliegues ajenos."""
    from shutil import copyfile

    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name == "components":
            copytree(path, destination / path.name)
        elif path.name == MODEL_FILE or path.suffix == ".tabpfn_fit":
            copyfile(path, destination / path.name)


def model_signatures(directory: Path) -> dict[str, str]:
    from truss_research.datos.preparacion import source_digest

    return {
        path.relative_to(directory).as_posix(): source_digest(path)
        for path in directory.rglob("*")
        if path.is_file()
        and not path.relative_to(directory).parts[0].startswith("fold_")
        and (path.name == MODEL_FILE or path.suffix == ".tabpfn_fit")
    }


def prediction_devices(model: Pipeline) -> list[dict[str, str]]:
    """Identifica el dispositivo efectivo de cada respuesta del compuesto."""
    from truss_research.surrogates.especializacion import TargetwiseRegressor

    regressor = model.named_steps["regressor"]
    if isinstance(regressor, TargetwiseRegressor):
        return [
            {
                "target": str(config.target_index),
                "model": config.model,
                "device": prediction_device(component),
            }
            for config, component in regressor.components_
        ]
    return [{"device": prediction_device(model)}]


def prediction_device(model: Pipeline) -> str:
    from truss_research.surrogates.especializacion import TargetwiseRegressor
    from truss_research.surrogates.mlp import TorchRegressor
    from truss_research.surrogates.modelos import NativeBoostRegressor

    regressor = model.named_steps["regressor"]
    if isinstance(regressor, TargetwiseRegressor):
        return "+".join(
            sorted(
                {prediction_device(component) for _, component in regressor.components_}
            )
        )
    if isinstance(regressor, TabPFNRegressor):
        return regressor.settings.device
    if isinstance(regressor, TorchRegressor):
        return regressor.device
    if (
        isinstance(regressor, NativeBoostRegressor)
        and regressor.config.model == "xgboost"
    ):
        return regressor.config.device
    return "cpu"
