"""Familia tabular preentrenada local con pesos y presupuesto congelados."""

import errno
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import sleep
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.preparacion import source_digest

PACKAGE_VERSION = "9.2.0"
CHECKPOINT_REPOSITORY = "Prior-Labs/tabpfn_2_5"
CHECKPOINT_REVISION = "6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b"
CHECKPOINT_FILENAME = "tabpfn-v2.5-regressor-v2.5_default.ckpt"
CHECKPOINT_SHA256 = "6e0308b7e5f3a8325b798b96d986b2b282c3608b1cda6e2991425a2a261d40b8"
CHECKPOINT_DIRECTORY = Path("artifacts/pretrained/tabpfn")
PILOT_SECONDS = 600
PILOT_SIZES = (128,)
SLOT_POLL_SECONDS = 0.05
type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TabPFNParameters:
    n_estimators: int = 8
    device: str = "cpu"
    fit_mode: str = "fit_with_cache"
    preprocessing_workers: int = 1
    numeric_threads: int = 1
    checkpoint_dir: str = CHECKPOINT_DIRECTORY.as_posix()

    def __post_init__(self) -> None:
        if type(self.n_estimators) is not int or self.n_estimators < 1:
            raise ValueError("TabPFN requiere un número positivo de estimadores")
        if any(
            type(value) is not int
            for value in (self.preprocessing_workers, self.numeric_threads)
        ):
            raise ValueError("Hilos y procesos TabPFN deben ser enteros")
        if (
            self.device,
            self.fit_mode,
            self.preprocessing_workers,
            self.numeric_threads,
        ) not in (("cpu", "fit_with_cache", 1, 1), ("cuda", "fit_with_cache", 1, 1)):
            raise ValueError(
                "TabPFN requiere cpu/cuda, fit_with_cache y un hilo/proceso"
            )
        if not isinstance(self.checkpoint_dir, str) or not self.checkpoint_dir:
            raise ValueError("Falta el directorio del checkpoint TabPFN")

    @property
    def checkpoint(self) -> Path:
        return Path(self.checkpoint_dir).resolve() / CHECKPOINT_FILENAME


def verify_tabpfn(settings: TabPFNParameters) -> None:
    if settings.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA no está disponible; ejecutar make setup y comprobar el controlador NVIDIA"
            )
    try:
        installed = version("tabpfn")
    except PackageNotFoundError as error:
        raise ValueError("Falta TabPFN; ejecutar make setup") from error
    if installed != PACKAGE_VERSION:
        raise ValueError(
            f"Se requiere tabpfn=={PACKAGE_VERSION}; instalado: {installed}"
        )
    if not settings.checkpoint.is_file():
        raise ValueError("Falta el checkpoint local; ejecutar make tabpfn-weights")
    if source_digest(settings.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("El hash del checkpoint TabPFN no coincide con el protocolo")


@contextmanager
def tabpfn_slot(settings: TabPFNParameters) -> Iterator[None]:
    """Serializa operaciones TabPFN entre procesos sin serializar el aprendizaje RL."""
    if settings.device == "cuda" and os.environ.get("TRUSS_WORKER_DEVICE") == "cuda":
        yield
        return
    settings.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock_name = ".tabpfn_cuda.lock" if settings.device == "cuda" else ".tabpfn.lock"
    with (settings.checkpoint.parent / lock_name).open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    sleep(SLOT_POLL_SECONDS)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


class TabPFNRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self, settings: TabPFNParameters, seed: int = 42, target: int | None = None
    ) -> None:
        self.settings = settings
        self.seed = seed
        self.target = target

    @contextmanager
    def operation(self) -> Iterator[None]:
        import torch

        devices = (
            [torch.cuda.current_device()] if self.settings.device == "cuda" else []
        )
        with tabpfn_slot(self.settings), torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(self.seed)
            if devices:
                torch.cuda.manual_seed(self.seed)
            yield

    def fit(self, features: pd.DataFrame, responses: FloatArray) -> Self:
        verify_tabpfn(self.settings)
        with self.operation():
            return self.fit_context(features, responses)

    def fit_context(self, features: pd.DataFrame, responses: FloatArray) -> Self:
        import torch
        from tabpfn import TabPFNRegressor as ScalarRegressor

        from truss_research.surrogates.modelos import validate_features

        table = validate_features(features)
        values = np.asarray(responses, dtype=float)
        self.output_names_ = (
            OUTPUT_COLUMNS if self.target is None else (OUTPUT_COLUMNS[self.target],)
        )
        if self.target is not None:
            values = values.reshape(-1, 1)
        if (
            values.shape != (len(table), len(self.output_names_))
            or not np.isfinite(values).all()
        ):
            raise ValueError("TabPFN requiere tres respuestas finitas alineadas")
        torch.set_num_threads(self.settings.numeric_threads)
        self.estimators_: list[ScalarRegressor] = []
        for column in range(len(self.output_names_)):
            estimator = ScalarRegressor(
                model_path=(
                    Path(self.settings.checkpoint_dir) / CHECKPOINT_FILENAME
                ).as_posix(),
                device=self.settings.device,
                n_estimators=self.settings.n_estimators,
                random_state=self.seed,
                categorical_features_indices=[INPUT_COLUMNS.index("web_config")],
                fit_mode=self.settings.fit_mode,
                n_preprocessing_jobs=self.settings.preprocessing_workers,
                inference_precision=torch.float32,
                enable_torch_compile=False,
                tuning_config=None,
            )
            estimator.fit(table, values[:, column])
            self.estimators_.append(estimator)
        self.n_features_in_ = len(INPUT_COLUMNS)
        return self

    def predict(self, features: pd.DataFrame) -> FloatArray:
        with self.operation():
            return self.predict_context(features)

    def predict_context(self, features: pd.DataFrame) -> FloatArray:
        from truss_research.surrogates.modelos import validate_features

        table = validate_features(features)
        values = np.column_stack(
            [estimator.predict(table) for estimator in self.estimators_]
        )
        return values if self.target is None else values.ravel()

    def save_states(self, destination: Path) -> dict[str, str]:
        from tabpfn.model_loading import save_fitted_tabpfn_model

        signatures = {}
        for name, estimator in zip(self.output_names_, self.estimators_, strict=True):
            path = destination / f"{name}.tabpfn_fit"
            save_fitted_tabpfn_model(estimator, path)
            signatures[path.name] = source_digest(path)
        return signatures

    def restore_states(self, destination: Path) -> None:
        verify_tabpfn(self.settings)
        with self.operation():
            self.restore_context(destination)

    def restore_context(self, destination: Path) -> None:
        import torch
        from tabpfn.model_loading import load_fitted_tabpfn_model

        verify_tabpfn(self.settings)
        torch.set_num_threads(self.settings.numeric_threads)
        if any(
            source_digest(destination / name) != digest
            for name, digest in self.state_signatures_.items()
        ):
            raise ValueError("El estado ajustado TabPFN cambió")
        self.estimators_ = [
            load_fitted_tabpfn_model(
                destination / f"{name}.tabpfn_fit", device=self.settings.device
            )
            for name in self.output_names_
        ]
