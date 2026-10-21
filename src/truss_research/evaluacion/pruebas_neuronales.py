"""Operaciones reales de MLP y Double DQN sobre entradas sintéticas pareadas."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter, process_time
from typing import cast

import numpy as np
import torch
from tensordict import TensorDict

from truss_research.datos.procesados import JsonValue
from truss_research.optimizacion.entrenamiento import Learner, update_learner
from truss_research.optimizacion.politica import CandidateQNetwork, DQNParameters
from truss_research.surrogates.configuracion import FloatArray, ModelParameters
from truss_research.surrogates.mlp import MINIMUM_ROWS, TorchRegressor

MLP_DIMENSIONS = 11
DQN_DIMENSIONS = 18
MINIMUM_DQN_CANDIDATES = 3
MIB = 1024**2
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeuralBenchParameters:
    seeds: tuple[int, ...] = (42, 43, 44)
    epochs: int = 300
    patience: int = 20
    warmup: int = 3
    inference_repeats: int = 30
    update_steps: int = 200
    inference_batches: tuple[int, ...] = (1, 32, 64)

    def __post_init__(self) -> None:
        values = (
            self.epochs,
            self.patience,
            self.inference_repeats,
            self.update_steps,
            *self.inference_batches,
        )
        if (
            any(type(value) is not int or value < 1 for value in values)
            or not self.inference_batches
        ):
            raise ValueError("Tamaños y repeticiones deben ser enteros positivos")
        if type(self.warmup) is not int or self.warmup < 0:
            raise ValueError("El calentamiento debe ser no negativo")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(seed) is not int or seed < 0 for seed in self.seeds)
        ):
            raise ValueError("Semillas no negativas, únicas y no vacías")


@dataclass(frozen=True)
class NeuralCase:
    name: str
    width: int
    batch: int
    rows: int = 1018
    catalog_size: int = 32

    def __post_init__(self) -> None:
        if not self.name or any(
            type(value) is not int or value < 1
            for value in (self.width, self.batch, self.rows, self.catalog_size)
        ):
            raise ValueError("Nombre y tamaños válidos requeridos para cada caso")
        if self.rows < MINIMUM_ROWS or self.catalog_size < MINIMUM_DQN_CANDIDATES:
            raise ValueError("Se requieren diez filas MLP y tres candidatos DQN")


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed_neural[T](
    device: str, operation: Callable[[], T]
) -> tuple[T, dict[str, float]]:
    synchronize(device)
    baseline = torch.cuda.memory_allocated() if device == "cuda" else 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    wall, cpu = perf_counter(), process_time()
    result = operation()
    synchronize(device)
    timing = {
        "wall_seconds": perf_counter() - wall,
        "cpu_seconds": process_time() - cpu,
    }
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    timing.update(
        gpu_peak_allocated_mib=peak / MIB, gpu_increment_mib=(peak - baseline) / MIB
    )
    return result, timing


def synthetic_regression(case: NeuralCase, seed: int) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(case.rows, MLP_DIMENSIONS))
    responses = np.column_stack(
        (
            np.sin(features[:, 0]) + features[:, 1] ** 2,
            features[:, 2] * features[:, 3],
            np.cos(features[:, 4]) + features[:, 5],
        )
    )
    responses += rng.normal(scale=0.05, size=responses.shape)
    return features, responses


def device_order(seed: int) -> tuple[str, str]:
    return ("cpu", "cuda") if seed % 2 == 0 else ("cuda", "cpu")


def mlp_samples(
    parameters: NeuralBenchParameters, case: NeuralCase, seed: int
) -> list[dict[str, JsonValue]]:
    features, responses = synthetic_regression(case, seed)
    settings = ModelParameters(
        hidden_width=case.width,
        batch_size=case.batch,
        epochs=parameters.epochs,
        patience=parameters.patience,
    )
    rows: list[dict[str, JsonValue]] = []
    predictions: dict[str, FloatArray] = {}
    for device in device_order(seed):
        logger.info("MLP %s, semilla %s, %s", case.name, seed, device)
        TorchRegressor(
            ModelParameters(epochs=1, hidden_width=case.width, batch_size=case.batch),
            seed,
            device,
        ).fit(features, responses)
        model = TorchRegressor(settings, seed, device)
        _, timing = timed_neural(device, lambda: model.fit(features, responses))
        identity = {
            "network": "mlp",
            "case": case.name,
            "seed": seed,
            "device": device,
            "width": case.width,
            "rows": case.rows,
        }
        predictions[device] = model.predict(features)
        rows.append(
            {
                **identity,
                "operation": "fit_end_to_end",
                "batch": case.batch,
                "repetition": 0,
                "epochs": model.epochs_,
                "training_mae": float(np.abs(predictions[device] - responses).mean()),
                **timing,
            }
        )
        for size in parameters.inference_batches:
            selected = features[:size]
            for _ in range(parameters.warmup):
                model.predict(selected)
            for repetition in range(parameters.inference_repeats):
                _, timing = timed_neural(device, lambda: model.predict(selected))
                rows.append(
                    {
                        **identity,
                        "operation": "predict_end_to_end",
                        "batch": size,
                        "repetition": repetition,
                        **timing,
                    }
                )
        model.network_.cpu()
    delta = float(np.max(np.abs(predictions["cpu"] - predictions["cuda"])))
    for row in rows:
        row["paired_max_prediction_delta"] = delta
    return rows


def synthetic_replay(case: NeuralCase, seed: int) -> TensorDict:
    generator = torch.Generator().manual_seed(seed)
    shape = (case.batch, case.catalog_size, DQN_DIMENSIONS)
    available = torch.ones((case.batch, case.catalog_size), dtype=torch.bool)
    available[:, -1] = False
    state = TensorDict(
        {
            "values": torch.randn(shape, generator=generator),
            "available": available,
            "budget": torch.full((case.batch, 2), 0.5),
        },
        batch_size=[case.batch],
    )
    next_state = state.clone()
    next_state["available"][:, 0] = False
    done = torch.zeros((case.batch, 1), dtype=torch.bool)
    return TensorDict(
        {
            "state": state,
            "action": torch.zeros(case.batch, dtype=torch.long),
            "next": TensorDict(
                {
                    "state": next_state,
                    "reward": torch.rand((case.batch, 1), generator=generator),
                    "done": done,
                    "terminated": done.clone(),
                    "truncated": done.clone(),
                },
                batch_size=[case.batch],
            ),
        },
        batch_size=[case.batch],
    )


def learner_for(case: NeuralCase, context: tuple[int, str]) -> Learner:
    seed, device = context
    torch.manual_seed(seed)
    network = CandidateQNetwork(DQN_DIMENSIONS, case.width).to(device)
    learner = Learner(
        network, DQNParameters(hidden_width=case.width, batch_size=case.batch)
    )
    learner.loss.to(device)
    return learner


def update_sequence(
    learner: Learner, context: tuple[TensorDict, str, bool], steps: int
) -> float:
    batch, device, transfer = context
    loss = 0.0
    for _ in range(steps):
        selected = batch.to(device) if transfer else batch
        loss = update_learner(learner, selected)
    return loss


def dqn_samples(
    parameters: NeuralBenchParameters, case: NeuralCase, seed: int
) -> list[dict[str, JsonValue]]:
    source = synthetic_replay(case, seed)
    rows: list[dict[str, JsonValue]] = []
    outputs: dict[str, torch.Tensor] = {}
    for device in device_order(seed):
        logger.info("DQN %s, semilla %s, %s", case.name, seed, device)
        identity = {
            "network": "dqn",
            "case": case.name,
            "seed": seed,
            "device": device,
            "width": case.width,
            "catalog_size": case.catalog_size,
        }
        for transfer in (True, False):
            learner = learner_for(case, (seed, device))
            batch = source if transfer else source.to(device)
            context = (batch, device, transfer)
            update_sequence(learner, context, parameters.warmup)
            loss, timing = timed_neural(
                device,
                lambda: update_sequence(learner, context, parameters.update_steps),
            )
            rows.append(
                {
                    **identity,
                    "operation": "update_with_transfer"
                    if transfer
                    else "update_resident",
                    "batch": case.batch,
                    "repetition": 0,
                    "steps": parameters.update_steps,
                    "final_loss": loss,
                    **{
                        name: value / parameters.update_steps
                        if name.endswith("seconds")
                        else value
                        for name, value in timing.items()
                    },
                }
            )
        network = learner_for(case, (seed, device)).online.eval()
        state = cast(TensorDict, source["state"][:1])

        def predict() -> torch.Tensor:
            with torch.no_grad():
                return network(state.to(device)).cpu()

        for _ in range(parameters.warmup):
            predict()
        for repetition in range(parameters.inference_repeats):
            output, timing = timed_neural(device, predict)
            rows.append(
                {
                    **identity,
                    "operation": "choose_scores_with_transfer",
                    "batch": 1,
                    "repetition": repetition,
                    **timing,
                }
            )
        outputs[device] = output
    available = source["state", "available"][:1]
    torch.testing.assert_close(
        outputs["cpu"][available], outputs["cuda"][available], rtol=1e-5, atol=1e-6
    )
    return rows
