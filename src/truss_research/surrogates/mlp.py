"""Regresor neuronal con reserva interna; CPU predeterminada y CUDA explícita."""

from copy import deepcopy
from time import perf_counter
from typing import Self

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from truss_research.datos.procesados import JsonValue
from truss_research.surrogates.configuracion import FloatArray, ModelParameters

VALIDATION_FRACTION = 0.1

MINIMUM_ROWS = 10


class TorchRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self, parameters: ModelParameters, seed: int = 42, device: str = "cpu"
    ) -> None:
        self.parameters = parameters
        self.seed = seed
        self.device = device

    def fit(self, features: FloatArray, responses: FloatArray) -> Self:
        if self.device not in ("cpu", "cuda"):
            raise ValueError("El dispositivo neuronal debe ser cpu o cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA no está disponible; instalar PyTorch CUDA en un entorno separado"
            )
        if len(features) < MINIMUM_ROWS:
            raise ValueError("MLP requiere al menos diez filas de entrenamiento")
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        x_train, x_stop, y_train, y_stop = train_test_split(
            features, responses, test_size=VALIDATION_FRACTION, random_state=self.seed
        )
        return self.fit_reserved((x_train, y_train), (x_stop, y_stop))

    def fit_reserved(
        self,
        fitting: tuple[FloatArray, FloatArray],
        stopping: tuple[FloatArray, FloatArray],
    ) -> Self:
        started = perf_counter()
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        x_train, y_train = fitting
        x_stop, y_stop = stopping
        features = x_train
        self.scaler_ = StandardScaler().fit(y_train)
        train_x = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
        train_y = torch.as_tensor(
            self.scaler_.transform(y_train), dtype=torch.float32, device=self.device
        )
        stop_x = torch.as_tensor(x_stop, dtype=torch.float32, device=self.device)
        stop_y = torch.as_tensor(
            self.scaler_.transform(y_stop), dtype=torch.float32, device=self.device
        )
        width = self.parameters.hidden_width
        self.network_ = nn.Sequential(
            nn.Linear(features.shape[1], width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, y_train.shape[1]),
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.network_.parameters(),
            lr=self.parameters.learning_rate,
            weight_decay=self.parameters.mlp_weight_decay,
        )
        self.best_epoch_ = 0
        best_loss, stale = float("inf"), 0
        best_state = deepcopy(self.network_.state_dict())
        for epoch in range(self.parameters.epochs):
            order = torch.randperm(len(train_x)).to(self.device)
            self.network_.train()
            for batch in order.split(self.parameters.batch_size):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(
                    self.network_(train_x[batch]), train_y[batch]
                )
                loss.backward()
                optimizer.step()
            self.network_.eval()
            with torch.no_grad():
                current = float(nn.functional.mse_loss(self.network_(stop_x), stop_y))
            improved = current < best_loss
            best_state = (
                deepcopy(self.network_.state_dict()) if improved else best_state
            )
            self.best_epoch_ = epoch + 1 if improved else self.best_epoch_
            best_loss = min(best_loss, current)
            stale = 0 if improved else stale + 1
            self.epochs_ = epoch + 1
            if stale >= self.parameters.patience:
                break
        self.network_.load_state_dict(best_state)
        self.network_.eval()
        self.early_stopping_: list[dict[str, JsonValue]] = [
            {
                "executed_epochs": self.epochs_,
                "retained_epoch": self.best_epoch_,
                "maximum_epochs": self.parameters.epochs,
                "patience": self.parameters.patience,
                "metric": "standardized_mse",
                "evaluation": "internal_training_reserve",
                "early_stopped": self.epochs_ < self.parameters.epochs,
                "stop_reason": "patience"
                if self.epochs_ < self.parameters.epochs
                else "limit",
                "fit_seconds": perf_counter() - started,
            }
        ]
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, features: FloatArray) -> FloatArray:
        with torch.no_grad():
            predicted = (
                self.network_(
                    torch.as_tensor(features, dtype=torch.float32, device=self.device)
                )
                .cpu()
                .numpy()
            )
        return np.asarray(self.scaler_.inverse_transform(predicted), dtype=float)
