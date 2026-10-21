"""Regresión de recompensa inmediata para selección contextual con feedback parcial."""

from dataclasses import dataclass, field

import torch
from tensordict import TensorDictBase
from torch import nn
from torch.nn.functional import smooth_l1_loss

from truss_research.optimizacion.politica import CandidateQNetwork, DQNParameters


class ImmediateRewardLoss(nn.Module):
    def __init__(self, network: CandidateQNetwork) -> None:
        super().__init__()
        self.network = network

    def forward(self, batch: TensorDictBase) -> dict[str, torch.Tensor]:
        scores = self.network(batch["state"])
        actions = batch["action"].reshape(-1, 1)
        selected = scores.gather(1, actions).squeeze(-1)
        target = batch["next", "reward"].reshape(-1).detach()
        return {"loss": smooth_l1_loss(selected, target)}


@dataclass
class ContextualBanditLearner:
    online: CandidateQNetwork
    parameters: DQNParameters
    optimizer: torch.optim.Adam = field(init=False)
    loss: ImmediateRewardLoss = field(init=False)

    def __post_init__(self) -> None:
        self.loss = ImmediateRewardLoss(self.online)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.parameters.learning_rate
        )

    def update_target(self) -> None:
        """El objetivo inmediato no necesita red objetivo ni bootstrap temporal."""
