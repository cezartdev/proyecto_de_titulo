"""PPO categórico en CPU, con rollouts completos y reanudación entre actualizaciones."""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, process_time
from typing import cast

import joblib
import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import InteractionType, TensorDictModule
from torch import nn
from torchrl.modules import MaskedCategorical, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from truss_research.datos.procesados import JsonValue
from truss_research.evaluacion.recursos import measure_phase
from truss_research.optimizacion.catalogos import SelectionBank
from truss_research.optimizacion.contratos import EpisodeSpec
from truss_research.optimizacion.entorno import QueryEnvironment
from truss_research.optimizacion.entrenamiento import (
    OptimizerValue,
    PolicyRun,
    TorchRLCheckpoint,
    Transition,
    policy_inputs,
    read_checkpoint,
    sample_episode,
    save_policy_run,
    training_episode,
    training_generators,
    training_signature,
    transition_batch,
    validate_network,
    validation_score,
)
from truss_research.optimizacion.politica import (
    CandidatePpoNetwork,
    PolicyComponents,
    PolicyConfig,
    PolicyEncoder,
    PolicyStrategy,
    PPOParameters,
    tensor_states,
)
from truss_research.surrogates.experimentos import read_run

logger = logging.getLogger(__name__)


class StateValue(nn.Module):
    def __init__(self, network: CandidatePpoNetwork) -> None:
        super().__init__()
        self.network = network

    def forward(self, state: TensorDictBase) -> torch.Tensor:
        return self.network.state_value(state)


@dataclass
class PpoLearner:
    network: CandidatePpoNetwork
    parameters: PPOParameters
    actor: ProbabilisticActor = field(init=False)
    critic: TensorDictModule = field(init=False)
    loss: ClipPPOLoss = field(init=False)
    gae: GAE = field(init=False)
    optimizer: torch.optim.Adam = field(init=False)

    def __post_init__(self) -> None:
        p = self.parameters
        self.actor = ProbabilisticActor(
            TensorDictModule(self.network, in_keys=["state"], out_keys=["logits"]),
            in_keys={"logits": "logits", "mask": ("state", "available")},
            out_keys=["action"],
            distribution_class=MaskedCategorical,
            return_log_prob=True,
            log_prob_key="action_log_prob",
            default_interaction_type=InteractionType.RANDOM,
        )
        self.critic = TensorDictModule(
            StateValue(self.network), in_keys=["state"], out_keys=["state_value"]
        )
        self.loss = ClipPPOLoss(
            self.actor,
            self.critic,
            clip_epsilon=p.clip_epsilon,
            entropy_coeff=p.entropy_coeff,
            critic_coeff=p.critic_coeff,
            loss_critic_type="l2",
            normalize_advantage=False,
            functional=False,
            log_explained_variance=False,
        )
        self.gae = GAE(
            gamma=p.discount,
            lmbda=p.gae_lambda,
            value_network=None,
            average_gae=False,
            time_dim=0,
            vectorized=False,
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=p.learning_rate)

    def advantages(self, rollout: TensorDictBase) -> None:
        with torch.no_grad():
            self.gae(rollout)
        advantage = rollout["advantage"]
        rollout["advantage"] = (advantage - advantage.mean()) / advantage.std(
            unbiased=False
        ).clamp(min=self.parameters.advantage_epsilon)

    def update(self, batch: TensorDictBase) -> dict[str, float]:
        losses = self.loss(batch)
        loss = sum(
            losses[name] for name in ("loss_objective", "loss_critic", "loss_entropy")
        )
        if not torch.isfinite(loss).all():
            raise ValueError("Pérdida PPO no finita")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.parameters.gradient_clip,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        validate_network(self.network)
        metrics = {
            name: float(losses[name].detach())
            for name in (
                "loss_objective",
                "loss_critic",
                "loss_entropy",
                "entropy",
                "kl_approx",
                "clip_fraction",
            )
        }
        metrics["loss"] = float(loss.detach())
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("Diagnósticos PPO no finitos")
        return metrics


@dataclass
class PpoSession:
    config: PolicyConfig
    inputs: tuple[SelectionBank, SelectionBank, PolicyEncoder]
    parameters: PPOParameters = field(init=False)
    catalog_rng: np.random.Generator = field(init=False)
    exploration_rng: np.random.Generator = field(init=False)
    rng: np.random.Generator = field(init=False)
    spec: EpisodeSpec = field(init=False)
    environment: QueryEnvironment = field(init=False)
    network: CandidatePpoNetwork = field(init=False)
    learner: PpoLearner = field(init=False)
    policy: PolicyStrategy = field(init=False)
    signature: str = field(init=False)
    dimensions: int = field(init=False)
    run: PolicyRun = field(init=False)
    cpu_started: float = field(init=False)
    cpu_before: float = 0.0
    step: int = 0
    actions: list[int] = field(default_factory=list)
    history: list[dict[str, float | int]] = field(default_factory=list)
    best_state: dict[str, torch.Tensor] = field(init=False)
    best_score: float = float("-inf")
    best_decision_seconds: float = float("inf")
    zero_rewards: int = 0
    optimizer_updates: int = 0
    samples_used: int = 0
    episodes: int = 1

    def __post_init__(self) -> None:
        self.parameters = cast(PPOParameters, self.config.parameters)
        bank, _, encoder = self.inputs
        self.catalog_rng, self.exploration_rng, self.rng = training_generators(
            self.config.seed
        )
        self.spec, self.environment = sample_episode(
            bank, (self.config, self.catalog_rng)
        )
        self.dimensions = encoder.encode(self.environment.observe()).values.shape[1]
        self.network = CandidatePpoNetwork(
            self.dimensions, self.parameters.hidden_width, self.config.enhanced_state
        ).to(self.config.device)
        self.learner = PpoLearner(self.network, self.parameters)
        self.policy = PolicyStrategy(PolicyComponents(self.network, encoder))
        self.best_state = deepcopy(self.network.state_dict())
        self.signature = training_signature(self.config, bank)

    def restore(self, directory: Path) -> None:
        saved = read_checkpoint(directory, self.signature)
        self.network.load_state_dict(saved.online)
        self.learner.loss.load_state_dict(saved.loss_state)
        self.learner.optimizer.load_state_dict(saved.optimizer)
        self.rng.bit_generator.state = saved.rng_state
        self.catalog_rng.bit_generator.state = saved.catalog_rng_state
        self.exploration_rng.bit_generator.state = saved.exploration_rng_state
        torch.set_rng_state(saved.torch_rng)
        if self.config.device == "cuda" and saved.cuda_rng is not None:
            torch.cuda.set_rng_state(saved.cuda_rng)
        self.step, self.spec, self.actions = (
            saved.step,
            saved.episode,
            list(saved.actions),
        )
        self.history, self.best_state, self.best_score = (
            saved.history,
            saved.best_state,
            saved.best_score,
        )
        self.best_decision_seconds, self.zero_rewards = (
            saved.best_decision_seconds,
            saved.zero_rewards,
        )
        self.optimizer_updates, self.samples_used, self.cpu_before = (
            saved.optimizer_updates,
            saved.samples_used,
            saved.cpu_seconds,
        )
        self.episodes = saved.episodes
        self.environment = training_episode(self.inputs[0], self.spec, self.config)
        for action in self.actions:
            self.environment.query(action)

    def checkpoint(self) -> None:
        snapshot = TorchRLCheckpoint(
            self.signature,
            self.step,
            deepcopy(self.network.state_dict()),
            deepcopy(self.learner.loss.state_dict()),
            cast(dict[str, OptimizerValue], self.learner.optimizer.state_dict()),
            {},
            cast(dict[str, JsonValue], self.rng.bit_generator.state),
            torch.get_rng_state(),
            self.spec,
            tuple(self.actions),
            self.best_state,
            self.best_score,
            self.history,
            self.best_decision_seconds,
            self.zero_rewards,
            cast(dict[str, JsonValue], self.catalog_rng.bit_generator.state),
            cast(dict[str, JsonValue], self.exploration_rng.bit_generator.state),
            self.optimizer_updates,
            self.samples_used,
            self.cpu_before + process_time() - self.cpu_started,
            self.episodes,
        )
        save_policy_run(self.run, snapshot)

    def collect(self, count: int) -> TensorDictBase:
        records: list[TensorDict] = []
        bank, _, encoder = self.inputs
        cached_observation = self.environment.observe()
        cached_state = (
            None if cached_observation.terminal else encoder.encode(cached_observation)
        )
        for _ in range(count):
            if cached_observation.terminal or cached_state is None:
                self.spec, self.environment = sample_episode(
                    bank, (self.config, self.catalog_rng)
                )
                self.actions = []
                self.episodes += 1
                cached_observation = self.environment.observe()
                cached_state = encoder.encode(cached_observation)
            state = cached_state
            with torch.no_grad():
                sampled = self.learner.actor(
                    TensorDict(
                        {"state": tensor_states([state]).to(self.config.device)},
                        batch_size=[1],
                    )
                )
                self.learner.critic(sampled)
            action = int(sampled["action"].item())
            _, reward, terminal, truncated, _ = self.environment.step(action)
            if truncated:
                raise ValueError(
                    "Un corte del recolector no puede marcarse como terminal"
                )
            next_observation = self.environment.observe()
            next_state = encoder.encode(next_observation)
            record = transition_batch(
                Transition(
                    state,
                    action,
                    reward,
                    next_state,
                    terminal,
                )
            ).to(self.config.device)
            cached_observation = next_observation
            cached_state = None if terminal else next_state
            record["action_log_prob"] = sampled["action_log_prob"].detach().clone()
            record["state_value"] = sampled["state_value"].detach().clone()
            with torch.no_grad():
                record["next", "state_value"] = self.network.state_value(
                    record["next", "state"]
                )
            records.append(record)
            self.actions.append(action)
            self.step += 1
            self.zero_rewards += int(reward == 0)
        return torch.cat(records, dim=0)

    def optimize(self, rollout: TensorDictBase) -> dict[str, float]:
        self.learner.advantages(rollout)
        sums: dict[str, float] = {}
        samples = 0
        for _ in range(self.parameters.epochs):
            indices = self.rng.permutation(len(rollout))
            for start in range(0, len(rollout), self.parameters.batch_size):
                batch = rollout[
                    torch.as_tensor(
                        indices[start : start + self.parameters.batch_size],
                        device=self.config.device,
                    )
                ]
                metrics = self.learner.update(batch)
                for name, value in metrics.items():
                    sums[name] = sums.get(name, 0.0) + value * len(batch)
                samples += len(batch)
                self.optimizer_updates += 1
                self.samples_used += len(batch)
        return {name: value / samples for name, value in sums.items()}

    def validate(self, metrics: dict[str, float]) -> None:
        started, cpu_started = perf_counter(), process_time()
        with measure_phase("policy_validation"):
            result = validation_score(self.policy, self.inputs[1])
        wall, cpu = perf_counter() - started, process_time() - cpu_started
        if result.score > self.best_score:
            self.best_score, self.best_state = (
                result.score,
                deepcopy(self.network.state_dict()),
            )
            self.best_decision_seconds = result.decision_seconds
        self.history.append(
            {
                "step": self.step,
                "validation_auc": result.score,
                "validation_seconds": result.decision_seconds,
                "validation_wall_seconds": wall,
                "validation_cpu_seconds": cpu,
                "validation_useful_episodes": result.useful_episodes,
                "optimizer_updates": self.optimizer_updates,
                "samples_used": self.samples_used,
                "episodes": self.episodes,
                "optimization_wall_seconds": self.run.elapsed_before
                + perf_counter()
                - self.run.start_time
                - sum(float(row["validation_wall_seconds"]) for row in self.history)
                - wall,
                "optimization_cpu_seconds": self.cpu_before
                + process_time()
                - self.cpu_started
                - sum(float(row["validation_cpu_seconds"]) for row in self.history)
                - cpu,
                **metrics,
            }
        )
        logger.info("PPO paso %s: AUC de validación %.6f", self.step, result.score)


def train_ppo(config: PolicyConfig) -> pd.DataFrame:
    if config.output_dir.exists():
        raise ValueError("El destino ya existe; elegir una ejecución nueva")
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    session = PpoSession(config, policy_inputs(config))
    elapsed_before = 0.0
    if config.resume_dir is not None:
        session.restore(config.resume_dir)
        elapsed_before = float(str(read_run(config.resume_dir)["training_seconds"]))
    if session.step >= session.parameters.transitions:
        raise ValueError("La política ya consumió su presupuesto de transiciones")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(session.inputs[2], config.output_dir / "encoder.joblib")
    session.run = PolicyRun(
        config, session.inputs[0], session.dimensions, perf_counter(), elapsed_before
    )
    session.cpu_started = process_time()
    session.checkpoint()
    while session.step < session.parameters.transitions:
        until_validation = (
            session.parameters.validation_interval
            - session.step % session.parameters.validation_interval
        )
        count = min(
            session.parameters.rollout_steps,
            until_validation,
            session.parameters.transitions - session.step,
        )
        metrics = session.optimize(session.collect(count))
        if (
            session.step % session.parameters.validation_interval == 0
            or session.step == session.parameters.transitions
        ):
            session.validate(metrics)
        session.checkpoint()
    return pd.DataFrame(session.history)
