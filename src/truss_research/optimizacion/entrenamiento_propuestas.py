"""Double DQN sobre propuestas dinámicas, con archivo y padding explícitos."""

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter, process_time
from typing import Literal, cast

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from truss_research.cerchas.schemas import INPUT_COLUMNS
from truss_research.datos.preparacion import source_digest
from truss_research.datos.procesados import JsonValue
from truss_research.optimizacion.bandido_contextual import ContextualBanditLearner
from truss_research.optimizacion.busqueda import (
    PROPOSAL_SCHEMA_VERSION,
    EvolutionSearch,
    SearchContext,
    SearchSnapshot,
)
from truss_research.optimizacion.entrenamiento import (
    Learner,
    OptimizerValue,
    Transition,
    replay_checkpoint,
    restore_replay,
    training_generators,
    transition_batch,
    update_learner,
)
from truss_research.optimizacion.politica import (
    CandidateQNetwork,
    DQNParameters,
    EncodedState,
    tensor_states,
)
from truss_research.surrogates.configuracion import PredictorConfig
from truss_research.surrogates.experimentos import read_run, write_json
from truss_research.surrogates.modelos import preprocessing

VALIDATION_SEED_OFFSET = 1_000_000


@dataclass(frozen=True)
class ProposalEncoder:
    transformer: ColumnTransformer

    @classmethod
    def fit(cls, context: SearchContext) -> "ProposalEncoder":
        transformer = preprocessing(PredictorConfig(model="mlp"))
        transformer.fit(pd.DataFrame(context.training, columns=INPUT_COLUMNS))
        return cls(transformer)

    def encode(self, search: EvolutionSearch) -> EncodedState:
        context = search.context
        pool_size = context.parameters.pool_size
        size = pool_size + context.archive_size
        batch = search.batch
        proposals = (
            np.empty((0, len(INPUT_COLUMNS))) if batch is None else batch.features
        )
        features = np.vstack((proposals, search.archive))
        transformed = self.transformer.transform(
            pd.DataFrame(features, columns=INPUT_COLUMNS)
        )
        known = np.arange(len(features)) >= len(proposals)
        responses = np.zeros((len(features), 3))
        responses[len(proposals) :] = context.space.transform(search.responses)
        cheap = np.zeros_like(responses)
        if batch is not None and batch.cheap_values is not None:
            cheap[: len(proposals)] = context.space.transform(batch.cheap_values)
        distances = np.zeros(len(features))
        if batch is not None:
            distances[: len(proposals)] = batch.distances
        front_values = context.space.front(search.responses)
        front = np.zeros(len(features), dtype=bool)
        front[len(proposals) :] = [
            any(np.array_equal(row, value) for value in front_values)
            for row in search.responses
        ]
        items = np.column_stack(
            (
                transformed,
                responses,
                cheap,
                known.astype(float),
                front.astype(float),
                distances,
            )
        )
        slots = np.r_[
            np.arange(len(proposals)), pool_size + np.arange(len(search.archive))
        ]
        values = np.zeros((size, items.shape[1]), dtype=np.float32)
        values[slots] = items
        available, known_mask, valid, front_mask = (
            np.zeros(size, dtype=bool) for _ in range(4)
        )
        available[: len(proposals)] = True
        known_mask[pool_size : pool_size + len(search.archive)] = True
        valid[slots] = True
        front_mask[slots] = front
        budget = np.array(
            [
                search.remaining_evaluations / search.budget,
                search.generated / context.parameters.max_generated,
            ],
            dtype=np.float32,
        )
        return EncodedState(values, available, budget, front_mask, known_mask, valid)


@dataclass(frozen=True)
class ProposalTrainingConfig:
    seed: int
    budgets: tuple[int, ...]
    parameters: DQNParameters
    signature: str
    validation_episodes: int = 10
    objective: Literal["return", "immediate"] = "return"

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0 or not self.signature:
            raise ValueError(
                "La política de propuestas requiere semilla y firma válidas"
            )
        if self.objective not in ("return", "immediate"):
            raise ValueError("Objetivo de aprendizaje de propuestas desconocido")
        if (
            not self.budgets
            or len(set(self.budgets)) != len(self.budgets)
            or any(type(value) is not int or value < 1 for value in self.budgets)
        ):
            raise ValueError("Presupuestos de propuestas inválidos")
        if (
            type(self.validation_episodes) is not int
            or self.validation_episodes < 1
            or self.parameters.discount != 1.0
        ):
            raise ValueError("La validación requiere episodios positivos y descuento 1")


@dataclass
class ProposalCheckpoint:
    signature: str
    step: int
    online: dict[str, torch.Tensor]
    loss: dict[str, torch.Tensor]
    optimizer: dict[str, OptimizerValue]
    replay: dict[str, OptimizerValue]
    best: dict[str, torch.Tensor]
    best_score: float
    search: SearchSnapshot
    catalog_rng: dict[str, JsonValue]
    action_rng: dict[str, JsonValue]
    replay_rng: dict[str, JsonValue]
    torch_rng: torch.Tensor
    history: list[dict[str, float | int]]
    evaluations: int = 0
    episodes: int = 1
    optimizer_updates: int = 0
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    dimensions: int = 0
    completed_costs: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposalValidation:
    score: float
    evaluations: int
    costs: dict[str, int]


SEARCH_COST_COUNTERS = (
    "generated",
    "cheap_evaluations",
    "member_predictions",
    "cheap_member_predictions",
)


def add_search_costs(totals: dict[str, int], search: EvolutionSearch) -> None:
    for name in SEARCH_COST_COUNTERS:
        totals[name] = totals.get(name, 0) + int(getattr(search, name))


def sample_proposal_search(
    context: SearchContext, budgets: tuple[int, ...], rng: np.random.Generator
) -> tuple[EvolutionSearch, dict[str, int]]:
    discarded_costs: dict[str, int] = {"evaluations": 0}
    for _ in range(context.parameters.max_regeneration):
        search = EvolutionSearch(
            context,
            (int(rng.choice(budgets)), int(rng.integers(np.iinfo(np.int32).max))),
        )
        if search.prepare():
            return search, discarded_costs
        discarded_costs["evaluations"] += search.evaluations
        add_search_costs(discarded_costs, search)
    raise ValueError(
        "El piloto no produce acciones admisibles; revisar cobertura y predictores"
    )


def predicted_auc(search: EvolutionSearch) -> float:
    volumes = np.pad(
        np.asarray(search.volumes),
        (0, search.budget + 1 - len(search.volumes)),
        constant_values=search.volumes[-1],
    )
    return float(np.trapezoid(volumes) / (search.budget * search.context.reward_scale))


def choose_proposal(network: CandidateQNetwork, state: EncodedState) -> int:
    if not state.available.any():
        raise ValueError("El lote no tiene propuestas seleccionables")
    with torch.no_grad():
        return int(network(tensor_states([state]))[0].argmax().item())


def validate_proposals(
    context: SearchContext,
    components: tuple[CandidateQNetwork, ProposalEncoder],
    config: ProposalTrainingConfig,
) -> ProposalValidation:
    network, encoder = components
    scores = []
    evaluations = 0
    costs: dict[str, int] = {}
    for budget in config.budgets:
        for number in range(config.validation_episodes):
            seed = int(
                np.random.SeedSequence(
                    [VALIDATION_SEED_OFFSET, config.seed, number, budget]
                ).generate_state(1)[0]
            )
            validation_context = (
                replace(context, worlds=(context.validation_world,))
                if context.validation_world is not None
                else context
            )
            search = EvolutionSearch(validation_context, (budget, seed))
            while search.prepare():
                search.choose(choose_proposal(network, encoder.encode(search)))
            scores.append(predicted_auc(search))
            evaluations += search.evaluations
            add_search_costs(costs, search)
    return ProposalValidation(float(np.mean(scores)), evaluations, costs)


def save_proposal_checkpoint(
    destination: Path, checkpoint: ProposalCheckpoint, config: ProposalTrainingConfig
) -> None:
    torch.save(checkpoint, destination / "checkpoint.tmp.pt")
    if (destination / "checkpoint.pt").exists():
        (destination / "checkpoint.pt").replace(destination / "checkpoint.previous.pt")
    (destination / "checkpoint.tmp.pt").replace(destination / "checkpoint.pt")
    torch.save(checkpoint.best, destination / "policy.tmp.pt")
    (destination / "policy.tmp.pt").replace(destination / "policy.pt")
    pd.DataFrame(checkpoint.history).to_csv(
        destination / "learning_curve.csv", index=False
    )
    write_json(
        destination / "run.json",
        {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "signature": checkpoint.signature,
            "phase": "complete"
            if checkpoint.step == config.parameters.transitions
            else "in_progress",
            "config": asdict(config),
            "completed_steps": checkpoint.step,
            "dimensions": checkpoint.dimensions,
            "training_surrogate_evaluations": checkpoint.evaluations
            + checkpoint.search.evaluations,
            "training_episodes": checkpoint.episodes,
            "training_costs": {
                name: checkpoint.completed_costs.get(name, 0)
                + int(getattr(checkpoint.search, name))
                for name in SEARCH_COST_COUNTERS
            },
            "validation_costs": {
                name: sum(
                    row.get(f"validation_{name}", 0) for row in checkpoint.history
                )
                for name in SEARCH_COST_COUNTERS
            },
            "optimizer_updates": checkpoint.optimizer_updates,
            "wall_seconds": checkpoint.wall_seconds,
            "cpu_seconds": checkpoint.cpu_seconds,
            "validation_surrogate_evaluations": sum(
                row["validation_surrogate_evaluations"] for row in checkpoint.history
            ),
            "validation_wall_seconds": sum(
                row["validation_wall_seconds"] for row in checkpoint.history
            ),
            "checkpoint_signature": source_digest(destination / "checkpoint.pt"),
            "policy_signature": source_digest(destination / "policy.pt"),
            "encoder_signature": source_digest(destination / "encoder.joblib"),
            "learning_curve_signature": source_digest(
                destination / "learning_curve.csv"
            ),
            "evidence": "predicted_only_no_fem",
        },
    )


def read_proposal_checkpoint(
    destination: Path, config: ProposalTrainingConfig
) -> ProposalCheckpoint:
    metadata = read_run(destination)
    previous = metadata.get("config")
    if not isinstance(previous, dict) or not isinstance(
        previous.get("parameters"), dict
    ):
        raise ValueError("Falta la configuración del checkpoint de propuestas")
    requested = json.loads(json.dumps(asdict(config)))
    previous["parameters"].pop("transitions")
    requested["parameters"].pop("transitions")
    if (
        metadata.get("schema_version") != PROPOSAL_SCHEMA_VERSION
        or metadata.get("signature") != config.signature
        or previous != requested
    ):
        raise ValueError("El checkpoint de propuestas pertenece a otra configuración")
    candidates = [
        path
        for path in (
            destination / "checkpoint.pt",
            destination / "checkpoint.previous.pt",
        )
        if path.is_file()
        and source_digest(path) == metadata.get("checkpoint_signature")
    ]
    if not candidates:
        raise ValueError("Cambió el checkpoint de propuestas")
    if metadata.get("encoder_signature") != source_digest(
        destination / "encoder.joblib"
    ):
        raise ValueError("Cambió el codificador de propuestas")
    if metadata.get("phase") == "complete" and any(
        metadata.get(key) != source_digest(destination / filename)
        for key, filename in (
            ("policy_signature", "policy.pt"),
            ("learning_curve_signature", "learning_curve.csv"),
        )
    ):
        raise ValueError("Cambió una política de propuestas completa")
    saved = torch.load(candidates[0], weights_only=False, map_location="cpu")
    if (
        not isinstance(saved, ProposalCheckpoint)
        or saved.signature != config.signature
        or saved.step > config.parameters.transitions
    ):
        raise ValueError("Checkpoint de propuestas incompatible con el presupuesto")
    return saved


def load_proposal_policy(
    destination: Path,
) -> tuple[CandidateQNetwork, ProposalEncoder]:
    metadata = read_run(destination)
    settings = metadata.get("config")
    if metadata.get("phase") != "complete" or not isinstance(settings, dict):
        raise ValueError("La política de propuestas no está completa")
    if metadata.get("schema_version") != PROPOSAL_SCHEMA_VERSION:
        raise ValueError("Formato de política evolutiva incompatible")
    for key, filename in (
        ("policy_signature", "policy.pt"),
        ("encoder_signature", "encoder.joblib"),
    ):
        if not (destination / filename).is_file() or source_digest(
            destination / filename
        ) != metadata.get(key):
            raise ValueError("Cambió una política evolutiva completa")
    encoder = joblib.load(destination / "encoder.joblib")
    if not isinstance(encoder, ProposalEncoder):
        raise ValueError("El artefacto no contiene un codificador de propuestas")
    network = CandidateQNetwork(
        int(metadata["dimensions"]), int(settings["parameters"]["hidden_width"])
    )
    network.load_state_dict(
        torch.load(destination / "policy.pt", weights_only=True, map_location="cpu")
    )
    network.eval()
    return network, encoder


def train_proposal_policy(
    context: SearchContext, config: ProposalTrainingConfig, destination: Path
) -> tuple[CandidateQNetwork, ProposalEncoder]:
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    started, cpu_started = perf_counter(), process_time()
    catalog_rng, action_rng, replay_rng = training_generators(config.seed)
    saved = (
        read_proposal_checkpoint(destination, config) if destination.exists() else None
    )
    encoder = (
        joblib.load(destination / "encoder.joblib")
        if saved
        else ProposalEncoder.fit(context)
    )
    search, discarded = (
        (EvolutionSearch.restore(context, saved.search), {"evaluations": 0})
        if saved
        else sample_proposal_search(context, config.budgets, catalog_rng)
    )
    online = CandidateQNetwork(
        encoder.encode(search).values.shape[1], config.parameters.hidden_width
    )
    learner = (
        ContextualBanditLearner(online, config.parameters)
        if config.objective == "immediate"
        else Learner(online, config.parameters)
    )
    replay = TensorDictReplayBuffer(
        storage=LazyTensorStorage(config.parameters.replay_size),
        batch_size=config.parameters.batch_size,
    )
    checkpoint = ProposalCheckpoint(
        config.signature,
        0,
        deepcopy(online.state_dict()),
        deepcopy(learner.loss.state_dict()),
        cast(dict[str, OptimizerValue], learner.optimizer.state_dict()),
        {},
        deepcopy(online.state_dict()),
        float("-inf"),
        search.snapshot(),
        {},
        {},
        {},
        torch.get_rng_state(),
        [],
        evaluations=discarded["evaluations"],
        dimensions=online.encoder[0].in_features,
        completed_costs={name: discarded.get(name, 0) for name in SEARCH_COST_COUNTERS},
    )
    if saved is not None:
        checkpoint = saved
        online.load_state_dict(checkpoint.online)
        learner.loss.load_state_dict(checkpoint.loss)
        learner.optimizer.load_state_dict(checkpoint.optimizer)
        restore_replay(replay, checkpoint.replay)
        catalog_rng.bit_generator.state = checkpoint.catalog_rng
        action_rng.bit_generator.state = checkpoint.action_rng
        replay_rng.bit_generator.state = checkpoint.replay_rng
        torch.set_rng_state(checkpoint.torch_rng)
    destination.mkdir(parents=True, exist_ok=True)
    if saved is None:
        joblib.dump(encoder, destination / "encoder.joblib")
        checkpoint.catalog_rng = cast(
            dict[str, JsonValue], catalog_rng.bit_generator.state
        )
        checkpoint.action_rng = cast(
            dict[str, JsonValue], action_rng.bit_generator.state
        )
        checkpoint.replay_rng = cast(
            dict[str, JsonValue], replay_rng.bit_generator.state
        )
        checkpoint.torch_rng = torch.get_rng_state()
        save_proposal_checkpoint(destination, checkpoint, config)
    before_wall, before_cpu = checkpoint.wall_seconds, checkpoint.cpu_seconds
    for step in range(checkpoint.step + 1, config.parameters.transitions + 1):
        if search.batch is None:
            checkpoint.evaluations += search.evaluations
            add_search_costs(checkpoint.completed_costs, search)
            checkpoint.episodes += 1
            search, discarded = sample_proposal_search(
                context, config.budgets, catalog_rng
            )
            checkpoint.evaluations += discarded["evaluations"]
            for name in SEARCH_COST_COUNTERS:
                checkpoint.completed_costs[name] = checkpoint.completed_costs.get(
                    name, 0
                ) + discarded.get(name, 0)
        state = encoder.encode(search)
        progress = min(1.0, step / config.parameters.epsilon_steps)
        epsilon = config.parameters.epsilon_start + progress * (
            config.parameters.epsilon_end - config.parameters.epsilon_start
        )
        action = (
            int(action_rng.choice(np.flatnonzero(state.available)))
            if action_rng.random() < epsilon
            else choose_proposal(online, state)
        )
        before = search.evaluations
        reward = search.choose(action)
        terminal = not search.prepare()
        next_state = encoder.encode(search)
        replay.extend(
            transition_batch(Transition(state, action, reward, next_state, terminal))
        )
        loss = 0.0
        if (
            step >= config.parameters.warmup
            and step % config.parameters.train_frequency == 0
            and len(replay) >= config.parameters.batch_size
        ):
            for _ in range(config.parameters.gradient_steps):
                indices = replay_rng.choice(
                    len(replay), config.parameters.batch_size, replace=False
                )
                loss = update_learner(learner, replay[torch.as_tensor(indices)])
                checkpoint.optimizer_updates += 1
        if step % config.parameters.target_interval == 0:
            learner.update_target()
        if (
            step % config.parameters.validation_interval != 0
            and step != config.parameters.transitions
        ):
            continue
        validation_started = perf_counter()
        validation = validate_proposals(context, (online, encoder), config)
        score = validation.score
        if score > checkpoint.best_score:
            checkpoint.best_score, checkpoint.best = (
                score,
                deepcopy(online.state_dict()),
            )
        checkpoint.history.append(
            {
                "step": step,
                "predicted_validation_auc": score,
                "loss": loss,
                "epsilon": epsilon,
                "validation_wall_seconds": perf_counter() - validation_started,
                "validation_surrogate_evaluations": validation.evaluations,
                "optimizer_updates": checkpoint.optimizer_updates,
                "last_step_surrogate_evaluations": search.evaluations - before,
                **{
                    f"validation_{name}": value
                    for name, value in validation.costs.items()
                },
            }
        )
        checkpoint.step, checkpoint.search = step, search.snapshot()
        checkpoint.online, checkpoint.loss = (
            deepcopy(online.state_dict()),
            deepcopy(learner.loss.state_dict()),
        )
        checkpoint.optimizer = cast(
            dict[str, OptimizerValue], learner.optimizer.state_dict()
        )
        checkpoint.replay = replay_checkpoint(replay)
        checkpoint.catalog_rng = cast(
            dict[str, JsonValue], catalog_rng.bit_generator.state
        )
        checkpoint.action_rng = cast(
            dict[str, JsonValue], action_rng.bit_generator.state
        )
        checkpoint.replay_rng = cast(
            dict[str, JsonValue], replay_rng.bit_generator.state
        )
        checkpoint.torch_rng = torch.get_rng_state()
        checkpoint.wall_seconds, checkpoint.cpu_seconds = (
            before_wall + perf_counter() - started,
            before_cpu + process_time() - cpu_started,
        )
        save_proposal_checkpoint(destination, checkpoint, config)
    online.load_state_dict(checkpoint.best)
    online.eval()
    return online, encoder
