"""Máscaras, invariancia, bootstrap terminal y reanudación de Double DQN."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDictBase

from truss_research.optimizacion.entrenamiento import (
    Learner,
    TorchRLCheckpoint,
    Transition,
    replay_checkpoint,
    restore_replay,
    train_policy,
    transition_batch,
)
from truss_research.optimizacion.politica import (
    CandidateQNetwork,
    DQNParameters,
    EncodedState,
    PolicyConfig,
    load_policy,
    tensor_states,
)
from truss_research.optimizacion.variantes_rl import RLAlgorithm
from truss_research.surrogates.configuracion import PredictorConfig
from truss_research.surrogates.experimentos import read_run, train_predictor, write_json


class FixedQ(CandidateQNetwork):
    def __init__(self, scores: list[float]) -> None:
        torch.nn.Module.__init__(self)
        self.scores = torch.nn.Parameter(torch.tensor(scores))

    def forward(self, state: TensorDictBase) -> torch.Tensor:
        values = self.scores.expand(len(state), -1)
        return values.masked_fill(~state["available"], torch.finfo(values.dtype).min)


def test_training_signature_distinguishes_reward_and_representation(
    design_data: Path, tmp_path: Path
) -> None:
    from truss_research.optimizacion.entrenamiento import (
        policy_inputs,
        training_signature,
    )

    predictor = tmp_path / "predictor"
    train_predictor(PredictorConfig(design_data, predictor, model="dummy", folds=2))
    config = PolicyConfig(
        design_data,
        predictor,
        tmp_path / "policy",
        catalog_size=4,
        budgets=(2, 3),
        validation_episodes=1,
    )
    bank, _, _ = policy_inputs(config)
    signature = training_signature(config, bank)
    assert signature != training_signature(
        replace(config, reward_mode="trapezoidal"), bank
    )
    assert signature != training_signature(replace(config, enhanced_state=False), bank)


def test_torchrl_matches_double_dqn_loss_and_gradients() -> None:
    state = EncodedState(
        np.ones((3, 1), dtype=np.float32),
        np.array([False, True, True]),
        np.ones(2, dtype=np.float32),
    )
    network = FixedQ([100.0, 0.9, 0.5])
    learner = Learner(network, DQNParameters(discount=0.5))
    learner.loss.target_value_network_params["module", "0", "module", "scores"].copy_(
        torch.tensor([100.0, 0.2, 0.8])
    )
    batch = transition_batch(Transition(state, 1, 0.3, state, False))
    loss = learner.loss(batch)["loss"]
    reference = torch.nn.functional.smooth_l1_loss(network.scores[1], torch.tensor(0.4))
    torch.testing.assert_close(loss, reference)
    expected_gradient = torch.autograd.grad(
        reference, network.scores, retain_graph=True
    )[0]
    actual_gradient = torch.autograd.grad(loss, network.scores)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    batch["next", "done"] = torch.ones((1, 1), dtype=torch.bool)
    batch["next", "truncated"] = torch.ones((1, 1), dtype=torch.bool)
    torch.testing.assert_close(learner.loss(batch)["loss"], reference)


def test_replay_checkpoint_preserves_ring_cursor_and_compacts() -> None:
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

    state = EncodedState(
        np.ones((3, 1), dtype=np.float32),
        np.array([False, True, True]),
        np.ones(2, dtype=np.float32),
    )
    replay = TensorDictReplayBuffer(storage=LazyTensorStorage(3), batch_size=2)
    for reward in range(5):
        replay.extend(
            transition_batch(Transition(state, 1, float(reward), state, False))
        )
    restored = TensorDictReplayBuffer(storage=LazyTensorStorage(3), batch_size=2)
    restore_replay(restored, replay_checkpoint(replay))
    more = transition_batch(Transition(state, 1, 5.0, state, False))
    replay.extend(more)
    restored.extend(more)
    torch.testing.assert_close(
        replay[:]["next", "reward"], restored[:]["next", "reward"]
    )


def test_q_scores_permute_and_terminal_target_has_no_bootstrap() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(4, 6)).astype(np.float32)
    available = np.array([False, True, True, True])
    state = EncodedState(values, available, np.array([0.5, 0.75], dtype=np.float32))
    network = CandidateQNetwork(6, 8)
    order = np.array([2, 0, 3, 1])
    permuted = replace(state, values=values[order], available=available[order])
    first = network(tensor_states([state])).detach().numpy()[0]
    second = network(tensor_states([permuted])).detach().numpy()[0]
    np.testing.assert_allclose(first[order], second, atol=1e-6)
    assert first[0] < first[1]
    terminal = replace(state, available=np.zeros(4, dtype=bool))
    learner = Learner(network, DQNParameters())
    batch = transition_batch(Transition(state, 1, 0.25, terminal, True))
    loss = learner.loss(batch)["loss"]
    expected = torch.nn.functional.smooth_l1_loss(
        network(tensor_states([state]))[0, 1], torch.tensor(0.25)
    )
    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "execution",
    [
        (algorithm, device)
        for algorithm in ("double_dqn", "dueling_double_dqn")
        for device in ("cpu", "cuda")
    ],
)
def test_policy_smoke_and_exact_resume(
    design_data: Path, tmp_path: Path, execution: tuple[RLAlgorithm, str]
) -> None:
    algorithm, device = execution
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requiere PyTorch CUDA y GPU")
    predictor = tmp_path / "predictor"
    train_predictor(PredictorConfig(design_data, predictor, model="dummy", folds=2))
    parameters = DQNParameters(
        transitions=6,
        hidden_width=8,
        batch_size=2,
        warmup=1,
        validation_interval=3,
        target_interval=2,
        epsilon_steps=4,
    )
    config = PolicyConfig(
        design_data,
        predictor,
        tmp_path / "policy",
        catalog_size=4,
        budgets=(2, 3),
        validation_episodes=1,
        parameters=parameters,
        algorithm=algorithm,
        device=device,
    )
    history = train_policy(config)
    assert len(history) == 2
    load_policy(config.output_dir, predictor)
    resumed = replace(
        config,
        output_dir=tmp_path / "resumed",
        resume_dir=config.output_dir,
        parameters=replace(parameters, transitions=8),
    )
    train_policy(resumed)
    uninterrupted = replace(
        config,
        output_dir=tmp_path / "uninterrupted",
        parameters=replace(parameters, transitions=8),
    )
    train_policy(uninterrupted)
    first = torch.load(resumed.output_dir / "checkpoint.pt", weights_only=False)
    second = torch.load(uninterrupted.output_dir / "checkpoint.pt", weights_only=False)
    assert isinstance(first, TorchRLCheckpoint) and first.step == 8
    for name in first.online:
        torch.testing.assert_close(
            first.online[name], second.online[name], rtol=0, atol=0
        )
    for name in first.loss_state:
        torch.testing.assert_close(
            first.loss_state[name], second.loss_state[name], rtol=0, atol=0
        )
    assert first.actions == second.actions
    assert first.episode == second.episode
    assert first.best_score == second.best_score
    for a, b in zip(first.history, second.history, strict=True):
        assert {k: v for k, v in a.items() if "seconds" not in k} == {
            k: v for k, v in b.items() if "seconds" not in k
        }
    metadata = read_run(predictor)
    metadata["dataset_signature"] = "different_dataset_with_same_model_bytes"
    write_json(predictor / "run.json", metadata)
    with pytest.raises(ValueError, match="datasets diferentes"):
        load_policy(config.output_dir, predictor)
