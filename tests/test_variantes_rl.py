"""Variantes RL: máscaras, objetivos calculables, catálogos y checkpoints."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from truss_research.optimizacion.entrenamiento import (
    TorchRLCheckpoint,
    policy_inputs,
    sample_episode,
    train_policy,
    training_generators,
)
from truss_research.optimizacion.entrenamiento_ppo import PpoLearner, PpoSession
from truss_research.optimizacion.politica import (
    NETWORK_TYPES,
    CandidatePpoNetwork,
    DQNParameters,
    DuelingCandidateQNetwork,
    EncodedState,
    PolicyConfig,
    PPOParameters,
    load_policy,
    tensor_states,
)
from truss_research.optimizacion.variantes_rl import (
    POLICY_VARIANTS,
    PolicyVariantSpec,
    policy_variant,
)
from truss_research.surrogates.configuracion import PredictorConfig
from truss_research.surrogates.experimentos import read_run, train_predictor


@pytest.fixture
def predictor(design_data: Path) -> Path:
    destination = design_data.parent / "predictor"
    train_predictor(PredictorConfig(design_data, destination, model="dummy", folds=2))
    return destination


def policy_config(predictor: Path, variant: PolicyVariantSpec) -> PolicyConfig:
    common = dict(transitions=12, hidden_width=8, batch_size=4, validation_interval=6)
    parameters = (
        PPOParameters(**common, rollout_steps=4, epochs=2)
        if variant.algorithm == "ppo"
        else DQNParameters(**common, warmup=2, target_interval=2)
    )
    return PolicyConfig(
        predictor.parent / "designs",
        predictor,
        predictor.parent / variant.name,
        algorithm=variant.algorithm,
        reward_mode=variant.reward_mode,
        enhanced_state=variant.enhanced_state,
        use_predictions=variant.use_predictions,
        parameters=parameters,
        catalog_size=8,
        budgets=(3, 5),
        validation_episodes=1,
    )


@pytest.mark.parametrize(
    "variant",
    (*POLICY_VARIANTS, policy_variant("rl_double_dqn_trapezoidal_basic_state")),
    ids=lambda part: part.name,
)
def test_every_variant_trains_loads_and_accounts(
    predictor: Path, variant: PolicyVariantSpec
) -> None:
    config = policy_config(predictor, variant)
    history = train_policy(config)
    policy = load_policy(config.output_dir, predictor)
    bank, _, _ = policy_inputs(config)
    _, environment = sample_episode(bank, (config, np.random.default_rng(42)))
    action = policy.choose(environment.observe())
    assert environment.observe().available[action]
    assert len(history) == 2 and history.step.iloc[-1] == 12
    assert history.optimization_cpu_seconds.ge(0).all()
    run = read_run(config.output_dir)
    assert run["algorithm"] == variant.algorithm and run["completed_steps"] == 12
    assert run["config"]["reward_mode"] == variant.reward_mode
    assert run["config"]["enhanced_state"] == variant.enhanced_state
    assert run["optimizer_updates"] > 0 and run["parameter_count"] > 0
    if variant.name == "rl_double_dqn_trapezoidal_basic_state":
        from truss_research.evaluacion.informe_rl import export_policy_diagnostics

        destination = config.output_dir / "diagnostics"
        destination.mkdir()
        export_policy_diagnostics(
            (
                {
                    "key": f"policy/{variant.name}/42",
                    "directory": str(config.output_dir),
                },
            ),
            destination,
        )
        summary = pd.read_csv(destination / "policy_training_summary.csv").iloc[0]
        assert summary.algorithm == "double_dqn"
        assert summary.reward_mode == "trapezoidal"
        assert not summary.enhanced_state
    if variant.algorithm == "ppo":
        assert run["samples_used"] == 24
        assert (
            history[
                [
                    "loss_objective",
                    "loss_critic",
                    "entropy",
                    "kl_approx",
                    "clip_fraction",
                ]
            ]
            .notna()
            .all()
            .all()
        )


def test_trapezoidal_basic_state_variant_is_configured_explicitly(
    tmp_path: Path,
) -> None:
    from truss_research.optimizacion.contratos import StudyTask
    from truss_research.optimizacion.politica import configured_policy
    from truss_research.optimizacion.protocolo import load_protocol

    name = "rl_double_dqn_trapezoidal_basic_state"
    study = load_protocol(Path("config/study.json"), (tmp_path, tmp_path))
    study = replace(
        study,
        experiment="catalog",
        split_protocol="design",
        budgets=(8, 16, 24),
        evolution=replace(study.evolution, population_size=4),
    )
    assert name in study.rl.variants
    task = StudyTask(f"policy/{name}/42", "policy", variant=name)
    policy = configured_policy(
        study, (tmp_path / "predictor", tmp_path / "policy", None), task
    )
    assert policy.algorithm == "double_dqn"
    assert policy.reward_mode == "trapezoidal"
    assert not policy.enhanced_state
    assert policy.use_predictions


@pytest.mark.parametrize("algorithm", tuple(NETWORK_TYPES))
def test_all_networks_permute_and_mask(algorithm: str) -> None:
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    values = rng.normal(size=(8, 6)).astype(np.float32)
    available = np.array([False, True, True, False, True, True, True, True])
    state = EncodedState(values, available, np.array([0.5, 0.75], dtype=np.float32))
    network = NETWORK_TYPES[algorithm](6, 8)
    order = rng.permutation(len(available))
    permuted = replace(state, values=values[order], available=available[order])
    first, second = network(tensor_states([state])), network(tensor_states([permuted]))
    torch.testing.assert_close(first[:, order], second)
    assert first[0, ~torch.as_tensor(available)].lt(-1e20).all()
    if isinstance(network, CandidatePpoNetwork):
        torch.testing.assert_close(
            network.state_value(tensor_states([state])),
            network.state_value(tensor_states([permuted])),
        )
    larger = replace(
        state, values=np.tile(values, (8, 1)), available=np.tile(available, 8)
    )
    assert network(tensor_states([larger])).shape == (1, 64)


def test_dueling_centers_only_available_advantages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = DuelingCandidateQNetwork(2, 4, enhanced_state=False)
    for parameter in network.parameters():
        torch.nn.init.zeros_(parameter)
    network.value_head[-1].bias.data.fill_(2.0)
    state = EncodedState(
        np.ones((3, 2), dtype=np.float32),
        np.array([False, True, True]),
        np.ones(2, dtype=np.float32),
    )
    scores = network(tensor_states([state]))
    torch.testing.assert_close(scores[0, 1:], torch.tensor([2.0, 2.0]))
    monkeypatch.setattr(
        network,
        "representation",
        lambda state: (torch.tensor([[[100.0], [1.0], [3.0]]]), torch.zeros((1, 10))),
    )
    network.scorer = torch.nn.Identity()
    torch.testing.assert_close(
        network(tensor_states([state]))[0, 1:], torch.tensor([1.0, 3.0])
    )
    terminal = replace(state, available=np.zeros(3, dtype=bool))
    assert torch.isfinite(network(tensor_states([terminal]))).all()


def test_catalog_rng_is_independent_of_action_and_batch_sampling(
    predictor: Path,
) -> None:
    config = policy_config(predictor, POLICY_VARIANTS[0])
    bank = policy_inputs(config)[0]
    catalogs = []
    for variant in POLICY_VARIANTS:
        catalog_rng, exploration_rng, batch_rng = training_generators(42)
        episodes = []
        for _ in range(4):
            episodes.append(
                sample_episode(
                    bank,
                    (
                        replace(config, use_predictions=variant.use_predictions),
                        catalog_rng,
                    ),
                )[0]
            )
            exploration_rng.random(7 if variant.use_predictions else 20)
            batch_rng.random(11 if variant.algorithm == "ppo" else 99)
        catalogs.append(episodes)
    assert all(episodes == catalogs[0] for episodes in catalogs)


@pytest.mark.parametrize(
    "scenario",
    [
        (interruption, device)
        for interruption in ("checkpoint", "update")
        for device in ("cpu", "cuda")
    ],
)
def test_ppo_exact_resume_after_completed_rollout(
    predictor: Path, monkeypatch: pytest.MonkeyPatch, scenario: tuple[str, str]
) -> None:
    interruption, device = scenario
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requiere PyTorch CUDA y GPU")
    config = replace(policy_config(predictor, POLICY_VARIANTS[4]), device=device)
    original = PpoSession.checkpoint
    original_update = PpoLearner.update

    def interrupt(session: PpoSession) -> None:
        original(session)
        if session.step == 4 and interruption == "checkpoint":
            raise RuntimeError("interrupción después de rollout")

    def interrupt_update(learner: PpoLearner, batch: TensorDict) -> dict[str, float]:
        result = original_update(learner, batch)
        if interruption == "update":
            raise RuntimeError("interrupción durante actualización")
        return result

    monkeypatch.setattr(PpoSession, "checkpoint", interrupt)
    monkeypatch.setattr(PpoLearner, "update", interrupt_update)
    with pytest.raises(RuntimeError, match="interrupción"):
        train_policy(config)
    monkeypatch.setattr(PpoSession, "checkpoint", original)
    monkeypatch.setattr(PpoLearner, "update", original_update)
    resumed = replace(
        config, output_dir=predictor.parent / "resumed", resume_dir=config.output_dir
    )
    continuous = replace(config, output_dir=predictor.parent / "continuous")
    train_policy(resumed)
    train_policy(continuous)
    a = torch.load(resumed.output_dir / "checkpoint.pt", weights_only=False)
    b = torch.load(continuous.output_dir / "checkpoint.pt", weights_only=False)
    assert isinstance(a, TorchRLCheckpoint) and a.step == b.step == 12
    for name in a.online:
        torch.testing.assert_close(a.online[name], b.online[name], rtol=0, atol=0)
    assert a.episode == b.episode and a.actions == b.actions
    assert (
        a.optimizer_updates == b.optimizer_updates and a.samples_used == b.samples_used
    )
    for left, right in zip(a.history, b.history, strict=True):
        assert {
            name: value for name, value in left.items() if "seconds" not in name
        } == {name: value for name, value in right.items() if "seconds" not in name}
    invalid = replace(
        resumed,
        output_dir=predictor.parent / "invalid",
        parameters=replace(config.parameters, transitions=14),
    )
    with pytest.raises(ValueError, match="configuración"):
        train_policy(invalid)


def test_ppo_gae_cuts_terminals_and_bootstraps_rollout_boundary() -> None:
    learner = PpoLearner(CandidatePpoNetwork(2, 4), PPOParameters())
    values = np.ones((3, 2), dtype=np.float32)
    state = EncodedState(
        values, np.array([False, True, True]), np.ones(2, dtype=np.float32)
    )
    terminal = torch.tensor([[False], [True], [False]])
    batch = TensorDict(
        {
            "state": tensor_states([state] * 3),
            "state_value": torch.tensor([[0.5], [0.7], [1.0]]),
            "next": TensorDict(
                {
                    "state": tensor_states([state] * 3),
                    "state_value": torch.tensor([[0.7], [9.0], [4.0]]),
                    "reward": torch.tensor([[1.0], [2.0], [3.0]]),
                    "done": terminal,
                    "terminated": terminal,
                },
                batch_size=[3],
            ),
        },
        batch_size=[3],
    )
    learner.gae(batch)
    torch.testing.assert_close(
        batch["advantage"], torch.tensor([[2.435], [1.3], [6.0]])
    )
    torch.testing.assert_close(
        batch["value_target"], torch.tensor([[2.935], [2.0], [7.0]])
    )


def test_ppo_clipped_objective_matches_calculable_example() -> None:
    network = CandidatePpoNetwork(2, 4)
    for parameter in network.parameters():
        torch.nn.init.zeros_(parameter)
    learner = PpoLearner(network, PPOParameters())
    state = EncodedState(
        np.ones((3, 2), dtype=np.float32),
        np.array([False, True, True]),
        np.ones(2, dtype=np.float32),
    )
    current = torch.full((2,), -float(np.log(2)))
    batch = TensorDict(
        {
            "state": tensor_states([state] * 2),
            "action": torch.ones(2, dtype=torch.long),
            "action_log_prob": current - torch.tensor([1.5, 0.5]).log(),
            "advantage": torch.tensor([[1.0], [-1.0]]),
            "value_target": torch.tensor([[2.0], [4.0]]),
        },
        batch_size=[2],
    )
    losses = learner.loss(batch)
    torch.testing.assert_close(losses["loss_objective"], torch.tensor(-0.2))
    torch.testing.assert_close(losses["loss_critic"], torch.tensor(5.0))
    torch.testing.assert_close(
        losses["entropy"], torch.tensor(float(np.log(2)), dtype=torch.float32)
    )
    assert losses["clip_fraction"] == 1
    metrics = learner.update(batch)
    assert all(np.isfinite(value) for value in metrics.values())
    assert network.value_head[-1].bias.ne(0).any()


@pytest.mark.parametrize(
    "variant",
    [part for part in POLICY_VARIANTS if not part.use_predictions],
    ids=lambda part: part.name,
)
def test_each_ablation_ignores_prediction_values(
    predictor: Path, variant: PolicyVariantSpec
) -> None:
    config = policy_config(predictor, variant)
    bank, _, encoder = policy_inputs(config)
    _, environment = sample_episode(bank, (config, np.random.default_rng(42)))
    observation = environment.observe()
    changed = replace(observation, predictions=observation.predictions + 123.0)
    first, second = encoder.encode(observation), encoder.encode(changed)
    np.testing.assert_array_equal(first.values, second.values)


def test_ppo_single_action_and_single_transition_are_finite() -> None:
    learner = PpoLearner(CandidatePpoNetwork(2, 4), PPOParameters())
    state = EncodedState(
        np.ones((3, 2), dtype=np.float32),
        np.array([False, True, False]),
        np.ones(2, dtype=np.float32),
    )
    batch = TensorDict({"state": tensor_states([state])}, batch_size=[1])
    with torch.no_grad():
        sampled = learner.actor(batch)
    assert sampled["action"].item() == 1 and sampled["action_log_prob"].item() == 0
    batch["advantage"] = torch.zeros((1, 1))
    batch["value_target"] = torch.zeros((1, 1))
    assert all(np.isfinite(value) for value in learner.update(batch).values())


def test_rl_contrasts_pair_seeds_without_cartesian_products() -> None:
    from truss_research.evaluacion.incertidumbre_seleccion import (
        paired_selection_differences,
    )

    rows = []
    for catalog in (42, 43):
        common = dict(
            split="validation",
            episode_id=f"{catalog}_0_8",
            catalog_seed=catalog,
            budget=8,
        )
        rows.append(
            {
                **common,
                "strategy": "random",
                "policy_training_seed": None,
                "normalized_auc": 0.1,
            }
        )
        rows.extend(
            {
                **common,
                "strategy": variant.name,
                "policy_training_seed": seed,
                "normalized_auc": index * 0.1 + seed * 0.001,
            }
            for index, variant in enumerate(POLICY_VARIANTS)
            for seed in (42, 43)
        )
    differences = paired_selection_differences(pd.DataFrame(rows))
    policies = differences.loc[differences.reference_strategy.ne("random")]
    assert len(policies) == 36
    assert policies.groupby(["strategy", "reference_strategy"]).size().eq(4).all()
    np.testing.assert_allclose(
        policies.loc[
            policies.strategy.eq("rl_ppo") & policies.reference_strategy.eq("rl"),
            "auc_difference",
        ],
        0.4,
    )
    assert not policies.duplicated(
        ["episode_id", "strategy", "reference_strategy", "policy_training_seed"]
    ).any()


@pytest.mark.parametrize(
    "payload",
    [
        {"algorithms": []},
        {"algorithms": ["ppo", "ppo"]},
        {"algorithms": ["unknown"]},
        {"algorithms": [True]},
        {"ppo": {"clip_epsilon": True}},
        {"ppo": {"discount": 0.9}},
        {"ppo": {"epochs": 0}},
        {"ppo": {"batch_size": 32}},
        {"dqn": {"epsilon_end": float("nan")}},
    ],
)
def test_rl_configuration_rejects_invalid_or_duplicated_parameters(
    payload: dict[str, list[str | bool] | dict[str, int | float | bool]],
) -> None:
    from truss_research.optimizacion.variantes_rl import read_rl_parameters

    with pytest.raises(ValueError):
        read_rl_parameters(payload)


def test_rl_plan_reports_complete_matrix_without_training(tmp_path: Path) -> None:
    import json

    from truss_research.optimizacion.protocolo import load_protocol, study_plan

    config = load_protocol(
        Path("config/study.json"), (tmp_path / "study", tmp_path / "absent_data")
    )
    plan = json.loads(study_plan(config, "todo"))
    assert plan["strategies"] == list(config.search.strategies)
    assert len(plan["strategies"]) == 9
    assert plan["policy_transitions"] == 1_000_000
    assert plan["initial_population"] == 8
    assert plan["evidence"] == ["historical_observed", "surrogate_estimated"]
    assert not config.output_dir.exists() and not config.data_dir.exists()
