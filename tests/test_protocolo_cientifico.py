"""Coherencia de AUC, incertidumbre, costos y análisis con modelos congelados."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from truss_research.datos.preparacion import source_digest
from truss_research.evaluacion.incertidumbre_seleccion import selection_intervals
from truss_research.evaluacion.pareto import ObjectiveSpace
from truss_research.evaluacion.seleccion import EpisodeReport, selection_metrics
from truss_research.evaluacion.sensibilidad import run_sensitivity
from truss_research.evaluacion.utilidad_predictiva import (
    family_correlations,
    utility_comparison,
)
from truss_research.optimizacion.contratos import AnalysisParameters, EpisodeSpec
from truss_research.optimizacion.entorno import EpisodeData, create_episode
from truss_research.optimizacion.estudio import run_study
from truss_research.optimizacion.estudio_estado import StudyConfig, artifact_signatures
from truss_research.surrogates.configuracion import ModelParameters


def test_reward_matches_auc_and_prefers_early_quality() -> None:
    responses = np.array([[20.0, 0.3, 0.04], [10.0, 0.8, 0.01], [15.0, 0.5, 0.02]])
    space = ObjectiveSpace.fit(responses)
    data = EpisodeData(
        EpisodeSpec(("a", "b", "c"), 0, 3, 42, "area"),
        np.arange(21, dtype=float).reshape(3, 7),
        responses.copy(),
        responses,
        space,
        reward_scale=space.hypervolume(responses),
    )
    scores: list[float] = []
    for order in ((1, 2), (2, 1)):
        environment = create_episode(data)
        rewards = [environment.initial_reward]
        selected = [0]
        volumes = [space.hypervolume(responses[selected])]
        for action in order:
            rewards.append(environment.step(action)[1])
            selected.append(action)
            volumes.append(space.hypervolume(responses[selected]))
        metrics = selection_metrics(
            EpisodeReport(responses, tuple(selected), tuple(volumes), space)
        )
        assert sum(rewards) == pytest.approx(metrics["normalized_auc"])
        assert sum(rewards[1:]) == pytest.approx(
            float(metrics["normalized_auc"]) - rewards[0]
        )
        scores.append(sum(rewards))
    assert scores[0] > scores[1]
    original = create_episode(data).public_observation()
    changed = create_episode(
        replace(data, reward_scale=2 * data.reward_scale)
    ).public_observation()
    assert set(original) == {
        "features",
        "predictions",
        "observed",
        "action_mask",
        "known_mask",
        "counters",
        "budget",
    }
    for name in original:
        np.testing.assert_equal(original[name], changed[name])
    with pytest.raises(ValueError, match="escala privada"):
        create_episode(replace(data, reward_scale=float("nan")))


def test_zero_reference_has_zero_reward_and_undefined_auc() -> None:
    responses = np.array([[10.0, 0.0, 0.02], [11.0, 0.0, 0.03]])
    space = ObjectiveSpace.fit(responses)
    data = EpisodeData(
        EpisodeSpec(("a", "b"), 0, 2, 42, "empty"),
        np.arange(14, dtype=float).reshape(2, 7),
        responses,
        responses,
        space,
        reward_scale=0.0,
    )
    environment = create_episode(data)
    assert environment.initial_reward == environment.step(1)[1] == 0.0
    metrics = selection_metrics(EpisodeReport(responses, (0, 1), (0.0, 0.0), space))
    assert metrics["normalized_auc"] is None
    assert metrics["relative_metric_reason"] is not None


def test_crossed_bootstrap_preserves_pairs_and_budget_repetitions() -> None:
    rows: list[dict[str, str | int | float | None]] = []
    for block in (42, 43):
        for catalog in range(3):
            for budget in (2, 3):
                identity = {
                    "split": "test",
                    "catalog_seed": block,
                    "episode_id": f"{block}_{catalog}_{budget}",
                    "budget": budget,
                }
                baseline = 0.3 + 0.01 * catalog
                rows.append(
                    {
                        **identity,
                        "strategy": "random",
                        "policy_training_seed": None,
                        "normalized_auc": baseline,
                    }
                )
                for policy in (42, 43):
                    rows.append(
                        {
                            **identity,
                            "strategy": "rl",
                            "policy_training_seed": policy,
                            "normalized_auc": baseline + 0.1,
                        }
                    )
                    rows.append(
                        {
                            **identity,
                            "strategy": "rl_ablation",
                            "policy_training_seed": policy,
                            "normalized_auc": baseline + 0.05,
                        }
                    )
    table = pd.DataFrame(rows)
    settings = AnalysisParameters(bootstrap_repeats=100)
    first = selection_intervals(table, settings)
    pd.testing.assert_frame_equal(first, selection_intervals(table, settings))
    compared = first.loc[
        first.strategy.eq("rl") & first.reference_strategy.eq("random")
    ]
    np.testing.assert_allclose(
        compared[["mean_difference", "ci_lower", "ci_upper"]], 0.1
    )
    assert compared.policy_seeds.eq(2).all() and compared.catalog_blocks.eq(2).all()
    assert compared.paired_episodes.eq(6).all()
    ablation = first.loc[
        first.strategy.eq("rl") & first.reference_strategy.eq("rl_ablation")
    ]
    np.testing.assert_allclose(
        ablation[["mean_difference", "ci_lower", "ci_upper"]], 0.05
    )
    with pytest.raises(ValueError, match="Faltan episodios"):
        selection_intervals(table.drop(index=0), settings)


def test_spearman_counts_families_and_reports_undefined_cases() -> None:
    families = pd.DataFrame(
        {
            "split": ["test"] * 3,
            "budget": [8] * 3,
            "model": ["dummy", "ridge", "random_forest"],
            "normalized_mae": [0.3, 0.2, 0.1],
            "normalized_auc": [0.1, 0.2, 0.3],
        }
    )
    result = family_correlations(families).iloc[0]
    assert result.families == 3 and result.spearman == pytest.approx(-1.0)
    assert (
        family_correlations(families.iloc[:2]).iloc[0].undefined_reason
        == "fewer_than_three_families"
    )
    assert (
        family_correlations(families.assign(normalized_auc=0.5))
        .iloc[0]
        .undefined_reason
        == "constant_metric"
    )


def test_utility_rejects_mismatched_predictor_signature(tmp_path: Path) -> None:
    predictor, ranking = tmp_path / "predictor", tmp_path / "ranking"
    predictor.mkdir()
    ranking.mkdir()
    (predictor / "run.json").write_text(
        json.dumps(
            {
                "evaluation_split": "validation",
                "dataset_signature": "data",
                "model_signature": "model",
            }
        ),
        encoding="utf-8",
    )
    (ranking / "run.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "dataset_signature": "data",
                "predictor_signature": "other",
            }
        ),
        encoding="utf-8",
    )
    records = (
        {"key": "predictor/dummy/42", "directory": str(predictor)},
        {"key": "ranking/dummy/42", "directory": str(ranking)},
    )
    with pytest.raises(ValueError, match="firmas"):
        utility_comparison(records)


def test_closed_study_sensitivity_never_fits_or_changes_signed_artifacts(
    design_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = StudyConfig(
        tmp_path / "study",
        data_dir=design_data,
        models=("dummy", "ridge", "random_forest"),
        seeds=(42, 43),
        outer_folds=2,
        inner_folds=2,
        trials=0,
        catalog_size=4,
        budgets=(2, 3),
        episodes_per_seed=1,
        rl_steps=4,
        rl_batch=2,
        rl_warmup=2,
        rl_hidden=8,
        validation_interval=2,
        background_size=2,
        permutation_rounds=1,
        parameters=ModelParameters(trees=2, depth=2),
        analysis=AnalysisParameters(catalog_sizes=(4, 8), bootstrap_repeats=20),
    )
    run_study(config, "validacion")
    output = tmp_path / "sensitivity"
    with pytest.raises(ValueError, match="cierre completo"):
        run_sensitivity(config.output_dir, output)
    assert not output.exists()

    def forbidden_fit(
        self: Pipeline, features: pd.DataFrame, responses: pd.DataFrame
    ) -> None:
        raise AssertionError("La evaluación final y sensibilidad no ajustan modelos")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    run_study(config, "final", True)
    before = artifact_signatures(config.output_dir)
    run_sensitivity(config.output_dir, output)
    assert before == artifact_signatures(config.output_dir)
    catalogs = pd.read_csv(output / "catalog_sensitivity.csv")
    assert set(catalogs.catalog_size) == {4, 8}
    primary_root = config.output_dir / "tasks/decision/validation/attempt_001"
    original = json.loads((primary_root / "run.json").read_text(encoding="utf-8"))[
        "episodes"
    ]
    expanded = json.loads(
        (output / "catalog_8/validation/run.json").read_text(encoding="utf-8")
    )["episodes"]
    for base, larger in zip(original, expanded, strict=True):
        assert base["candidate_ids"] == larger["candidate_ids"][:4]
        assert larger["initial_index"] == base["initial_index"]
    reference = pd.read_csv(output / "reference_sensitivity.csv")
    recalculated = reference.loc[
        reference.split.eq("validation") & reference.reference_margin.eq(0.1)
    ]
    source = pd.read_csv(primary_root / "summary.csv")
    keys = ["episode_id", "comparison_id"]
    compared = source.merge(
        recalculated, on=keys, suffixes=("_source", "_new"), validate="one_to_one"
    )
    np.testing.assert_allclose(
        compared.normalized_auc_source, compared.normalized_auc_new, atol=1e-12
    )
    report = sorted((config.output_dir / "reports").glob("report_*"))[-1]
    costs = pd.read_csv(report / "predictive_costs.csv")
    assert (
        costs.loc[
            costs.split.eq("test"),
            ["fit_seconds", "tuning_seconds", "crossfit_seconds"],
        ]
        .eq(0)
        .all()
        .all()
    )
    assert set(costs.loc[costs.split.eq("test"), "batch_rows"]) == {10}
    assert (report / "prediction_utility_spearman.csv").is_file()
    dispersion = pd.read_csv(report / "predictive_seed_dispersion.csv")
    assert (
        dispersion.loc[dispersion.model.eq("dummy"), "pred_mass_kg_count"].eq(2).all()
    )
    assert dispersion.loc[dispersion.model.eq("dummy"), "pred_mass_kg_std"].eq(0).all()
    assert (report / "selection_intervals.csv").is_file()
    assert (output / "catalog_size_test_2.png").is_file()
    assert source_digest(config.output_dir / "freeze.json")
