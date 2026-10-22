"""Etapa SHAP con modelos y tablas sintéticos pequeños, sin campaña real ni GPU."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from truss_research.cerchas.schemas import INPUT_COLUMNS, OUTPUT_COLUMNS
from truss_research.datos.preparacion import prepare_dataset, source_digest
from truss_research.datos.procesados import JsonValue, load_design_split
from truss_research.datos.schemas import DatasetSpec, PreparationConfig, SplitConfig
from truss_research.evaluacion import explicabilidad as stage
from truss_research.evaluacion import informe_explicabilidad as report_stage
from truss_research.optimizacion.estudio_estado import (
    STUDY_SCHEMA_VERSION,
    TaskRecord,
    artifact_signatures,
)
from truss_research.optimizacion.protocolo import load_protocol
from truss_research.surrogates.configuracion import (
    SCHEMA_VERSION,
    ModelParameters,
    PredictorConfig,
)
from truss_research.surrogates.especializacion import (
    TargetwiseRegressor,
    scalar_predictor,
)
from truss_research.surrogates.experimentos import data_signature, read_run, write_json
from truss_research.surrogates.explicabilidad import (
    ADDITIVITY_TOLERANCE,
    AttributionJob,
    ExplanationConfig,
    compute_attributions,
)
from truss_research.surrogates.modelos import build_predictor
from truss_research.surrogates.persistencia import model_signatures, save_model


@pytest.fixture(scope="module")
def shap_study(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("shap_sintetico")
    source = root / "source.xlsx"
    index = np.arange(100)
    pd.DataFrame(
        {
            "Unnamed: 0": [f"{{{i // 10};{i % 10}}}" for i in index],
            "Mass": 1000 + 10 * index,
            "Average Util": 0.2 + index * 0.005,
            "Z Disp": 0.01 + index * 0.0005,
            "Web Config": index % 5,
            "N of Trusses": 10,
            "L-R Height": 0.2 + index * 0.03,
            "LM-RM Height": 2.0,
            "M Height": 3.0,
            "Subdiv1": 2,
            "Subdiv2": 3,
        }
    ).to_excel(source, sheet_name="TOTAL DATA", index=False)
    data = root / "data"
    prepare_dataset(
        PreparationConfig(
            source,
            data,
            dataset=DatasetSpec(10, 10),
            split=SplitConfig(protocol="first_appearance"),
            calibration_fraction=0.2,
        )
    )
    study = root / "study"
    config = replace(
        load_protocol(Path("config/smoke.json"), (study, data)),
        models=("dummy", "ridge", "svr"),
        seeds=(42,),
    )
    training = load_design_split(data).training
    tasks: list[TaskRecord] = []
    for family in config.models:
        destination = study / "predictors" / f"{family}_42"
        settings = PredictorConfig(data, destination, model=family, folds=2)
        model = build_predictor(settings)
        model.fit(
            training[list(INPUT_COLUMNS)], training[list(OUTPUT_COLUMNS)].to_numpy()
        )
        save_model(model, destination)
        write_json(
            destination / "run.json",
            {
                "schema_version": SCHEMA_VERSION,
                "config": {"model": family, "seed": 42},
                "model_signature": source_digest(destination / "model.joblib"),
                "model_artifacts": model_signatures(destination),
                "dataset_signature": data_signature(data),
            },
        )
        tasks.append(
            TaskRecord(
                f"predictor/{family}/42",
                "complete",
                directory=str(destination),
                signatures=artifact_signatures(destination),
            )
        )
    write_json(
        study / "study.json",
        {
            "schema_version": STUDY_SCHEMA_VERSION,
            "config": asdict(config),
            "tasks": [asdict(task) for task in tasks],
        },
    )
    validation = study / "evolution/validation"
    validation.mkdir(parents=True)
    write_json(validation / "complete.json", {"files": {}, "complete": True})
    write_json(
        study / "evolution/search.json", {"selected": ["ridge", "ridge", "dummy"]}
    )
    return study


def request_for(study: Path, destination: Path) -> stage.ExplanationRequest:
    return stage.ExplanationRequest(study, destination, mode="pilot", accelerator="cpu")


def test_plan_has_no_side_effects_and_pilot_is_common(
    shap_study: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "pilot"
    before = artifact_signatures(shap_study)
    plan = stage.explanation_plan(request_for(shap_study, destination))
    assert not destination.exists()
    assert {row["evaluation_limit"] for row in plan["jobs"]} == {5}
    assert {row["seed"] for row in plan["jobs"]} == {42}
    assert {row["device"] for row in plan["jobs"]} == {"cpu"}
    assert artifact_signatures(shap_study) == before


def test_pilot_resumes_without_changing_models_and_measures_work(
    shap_study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = request_for(shap_study, tmp_path / "pilot")
    before = artifact_signatures(shap_study)
    operation = stage.execute_explanation
    calls = 0

    def interrupt_second(raw: dict[str, JsonValue]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        operation(raw)

    monkeypatch.setattr(stage, "execute_explanation", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        stage.run_explanations(request)
    first = request.destination / "dummy_42/run.json"
    first_bytes = first.read_bytes()
    monkeypatch.setattr(stage, "execute_explanation", operation)
    completed = stage.run_explanations(request)
    assert first.read_bytes() == first_bytes
    costs = pd.read_csv(completed / "shap_costs.csv")
    assert set(costs.method) == {"constant", "linear_independent", "permutation"}
    assert costs.evaluation_designs.eq(5).all()
    assert costs.loc[costs.model.eq("svr"), "counted_prediction_rows"].iloc[0] > 5
    assert costs.attempt_wall_seconds.gt(0).all()
    assert costs.peak_rss_mib.gt(0).all()
    assert (
        costs.loc[costs.model.eq("svr"), "estimated_full_permutation_seconds"].iloc[0]
        > 0
    )
    assert pd.read_csv(completed / "shap_stability.csv").empty
    values = pd.read_csv(completed / "shap_values.csv")
    assert (
        values.groupby("model")
        .iter_id.apply(set)
        .apply(lambda ids: ids == set(values.iter_id))
        .all()
    )
    signed = artifact_signatures(completed)
    monkeypatch.setattr(
        stage,
        "execute_explanation",
        lambda raw: pytest.fail("No debe recalcular una etapa completa"),
    )
    assert stage.run_explanations(request) == completed
    assert artifact_signatures(completed) == signed
    assert artifact_signatures(shap_study) == before


def test_changed_protocol_and_damaged_results_are_rejected(
    shap_study: Path, tmp_path: Path
) -> None:
    request = request_for(shap_study, tmp_path / "pilot")
    destination = stage.run_explanations(request)
    changed = json.loads(stage.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed["batch_size"] = 64
    protocol = tmp_path / "changed.json"
    protocol.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="Cambió el protocolo"):
        stage.run_explanations(replace(request, config=protocol))
    (destination / "shap_global.csv").write_text("alterado", encoding="utf-8")
    with pytest.raises(ValueError, match="Cambió una salida"):
        stage.run_explanations(request)


def test_full_export_includes_shap_but_excludes_pilot(
    shap_study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reports"
    monkeypatch.setattr(stage, "results_root", lambda study: root)
    monkeypatch.setattr(report_stage, "results_root", lambda study: root)
    stage.run_explanations(request_for(shap_study, root / "explicabilidad_piloto_001"))
    report = tmp_path / "report"
    report.mkdir()
    report_stage.export_explanations(shap_study, report)
    assert not (report / "shap_global.csv").exists()
    full = stage.run_explanations(
        replace(request_for(shap_study, root / "explicabilidad_001"), mode="full")
    )
    report_stage.export_explanations(shap_study, report)
    assert (report / "shap_global.csv").read_bytes() == (
        full / "shap_global.csv"
    ).read_bytes()
    assert (report / "shap_provenance.json").is_file()
    assert pd.read_csv(report / "shap_costs.csv").evaluation_designs.eq(10).all()


@pytest.mark.parametrize("middle", ["ridge", "svr"])
def test_targetwise_uses_component_methods(
    shap_study: Path, tmp_path: Path, middle: str
) -> None:
    config = load_protocol(
        Path("config/smoke.json"), (shap_study, shap_study.parent / "data")
    )
    table = load_design_split(config.data_dir)
    features = table.training[list(INPUT_COLUMNS)]
    regressor = TargetwiseRegressor(PredictorConfig(model="targetwise"))
    regressor.components_ = []
    for index, family in enumerate(("dummy", middle, "random_forest")):
        settings = PredictorConfig(
            model=family, target_index=index, parameters=ModelParameters(trees=3)
        )
        component = scalar_predictor(settings)
        component.fit(features, table.training[OUTPUT_COLUMNS[index]].to_numpy())
        regressor.components_.append((settings, component))
    model = Pipeline([("regressor", regressor)])
    held = table.validation[list(INPUT_COLUMNS)].iloc[:2]
    predicted = model.predict(held)
    job = AttributionJob(
        model,
        (features.iloc[:5], held),
        ExplanationConfig(config.data_dir, tmp_path, tmp_path / "components"),
        "targetwise",
        predicted,
    )
    result = compute_attributions(job)
    assert [row["method"] for row in result.components] == [
        "constant",
        "linear_independent" if middle == "ridge" else "permutation",
        "tree_interventional",
    ]
    assert (result.permutation_seconds == 0) == (middle == "ridge")
    np.testing.assert_allclose(
        result.bases + result.values.sum(axis=1),
        predicted,
        rtol=ADDITIVITY_TOLERANCE,
        atol=ADDITIVITY_TOLERANCE,
    )


def test_rejects_scientific_output_location_and_invalid_parameters(
    shap_study: Path,
) -> None:
    with pytest.raises(ValueError, match="fuera del estudio"):
        stage.check_output_location(shap_study, shap_study / "shap")
    with pytest.raises(ValueError, match="enteros positivos"):
        stage.PilotParameters(True, 1, 5)
    with pytest.raises(ValueError, match="enteros positivos"):
        ExplanationConfig(Path("data"), Path("predictor"), Path("output"), batch_size=0)


def test_cuda_requested_is_passed_without_rewriting_fitted_model(
    shap_study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from truss_research.surrogates import explicabilidad

    predictor = shap_study / "predictors/ridge_42"
    original = explicabilidad.load_predictor
    devices: list[str | None] = []

    def load(directory: Path, device: str | None = None) -> Pipeline:
        devices.append(device)
        return original(directory, "cpu")

    monkeypatch.setattr(explicabilidad, "load_predictor", load)
    before = model_signatures(predictor)
    output = tmp_path / "explanation"
    config = stage.load_study_config(shap_study)
    explicabilidad.explain_model(
        ExplanationConfig(
            config.data_dir, predictor, output, device="cuda", evaluation_limit=2
        )
    )
    assert devices == ["cuda"]
    assert read_run(output)["execution_device"] == "cpu"
    assert model_signatures(predictor) == before


def test_only_pending_tabpfn_components_request_cuda(
    shap_study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = stage.explanation_plan(request_for(shap_study, tmp_path / "stage"))
    raw = plan["jobs"][0] | {"device": "cuda", "model": "targetwise"}
    monkeypatch.setattr(
        stage,
        "read_run",
        lambda directory: {
            "model_artifacts": {"components/0/mass_kg.tabpfn_fit": "signature"}
        },
    )
    assert stage.pending_cuda([raw])
    assert not stage.pending_cuda([raw | {"device": "cpu"}])
    monkeypatch.setattr(
        stage,
        "read_run",
        lambda directory: {
            "model_artifacts": {"components/0/model.joblib": "signature"}
        },
    )
    assert not stage.pending_cuda([raw])


def test_notebook_reads_shap_exports_without_computing(
    shap_study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from truss_research.evaluacion.presentacion import NotebookContext

    report = stage.run_explanations(
        request_for(shap_study, tmp_path / "synthetic_exports")
    )
    config = stage.load_study_config(shap_study)
    context = NotebookContext(
        shap_study, report, config, {"selected": ["ridge", "ridge", "dummy"]}
    )
    notebook = nbformat.read(Path("notebooks/02_prediccion.ipynb"), as_version=4)
    source = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "shap_path =" in cell.source
    )
    before = artifact_signatures(report)
    monkeypatch.setattr(plt, "show", lambda: None)
    exec(
        compile(source, "notebook_shap", "exec"),
        {
            "REPORT_DIR": report,
            "CONFIG": config,
            "context": context,
            "pd": pd,
            "plt": plt,
            "display": lambda *values: None,
            "Markdown": str,
        },
    )
    plt.close("all")
    assert artifact_signatures(report) == before
