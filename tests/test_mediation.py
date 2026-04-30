import os
from types import SimpleNamespace

import pandas as pd

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from causal_pipe.mediation import (
    ResolvedMediationSpec,
    _install_mediation_background_knowledge,
    _restore_mediation_background_knowledge,
    build_mediation_background_constraints,
    build_mediation_model_strings,
    compare_mediation_models,
    resolve_mediation_spec,
    run_single_mediation,
)
from causal_pipe.pipe_config import (
    MediationAnalysisConfig,
    MediationSpec,
    MediationTemporalLags,
)


def _resolved(mode="parallel", mediators=None):
    mediators = mediators or ["m"]
    return ResolvedMediationSpec(
        name="x_m_y",
        original=MediationSpec(
            treatment="x",
            mediators=mediators,
            outcome="y",
            mode=mode,
        ),
        treatment="x",
        mediators=mediators,
        outcome="y",
        covariates=[],
        temporal=False,
    )


def _mock_sem(label):
    def fit(_data, model_string, **_kwargs):
        if "c_prime*x" in model_string:
            model = "partial"
        elif "b1*m" in model_string and "c*x" not in model_string:
            model = "full"
        else:
            model = "direct"

        indirect = {
            "full": 0.42,
            "partial": 0.38,
            "direct": 0.0,
        }[model]
        direct = {
            "full": None,
            "partial": 0.31 if label == "partial" else 0.02,
            "direct": 0.55,
        }[model]
        indirect_ci = {
            "full": (0.20, 0.64),
            "partial": (0.18, 0.58) if label != "direct" else (-0.05, 0.08),
            "direct": (-0.02, 0.02),
        }[model]
        direct_ci = (0.12, 0.48) if label == "partial" else (-0.07, 0.10)

        structural = [
            {"LV": "m", "Predictor": "x", "Coefficient": 0.7, "p-value": 0.001},
        ]
        if model == "partial":
            structural.append(
                {
                    "LV": "y",
                    "Predictor": "x",
                    "Coefficient": direct,
                    "ci.lower": direct_ci[0],
                    "ci.upper": direct_ci[1],
                    "p-value": 0.01 if label == "partial" else 0.6,
                }
            )
        if model == "direct":
            structural.append(
                {
                    "LV": "y",
                    "Predictor": "x",
                    "Coefficient": direct,
                    "ci.lower": 0.30,
                    "ci.upper": 0.80,
                    "p-value": 0.001,
                }
            )

        defined = []
        if model != "direct":
            defined = [
                {
                    "lhs": "indirect",
                    "est": indirect,
                    "ci.lower": indirect_ci[0],
                    "ci.upper": indirect_ci[1],
                    "pvalue": 0.001 if label != "direct" else 0.4,
                },
                {
                    "lhs": "total_indirect",
                    "est": indirect,
                    "ci.lower": indirect_ci[0],
                    "ci.upper": indirect_ci[1],
                    "pvalue": 0.001 if label != "direct" else 0.4,
                },
                {"lhs": "total", "est": indirect + (direct or 0.0)},
            ]

        bic = {"direct": 130.0, "full": 100.0, "partial": 104.0}[model]
        if label == "partial":
            bic = {"direct": 140.0, "full": 112.0, "partial": 100.0}[model]
        if label == "direct":
            bic = {"direct": 90.0, "full": 120.0, "partial": 118.0}[model]

        return {
            "fit_measures": {
                "aic": bic - 5,
                "bic": bic,
                "cfi.scaled": 0.98,
                "rmsea.scaled": 0.03,
                "srmr": 0.02,
                "npar": 5,
            },
            "structural_model": structural,
            "defined_parameters": defined,
            "r2": [{"Item": "y", "R2": 0.4}],
        }

    return fit


def _graph():
    nodes = {name: GraphNode(name) for name in ["x", "m", "y"]}
    graph = GeneralGraph(list(nodes.values()))
    graph.add_edge(Edge(nodes["x"], nodes["m"], Endpoint.TAIL, Endpoint.ARROW))
    graph.add_edge(Edge(nodes["m"], nodes["y"], Endpoint.TAIL, Endpoint.ARROW))
    return graph


def test_static_full_mediation_classification_and_outputs(tmp_path):
    data = pd.DataFrame({"x": [1, 2, 3], "m": [2, 3, 4], "y": [3, 4, 5]})
    result = run_single_mediation(
        data,
        _resolved(),
        MediationAnalysisConfig(specs=[_resolved().original], bootstrap_samples=10),
        directed_graph=_graph(),
        causal_effects=None,
        output_root=str(tmp_path),
        sem_fit_func=_mock_sem("full"),
    )

    assert result["classification"]["label"] == "full_mediation"
    assert result["effects"]["total_indirect_effect"]["estimate"] == 0.38
    assert result["graph_evidence"]["all_path_edges_present"] is True
    assert os.path.exists(tmp_path / "x_m_y" / "mediation_results.json")
    assert os.path.exists(tmp_path / "x_m_y" / "model_comparison.csv")


def test_static_partial_and_direct_only_classifications(tmp_path):
    data = pd.DataFrame({"x": [1, 2, 3], "m": [2, 3, 4], "y": [3, 4, 5]})

    partial = run_single_mediation(
        data,
        _resolved(),
        MediationAnalysisConfig(specs=[_resolved().original], bootstrap_samples=10),
        directed_graph=_graph(),
        causal_effects=None,
        output_root=str(tmp_path / "partial"),
        sem_fit_func=_mock_sem("partial"),
    )
    direct = run_single_mediation(
        data,
        _resolved(),
        MediationAnalysisConfig(specs=[_resolved().original], bootstrap_samples=10),
        directed_graph=_graph(),
        causal_effects=None,
        output_root=str(tmp_path / "direct"),
        sem_fit_func=_mock_sem("direct"),
    )

    assert partial["classification"]["label"] == "partial_mediation"
    assert direct["classification"]["label"] == "direct_only"


def test_parallel_and_serial_model_strings():
    parallel = build_mediation_model_strings(_resolved(mediators=["m1", "m2"]))
    serial = build_mediation_model_strings(_resolved(mode="serial", mediators=["m1", "m2"]))

    assert "m1 ~ a1*x" in parallel["partial_mediation"]
    assert "m2 ~ a2*x" in parallel["partial_mediation"]
    assert "y ~ c_prime*x + b1*m1 + b2*m2" in parallel["partial_mediation"]
    assert "m2 ~ a2*x + d1_2*m1" in serial["partial_mediation"]
    assert "indirect_serial := a1*d1_2*b2" in serial["partial_mediation"]


def test_temporal_mapping_lag2_and_lag1_warning():
    columns = ["x__lag2", "x__lag1", "m__lag1", "m__t", "y__t"]
    mapping = {
        "x": {"lag2": "x__lag2", "lag1": "x__lag1", "current": "x__t"},
        "m": {"lag1": "m__lag1", "current": "m__t"},
        "y": {"current": "y__t"},
    }
    spec = MediationSpec(treatment="x", mediators=["m"], outcome="y")
    lag2 = resolve_mediation_spec(
        spec,
        columns=columns,
        temporal_metadata={"enabled": True, "lags": [1, 2]},
        lagged_column_map=mapping,
        index=1,
    )
    lag1 = resolve_mediation_spec(
        spec,
        columns=columns,
        temporal_metadata={"enabled": True, "lags": [1]},
        lagged_column_map=mapping,
        index=1,
    )

    assert lag2.treatment == "x__lag2"
    assert lag2.mediators == ["m__lag1"]
    assert lag2.outcome == "y__t"
    assert lag1.treatment == "x__lag1"
    assert lag1.mediators == ["m__t"]
    assert "Only lag 1 is available" in lag1.temporal_note


def test_temporal_lag_override_and_background_constraints():
    spec = MediationSpec(
        treatment="x",
        mediators=["m"],
        outcome="y",
        temporal_lags=MediationTemporalLags(treatment=1, mediators=[1], outcome=0),
    )
    resolved = resolve_mediation_spec(
        spec,
        columns=["x__lag1", "m__lag1", "y__t"],
        temporal_metadata={"enabled": True, "lags": [1]},
        lagged_column_map={
            "x": {"lag1": "x__lag1"},
            "m": {"lag1": "m__lag1"},
            "y": {"current": "y__t"},
        },
        index=1,
    )
    constraints = build_mediation_background_constraints(resolved)

    assert resolved.mediators == ["m__lag1"]
    assert ("x__lag1", "m__lag1") in constraints["required"]
    assert ("m__lag1", "x__lag1") in constraints["forbidden"]


def test_mediation_background_knowledge_is_temporarily_installed():
    pipe = SimpleNamespace(
        preprocessed_data=pd.DataFrame({"x": [1, 2], "m": [2, 3], "y": [3, 4]}),
        temporal_metadata={},
        lagged_column_map={},
        skeleton_method=SimpleNamespace(knowledge=None),
        orientation_method=SimpleNamespace(background_knowledge=None),
    )
    config = MediationAnalysisConfig(
        specs=[MediationSpec(treatment="x", mediators=["m"], outcome="y")]
    )

    application = _install_mediation_background_knowledge(pipe, config)

    assert application["applied_to_discovery"] is True
    assert ("x", "m") in application["constraints"]["required"]
    assert ("m", "x") in application["constraints"]["forbidden"]
    assert pipe.skeleton_method.knowledge is not None
    assert pipe.orientation_method.background_knowledge is not None

    _restore_mediation_background_knowledge(pipe, application)

    assert pipe.skeleton_method.knowledge is None
    assert pipe.orientation_method.background_knowledge is None


def test_compare_models_flag_keeps_fit_rows_without_selecting_best():
    sem_outputs = {
        "direct_only": {"fit_measures": {"bic": 20.0}},
        "full_mediation": {"fit_measures": {"bic": 10.0}},
        "partial_mediation": {"fit_measures": {"bic": 12.0}},
    }

    comparison = compare_mediation_models(sem_outputs, compare_models=False)

    assert comparison["comparison_disabled"] is True
    assert comparison["selected_best_model"] is None
    assert len(comparison["models"]) == 3
