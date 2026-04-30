import types

import pandas as pd
import pytest

from causal_pipe.temporal import (
    build_temporal_constraints,
    expand_temporal_data,
    merge_temporal_background_knowledge,
)


def _types(continuous=None, ordinal=None, nominal=None):
    return types.SimpleNamespace(
        continuous=continuous or [],
        ordinal=ordinal or [],
        nominal=nominal or [],
    )


def _cfg(**kwargs):
    defaults = dict(
        time_col="time",
        id_col=None,
        lags=[1],
        variables=None,
        allow_contemporaneous_edges=True,
        force_autoregressive_edges=False,
        drop_rows_with_incomplete_lags=True,
        within_person_center=False,
        include_between_person_means=False,
        bootstrap_unit=None,
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def test_single_series_lag_expansion_drops_incomplete_rows():
    df = pd.DataFrame({"time": [1, 2, 3], "x": [10, 20, 30], "y": [1, 2, 3]})

    result = expand_temporal_data(
        df,
        temporal_config=_cfg(lags=[1, 2]),
        variable_types=_types(continuous=["x", "y"]),
    )

    assert list(result.data.columns) == [
        "x__t",
        "x__lag1",
        "x__lag2",
        "y__t",
        "y__lag1",
        "y__lag2",
    ]
    assert len(result.data) == 1
    assert result.data.iloc[0]["x__t"] == 30
    assert result.data.iloc[0]["x__lag1"] == 20
    assert result.data.iloc[0]["x__lag2"] == 10
    assert result.metadata["dropped_rows"] == 2
    assert result.bootstrap_unit == "block"
    assert result.bootstrap_block_length == 3


def test_panel_lag_expansion_does_not_cross_subjects():
    df = pd.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "time": [1, 2, 1, 2],
            "x": [10, 20, 100, 200],
        }
    )

    result = expand_temporal_data(
        df,
        temporal_config=_cfg(id_col="id"),
        variable_types=_types(continuous=["x"]),
    )

    assert result.data["x__t"].tolist() == [20, 200]
    assert result.data["x__lag1"].tolist() == [10, 100]
    assert result.bootstrap_unit == "cluster"
    assert result.bootstrap_cluster_labels == ["a", "b"]
    assert "id" not in result.data.columns
    assert "time" not in result.data.columns


def test_temporal_validation_rejects_missing_columns_and_bad_centering():
    df = pd.DataFrame({"id": [1, 1], "time": [1, 2], "x": [1, 2], "group": ["a", "b"]})

    with pytest.raises(ValueError, match="missing"):
        expand_temporal_data(
            df,
            temporal_config=_cfg(variables=["missing"]),
            variable_types=_types(continuous=["x"]),
        )

    with pytest.raises(ValueError, match="only supported for continuous"):
        expand_temporal_data(
            df,
            temporal_config=_cfg(
                id_col="id", variables=["group"], within_person_center=True
            ),
            variable_types=_types(nominal=["group"]),
        )


def test_temporal_constraints_forbid_future_to_past_and_contemporaneous():
    constraints = build_temporal_constraints(
        ["x__t", "x__lag1", "y__t", "y__lag1"],
        variables=["x", "y"],
        lags=[1],
        allow_contemporaneous_edges=False,
        force_autoregressive_edges=True,
    )

    assert ("x__t", "x__lag1") in constraints["forbidden"]
    assert ("x__t", "y__t") in constraints["forbidden"]
    assert ("x__lag1", "x__t") in constraints["required"]
    assert ("y__lag1", "y__t") in constraints["required"]


def test_merge_temporal_background_knowledge_records_conflicts():
    class DummyKnowledge:
        def __init__(self):
            self.forbidden_edges = []
            self.required_edges = [("x__lag1", "x__t")]

        def is_required(self, source, target):
            return (source, target) in self.required_edges

        def is_forbidden(self, source, target):
            return (source, target) in self.forbidden_edges

    knowledge, conflicts = merge_temporal_background_knowledge(
        DummyKnowledge(),
        {"forbidden": [("x__lag1", "x__t")], "required": [("y__lag1", "y__t")]},
        background_knowledge_cls=DummyKnowledge,
    )

    assert conflicts == [
        {
            "type": "forbidden_skipped_required_exists",
            "source": "x__lag1",
            "target": "x__t",
        }
    ]
    assert ("y__lag1", "y__t") in knowledge.required_edges
