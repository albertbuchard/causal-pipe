"""Temporal preprocessing helpers for lag-expanded causal discovery."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


CURRENT_SUFFIX = "__t"
BETWEEN_SUFFIX = "__between"


@dataclass
class TemporalPreprocessResult:
    """Output of temporal lag expansion."""

    data: pd.DataFrame
    variable_types: Dict[str, List[str]]
    metadata: Dict[str, Any]
    lagged_column_map: Dict[str, Dict[str, str]]
    knowledge_constraints: Dict[str, List[Tuple[str, str]]]
    bootstrap_unit: str
    bootstrap_cluster_labels: Optional[List[Any]]
    bootstrap_block_length: Optional[int]


def _all_variable_names(variable_types: Any) -> List[str]:
    names: List[str] = []
    for bucket in ("continuous", "ordinal", "nominal"):
        names.extend(list(getattr(variable_types, bucket, []) or []))
    return names


def _variable_bucket(variable_types: Any, variable: str) -> str:
    for bucket in ("continuous", "ordinal", "nominal"):
        if variable in (getattr(variable_types, bucket, []) or []):
            return bucket
    raise ValueError(f"Variable '{variable}' is not declared in variable_types")


def lagged_name(variable: str, lag: int) -> str:
    return f"{variable}__lag{lag}"


def current_name(variable: str) -> str:
    return f"{variable}{CURRENT_SUFFIX}"


def between_name(variable: str) -> str:
    return f"{variable}{BETWEEN_SUFFIX}"


def _parse_time_index(column: str) -> Optional[int]:
    if column.endswith(CURRENT_SUFFIX):
        return 0
    marker = "__lag"
    if marker not in column:
        return None
    try:
        return int(column.rsplit(marker, 1)[1])
    except ValueError:
        return None


def build_temporal_constraints(
    columns: List[str],
    *,
    variables: List[str],
    lags: List[int],
    allow_contemporaneous_edges: bool,
    force_autoregressive_edges: bool,
) -> Dict[str, List[Tuple[str, str]]]:
    """Build temporal edge constraints as ``(source, target)`` pairs."""

    forbidden: List[Tuple[str, str]] = []
    required: List[Tuple[str, str]] = []

    temporal_columns = [c for c in columns if _parse_time_index(c) is not None]
    for source in temporal_columns:
        source_lag = _parse_time_index(source)
        if source_lag is None:
            continue
        for target in temporal_columns:
            if source == target:
                continue
            target_lag = _parse_time_index(target)
            if target_lag is None:
                continue
            if source_lag < target_lag:
                forbidden.append((source, target))
            if (
                not allow_contemporaneous_edges
                and source_lag == 0
                and target_lag == 0
            ):
                forbidden.append((source, target))

    if force_autoregressive_edges and 1 in lags:
        for variable in variables:
            source = lagged_name(variable, 1)
            target = current_name(variable)
            if source in columns and target in columns:
                required.append((source, target))

    return {
        "forbidden": sorted(set(forbidden)),
        "required": sorted(set(required)),
    }


def expand_temporal_data(
    df: pd.DataFrame,
    temporal_config: Any,
    variable_types: Any,
) -> TemporalPreprocessResult:
    """Convert long temporal data into a lag-expanded cross-sectional table."""

    time_col = temporal_config.time_col
    id_col = temporal_config.id_col
    if time_col not in df.columns:
        raise ValueError(f"Temporal time_col '{time_col}' not found in data")
    if id_col is not None and id_col not in df.columns:
        raise ValueError(f"Temporal id_col '{id_col}' not found in data")

    declared_variables = _all_variable_names(variable_types)
    variables = list(temporal_config.variables or declared_variables)
    if not variables:
        raise ValueError("TemporalConfig variables resolved to an empty list")

    reserved = {time_col}
    if id_col is not None:
        reserved.add(id_col)
    overlap = reserved.intersection(variables)
    if overlap:
        raise ValueError(
            "Temporal variables must not include time_col or id_col: "
            f"{sorted(overlap)}"
        )
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Temporal variables missing from data: {missing}")
    for variable in variables:
        _variable_bucket(variable_types, variable)

    if (
        temporal_config.within_person_center
        or temporal_config.include_between_person_means
    ) and id_col is None:
        raise ValueError(
            "within_person_center and include_between_person_means require id_col"
        )

    non_continuous = [
        v for v in variables if _variable_bucket(variable_types, v) != "continuous"
    ]
    if temporal_config.within_person_center and non_continuous:
        raise ValueError(
            "within_person_center is only supported for continuous variables; "
            f"non-continuous variables: {non_continuous}"
        )

    sort_cols = [time_col] if id_col is None else [id_col, time_col]
    working = df.sort_values(sort_cols).copy()
    working = working.reset_index(drop=True)

    value_source = working.copy()
    between_source: Dict[str, pd.Series] = {}
    if temporal_config.within_person_center:
        for variable in variables:
            numeric = pd.to_numeric(working[variable], errors="coerce")
            means = numeric.groupby(working[id_col]).transform("mean")
            value_source[variable] = numeric - means
            between_source[variable] = means
    elif temporal_config.include_between_person_means:
        for variable in variables:
            numeric = pd.to_numeric(working[variable], errors="coerce")
            between_source[variable] = numeric.groupby(working[id_col]).transform(
                "mean"
            )

    lagged = pd.DataFrame(index=working.index)
    lagged_column_map: Dict[str, Dict[str, str]] = {}
    lag_columns: List[str] = []
    expanded_types = {"continuous": [], "ordinal": [], "nominal": []}

    groupby_obj = value_source.groupby(working[id_col], sort=False) if id_col else None
    for variable in variables:
        bucket = _variable_bucket(variable_types, variable)
        c_name = current_name(variable)
        lagged[c_name] = value_source[variable].values
        expanded_types[bucket].append(c_name)
        lagged_column_map[variable] = {"current": c_name}
        for lag in temporal_config.lags:
            l_name = lagged_name(variable, lag)
            if groupby_obj is None:
                lagged[l_name] = value_source[variable].shift(lag).values
            else:
                lagged[l_name] = groupby_obj[variable].shift(lag).values
            lag_columns.append(l_name)
            expanded_types[bucket].append(l_name)
            lagged_column_map[variable][f"lag{lag}"] = l_name

        if temporal_config.include_between_person_means:
            b_name = between_name(variable)
            lagged[b_name] = between_source[variable].values
            expanded_types["continuous"].append(b_name)
            lagged_column_map[variable]["between"] = b_name

    before_drop = len(lagged)
    if temporal_config.drop_rows_with_incomplete_lags:
        lagged = lagged.dropna(subset=lag_columns).copy()
    dropped_rows = before_drop - len(lagged)

    cluster_labels = None
    if id_col is not None:
        cluster_labels = working.loc[lagged.index, id_col].tolist()

    time_values = working.loc[lagged.index, time_col]
    bootstrap_unit = temporal_config.bootstrap_unit
    if bootstrap_unit is None:
        bootstrap_unit = "cluster" if id_col is not None else "block"
    if bootstrap_unit == "cluster" and id_col is None:
        raise ValueError("Temporal cluster bootstrap requires id_col")

    block_length = max(temporal_config.lags) + 1 if bootstrap_unit == "block" else None
    lagged = lagged.reset_index(drop=True)

    constraints = build_temporal_constraints(
        list(lagged.columns),
        variables=variables,
        lags=list(temporal_config.lags),
        allow_contemporaneous_edges=temporal_config.allow_contemporaneous_edges,
        force_autoregressive_edges=temporal_config.force_autoregressive_edges,
    )

    metadata = {
        "enabled": True,
        "time_col": time_col,
        "id_col": id_col,
        "lags": list(temporal_config.lags),
        "variables": variables,
        "generated_columns": list(lagged.columns),
        "lagged_column_map": lagged_column_map,
        "n_original_rows": int(len(df)),
        "n_lagged_rows": int(len(lagged)),
        "dropped_rows": int(dropped_rows),
        "subject_count": int(working[id_col].nunique()) if id_col else None,
        "time_min": _json_safe_scalar(time_values.min()) if len(time_values) else None,
        "time_max": _json_safe_scalar(time_values.max()) if len(time_values) else None,
        "allow_contemporaneous_edges": temporal_config.allow_contemporaneous_edges,
        "force_autoregressive_edges": temporal_config.force_autoregressive_edges,
        "within_person_center": temporal_config.within_person_center,
        "include_between_person_means": temporal_config.include_between_person_means,
        "bootstrap_unit": bootstrap_unit,
        "bootstrap_block_length": block_length,
        "background_knowledge": {
            "forbidden_edges": len(constraints["forbidden"]),
            "required_edges": len(constraints["required"]),
            "conflicts": [],
        },
    }

    return TemporalPreprocessResult(
        data=lagged,
        variable_types=expanded_types,
        metadata=metadata,
        lagged_column_map=lagged_column_map,
        knowledge_constraints=constraints,
        bootstrap_unit=bootstrap_unit,
        bootstrap_cluster_labels=cluster_labels,
        bootstrap_block_length=block_length,
    )


def _json_safe_scalar(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value.item() if hasattr(value, "item") else value


def _node_lookup_from_nodes(nodes: Optional[List[Any]]) -> Dict[str, Any]:
    if not nodes:
        return {}
    return {node.get_name(): node for node in nodes}


def merge_temporal_background_knowledge(
    base_knowledge: Any,
    constraints: Dict[str, List[Tuple[str, str]]],
    *,
    background_knowledge_cls: Any,
    nodes: Optional[List[Any]] = None,
) -> Tuple[Any, List[Dict[str, str]]]:
    """Merge temporal constraints into causal-learn background knowledge."""

    knowledge = base_knowledge if base_knowledge is not None else background_knowledge_cls()
    node_lookup = _node_lookup_from_nodes(nodes)
    conflicts: List[Dict[str, str]] = []

    for source, target in constraints.get("forbidden", []):
        if _is_required(knowledge, source, target, node_lookup):
            conflicts.append(
                {"type": "forbidden_skipped_required_exists", "source": source, "target": target}
            )
            continue
        _add_constraint(knowledge, "forbidden", source, target, node_lookup)

    for source, target in constraints.get("required", []):
        if _is_forbidden(knowledge, source, target, node_lookup):
            conflicts.append(
                {"type": "required_skipped_forbidden_exists", "source": source, "target": target}
            )
            continue
        _add_constraint(knowledge, "required", source, target, node_lookup)

    return knowledge, conflicts


def clone_background_knowledge(knowledge: Any) -> Any:
    """Best-effort copy of background knowledge to avoid mutating config objects."""

    if knowledge is None:
        return None
    try:
        return copy.deepcopy(knowledge)
    except Exception:
        return knowledge


def _add_constraint(
    knowledge: Any,
    kind: str,
    source: str,
    target: str,
    node_lookup: Dict[str, Any],
) -> None:
    node_method = f"add_{kind}_by_node"
    pattern_method = f"add_{kind}_by_pattern"
    if hasattr(knowledge, node_method) and source in node_lookup and target in node_lookup:
        getattr(knowledge, node_method)(node_lookup[source], node_lookup[target])
        return
    if hasattr(knowledge, pattern_method):
        getattr(knowledge, pattern_method)(source, target)
        return
    store_name = f"{kind}_edges"
    if not hasattr(knowledge, store_name):
        setattr(knowledge, store_name, [])
    getattr(knowledge, store_name).append((source, target))


def _is_required(
    knowledge: Any,
    source: str,
    target: str,
    node_lookup: Dict[str, Any],
) -> bool:
    return _check_knowledge(knowledge, "required", source, target, node_lookup)


def _is_forbidden(
    knowledge: Any,
    source: str,
    target: str,
    node_lookup: Dict[str, Any],
) -> bool:
    return _check_knowledge(knowledge, "forbidden", source, target, node_lookup)


def _check_knowledge(
    knowledge: Any,
    kind: str,
    source: str,
    target: str,
    node_lookup: Dict[str, Any],
) -> bool:
    method_names = [f"is_{kind}", f"is_{kind}_by_node"]
    for method_name in method_names:
        if not hasattr(knowledge, method_name):
            continue
        method = getattr(knowledge, method_name)
        try:
            if source in node_lookup and target in node_lookup:
                return bool(method(node_lookup[source], node_lookup[target]))
            return bool(method(source, target))
        except TypeError:
            continue
        except Exception:
            return False

    store_name = f"{kind}_edges"
    return (source, target) in getattr(knowledge, store_name, [])
