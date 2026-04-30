"""SEM-backed mediation analysis helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from causallearn.graph.GeneralGraph import GeneralGraph

from causal_pipe.pipe_config import (
    CausalPipeConfig,
    MediationAnalysisConfig,
    MediationSpec,
    MediationTemporalLags,
)
from causal_pipe.sem.sem import fit_sem_lavaan
from causal_pipe.temporal import current_name, lagged_name
from causal_pipe.utilities.utilities import dump_json_to


SemFitFunc = Callable[..., Dict[str, Any]]


@dataclass
class ResolvedMediationSpec:
    name: str
    original: MediationSpec
    treatment: str
    mediators: List[str]
    outcome: str
    covariates: List[str]
    temporal: bool
    temporal_note: Optional[str] = None


def analyze_mediation(
    data: pd.DataFrame,
    pipe_config: CausalPipeConfig,
    mediation_config: MediationAnalysisConfig,
) -> Dict[str, Any]:
    """Run a full CausalPipe workflow and then mediation analysis."""

    from causal_pipe.causal_pipe import CausalPipe

    pipe = CausalPipe(pipe_config)
    pipe.run_pipeline(data)
    return pipe.run_mediation_analysis(mediation_config=mediation_config)


def run_mediation_analysis_for_pipe(
    pipe: Any,
    data: Optional[pd.DataFrame] = None,
    mediation_config: Optional[MediationAnalysisConfig] = None,
    sem_fit_func: SemFitFunc = fit_sem_lavaan,
) -> Dict[str, Any]:
    """Run configured mediation analyses against a prepared ``CausalPipe``."""

    config = mediation_config or getattr(pipe.config, "mediation_config", None)
    if config is None:
        raise ValueError("mediation_config must be provided")

    if isinstance(config, dict):
        config = MediationAnalysisConfig(**config)

    if data is not None and pipe.preprocessed_data is None:
        pipe.preprocess_data(data)
        if pipe.undirected_graph is None:
            pipe.identify_skeleton()
        if pipe.directed_graph is None:
            pipe.orient_edges()

    if pipe.preprocessed_data is None:
        raise ValueError(
            "Mediation analysis requires preprocessed data. Run the pipeline first "
            "or pass raw data to run_mediation_analysis(data=...)."
        )

    results: Dict[str, Any] = {"analyses": {}, "config": config.model_dump()}
    out_root = os.path.join(pipe.output_path, "mediation")
    os.makedirs(out_root, exist_ok=True)

    for index, spec in enumerate(config.specs, start=1):
        resolved = resolve_mediation_spec(
            spec,
            columns=list(pipe.preprocessed_data.columns),
            temporal_metadata=pipe.temporal_metadata,
            lagged_column_map=pipe.lagged_column_map,
            index=index,
        )
        result = run_single_mediation(
            pipe.preprocessed_data,
            resolved,
            config,
            directed_graph=pipe.directed_graph,
            causal_effects=pipe.causal_effects,
            output_root=out_root,
            sem_fit_func=sem_fit_func,
        )
        results["analyses"][resolved.name] = result

    pipe.mediation_results = results
    dump_json_to(results, os.path.join(out_root, "mediation_results.json"))
    return results


def resolve_mediation_spec(
    spec: MediationSpec,
    *,
    columns: Sequence[str],
    temporal_metadata: Optional[Dict[str, Any]],
    lagged_column_map: Optional[Dict[str, Dict[str, str]]],
    index: int,
) -> ResolvedMediationSpec:
    """Resolve original or temporal variable names to concrete data columns."""

    name = spec.name or _slugify(
        f"{index}_{spec.treatment}_{'_'.join(spec.mediators)}_{spec.outcome}"
    )
    is_temporal = bool(temporal_metadata and temporal_metadata.get("enabled"))
    warnings: List[str] = []

    if spec.variables_are_lagged or not is_temporal:
        treatment = spec.treatment
        mediators = list(spec.mediators)
        outcome = spec.outcome
        covariates = list(spec.covariates)
        temporal_note = None
    else:
        lags = list(temporal_metadata.get("lags") or [])
        mapping = lagged_column_map or {}
        temporal_lags = spec.temporal_lags or _default_temporal_lags(
            lags, len(spec.mediators), warnings
        )
        if len(temporal_lags.mediators) != len(spec.mediators):
            raise ValueError(
                "temporal_lags.mediators must have the same length as mediators"
            )
        treatment = _resolve_temporal_column(
            spec.treatment, temporal_lags.treatment, mapping
        )
        mediators = [
            _resolve_temporal_column(variable, lag, mapping)
            for variable, lag in zip(spec.mediators, temporal_lags.mediators)
        ]
        outcome = _resolve_temporal_column(spec.outcome, temporal_lags.outcome, mapping)
        covariates = [
            _resolve_temporal_column(covariate, 0, mapping)
            for covariate in spec.covariates
        ]
        temporal_note = "; ".join(warnings) if warnings else None

    resolved_columns = [treatment, *mediators, outcome, *covariates]
    missing = [column for column in resolved_columns if column not in columns]
    if missing:
        raise ValueError(f"Mediation variables not found in analysis data: {missing}")

    return ResolvedMediationSpec(
        name=name,
        original=spec,
        treatment=treatment,
        mediators=mediators,
        outcome=outcome,
        covariates=covariates,
        temporal=is_temporal and not spec.variables_are_lagged,
        temporal_note=temporal_note,
    )


def run_single_mediation(
    data: pd.DataFrame,
    spec: ResolvedMediationSpec,
    config: MediationAnalysisConfig,
    *,
    directed_graph: Optional[GeneralGraph],
    causal_effects: Optional[Dict[str, Any]],
    output_root: str,
    sem_fit_func: SemFitFunc,
) -> Dict[str, Any]:
    """Fit and summarize one mediation specification."""

    out_dir = os.path.join(output_root, spec.name)
    os.makedirs(out_dir, exist_ok=True)

    model_strings = build_mediation_model_strings(spec)
    sem_outputs: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    if spec.temporal_note:
        warnings.append(spec.temporal_note)

    se = "bootstrap" if config.bootstrap_samples > 0 else None
    bootstrap = config.bootstrap_samples if config.bootstrap_samples > 0 else None
    for model_name, model_string in model_strings.items():
        try:
            sem_outputs[model_name] = sem_fit_func(
                data,
                model_string,
                estimator=config.estimator,
                se=se,
                bootstrap=bootstrap,
                exogenous_residual_covariances=True,
            )
        except TypeError:
            warnings.append(
                "SEM backend does not support bootstrap arguments; falling back "
                "to non-bootstrap SEM fitting."
            )
            try:
                sem_outputs[model_name] = sem_fit_func(
                    data,
                    model_string,
                    estimator=config.estimator,
                    exogenous_residual_covariances=True,
                )
            except TypeError:
                sem_outputs[model_name] = sem_fit_func(
                    data,
                    model_string,
                    estimator=config.estimator,
                )
        except Exception as exc:
            sem_outputs[model_name] = {}
            warnings.append(f"{model_name} SEM fit failed: {exc}")

    effects = summarize_mediation_effects(sem_outputs, spec)
    model_comparison = compare_mediation_models(sem_outputs)
    graph_evidence = summarize_graph_evidence(
        spec,
        directed_graph=directed_graph,
        causal_effects=causal_effects if config.include_effect_method_evidence else None,
        columns=list(data.columns),
        apply_background_knowledge=config.apply_background_knowledge,
    )

    if spec.original.require_discovered_path and not graph_evidence["all_path_edges_present"]:
        warnings.append("Required discovered mediation path is not fully present.")

    classification = classify_mediation(
        effects,
        model_comparison,
        graph_evidence,
        alpha=config.alpha,
        require_discovered_path=spec.original.require_discovered_path,
    )

    result = {
        "spec": {
            "original": spec.original.model_dump(),
            "resolved": {
                "treatment": spec.treatment,
                "mediators": spec.mediators,
                "outcome": spec.outcome,
                "covariates": spec.covariates,
                "temporal": spec.temporal,
            },
        },
        "classification": classification,
        "effects": effects,
        "model_comparison": model_comparison,
        "graph_evidence": graph_evidence,
        "models": model_strings,
        "warnings": warnings,
    }

    _write_mediation_outputs(result, out_dir)
    return result


def build_mediation_model_strings(spec: ResolvedMediationSpec) -> Dict[str, str]:
    """Build lavaan model strings for direct, full, and partial mediation."""

    return {
        "direct_only": _build_model(spec, include_direct=True, include_mediated=False),
        "full_mediation": _build_model(spec, include_direct=False, include_mediated=True),
        "partial_mediation": _build_model(spec, include_direct=True, include_mediated=True),
    }


def build_mediation_background_constraints(
    spec: ResolvedMediationSpec,
) -> Dict[str, List[Tuple[str, str]]]:
    """Return required forward and forbidden reverse mediation path constraints."""

    required = _mediation_path_edges(spec)
    forbidden = [(target, source) for source, target in required]
    return {"required": required, "forbidden": forbidden}


def summarize_mediation_effects(
    sem_outputs: Dict[str, Dict[str, Any]],
    spec: ResolvedMediationSpec,
) -> Dict[str, Any]:
    """Extract path and defined-parameter effects from fitted SEM outputs."""

    partial = sem_outputs.get("partial_mediation") or {}
    full = sem_outputs.get("full_mediation") or {}
    direct = sem_outputs.get("direct_only") or {}

    partial_defined = _defined_by_name(partial)
    full_defined = _defined_by_name(full)
    defined = partial_defined or full_defined

    total_indirect = _row_to_effect(
        defined.get("total_indirect")
        or defined.get("indirect")
        or full_defined.get("total_indirect")
        or full_defined.get("indirect")
    )
    total = _row_to_effect(defined.get("total") or full_defined.get("total"))
    direct_effect = _structural_effect(
        partial, lhs=spec.outcome, rhs=spec.treatment, fallback_name="c_prime"
    )
    direct_only = _structural_effect(
        direct, lhs=spec.outcome, rhs=spec.treatment, fallback_name="c"
    )

    indirect_effects: Dict[str, Any] = {}
    for name, row in defined.items():
        if name.startswith("indirect"):
            indirect_effects[name] = _row_to_effect(row)

    total_value = _effect_value(total)
    indirect_value = _effect_value(total_indirect)
    proportion_mediated = None
    if total_value not in (None, 0) and indirect_value is not None:
        proportion_mediated = indirect_value / total_value

    return {
        "total_effect": total or direct_only,
        "direct_effect": direct_effect,
        "direct_only_effect": direct_only,
        "indirect_effects": indirect_effects,
        "total_indirect_effect": total_indirect,
        "proportion_mediated": proportion_mediated,
        "path_coefficients": {
            "partial_mediation": partial.get("structural_model") or [],
            "full_mediation": full.get("structural_model") or [],
            "direct_only": direct.get("structural_model") or [],
        },
    }


def compare_mediation_models(
    sem_outputs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare direct, full, and partial SEM models using available fit metrics."""

    rows = []
    for model_name, output in sem_outputs.items():
        measures = output.get("fit_measures") or {}
        rows.append(
            {
                "model": model_name,
                "aic": _as_float(measures.get("aic")),
                "bic": _as_float(measures.get("bic")),
                "cfi": _as_float(measures.get("cfi.scaled") or measures.get("cfi")),
                "rmsea": _as_float(
                    measures.get("rmsea.scaled") or measures.get("rmsea")
                ),
                "srmr": _as_float(measures.get("srmr")),
                "npar": _as_float(measures.get("npar") or output.get("npar")),
                "r2": output.get("r2") or [],
            }
        )

    score_key = "bic" if any(row["bic"] is not None for row in rows) else "aic"
    scored = [row for row in rows if row.get(score_key) is not None]
    selected = min(scored, key=lambda row: row[score_key])["model"] if scored else None
    lookup = {row["model"]: row for row in rows}

    return {
        "models": rows,
        "selected_best_model": selected,
        "score_key": score_key,
        "partial_vs_full": _delta(lookup, "partial_mediation", "full_mediation", score_key),
        "partial_vs_direct": _delta(lookup, "partial_mediation", "direct_only", score_key),
    }


def classify_mediation(
    effects: Dict[str, Any],
    model_comparison: Dict[str, Any],
    graph_evidence: Dict[str, Any],
    *,
    alpha: float,
    require_discovered_path: bool,
) -> Dict[str, Any]:
    """Classify the mediation pattern from effects, model fit, and graph support."""

    indirect = effects.get("total_indirect_effect")
    direct = effects.get("direct_effect")
    direct_only = effects.get("direct_only_effect")

    indirect_present = _effect_significant(indirect, alpha)
    direct_present = _effect_significant(direct, alpha)
    direct_only_present = _effect_significant(direct_only, alpha)
    best = model_comparison.get("selected_best_model")
    partial_vs_full = model_comparison.get("partial_vs_full") or {}
    partial_not_worse_than_full = (
        partial_vs_full.get("delta") is None or partial_vs_full.get("delta") <= 10
    )

    if require_discovered_path and not graph_evidence.get("all_path_edges_present"):
        label = "no_clear_mediation"
    elif indirect_present and direct_present and partial_not_worse_than_full:
        label = "partial_mediation"
    elif indirect_present and not direct_present:
        label = "full_mediation"
    elif indirect_present and best == "full_mediation":
        label = "full_mediation"
    elif direct_only_present and not indirect_present:
        label = "direct_only"
    else:
        label = "no_clear_mediation"

    support_count = sum(
        [
            bool(indirect_present),
            bool(direct_present or label == "full_mediation"),
            bool(graph_evidence.get("all_path_edges_present")),
            bool(best in {"partial_mediation", "full_mediation", "direct_only"}),
        ]
    )
    if label == "no_clear_mediation":
        confidence = "weak"
    elif support_count >= 3:
        confidence = "strong"
    else:
        confidence = "mixed"

    return {
        "label": label,
        "confidence": confidence,
        "indirect_effect_significant": bool(indirect_present),
        "direct_effect_significant": bool(direct_present),
        "direct_only_effect_significant": bool(direct_only_present),
        "alpha": alpha,
    }


def summarize_graph_evidence(
    spec: ResolvedMediationSpec,
    *,
    directed_graph: Optional[GeneralGraph],
    causal_effects: Optional[Dict[str, Any]],
    columns: Sequence[str],
    apply_background_knowledge: bool,
) -> Dict[str, Any]:
    """Summarize discovered-graph and effect-matrix support for the path."""

    requested_edges = _mediation_path_edges(spec)
    constraints = build_mediation_background_constraints(spec)
    edge_rows = []
    for source, target in requested_edges:
        edge_rows.append(
            {
                "source": source,
                "target": target,
                **_graph_edge_status(directed_graph, source, target),
                "effect_method_evidence": _effect_method_values(
                    causal_effects, columns, source, target
                ),
            }
        )

    return {
        "requested_path_edges": edge_rows,
        "all_path_edges_present": all(row["present"] for row in edge_rows),
        "all_path_edges_oriented": all(row["oriented_forward"] for row in edge_rows),
        "background_knowledge_applied": apply_background_knowledge,
        "background_knowledge_constraints": constraints if apply_background_knowledge else {},
    }


def _build_model(
    spec: ResolvedMediationSpec,
    *,
    include_direct: bool,
    include_mediated: bool,
) -> str:
    lines: List[str] = ["# mediator regressions"]
    covariates = list(spec.covariates)

    if spec.original.mode == "parallel":
        for idx, mediator in enumerate(spec.mediators, start=1):
            rhs = [f"a{idx}*{spec.treatment}", *covariates]
            lines.append(f"{mediator} ~ {' + '.join(rhs)}")
    else:
        for idx, mediator in enumerate(spec.mediators, start=1):
            previous = spec.mediators[: idx - 1]
            rhs = [f"a{idx}*{spec.treatment}"]
            rhs.extend(
                [
                    f"d{prev_idx}_{idx}*{previous_mediator}"
                    for prev_idx, previous_mediator in enumerate(previous, start=1)
                ]
            )
            rhs.extend(covariates)
            lines.append(f"{mediator} ~ {' + '.join(rhs)}")

    lines.append("")
    lines.append("# outcome regression")
    outcome_rhs: List[str] = []
    if include_direct:
        direct_label = "c" if not include_mediated else "c_prime"
        outcome_rhs.append(f"{direct_label}*{spec.treatment}")
    if include_mediated:
        for idx, mediator in enumerate(spec.mediators, start=1):
            outcome_rhs.append(f"b{idx}*{mediator}")
    outcome_rhs.extend(covariates)
    if not outcome_rhs:
        outcome_rhs.append(f"c*{spec.treatment}")
    lines.append(f"{spec.outcome} ~ {' + '.join(outcome_rhs)}")

    if include_mediated:
        lines.append("")
        lines.append("# indirect effects")
        indirect_names = []
        if spec.original.mode == "serial" and len(spec.mediators) > 1:
            chain = "*".join(
                [
                    "a1",
                    *[
                        f"d{idx}_{idx + 1}"
                        for idx in range(1, len(spec.mediators))
                    ],
                    f"b{len(spec.mediators)}",
                ]
            )
            lines.append(f"indirect_serial := {chain}")
            indirect_names.append("indirect_serial")
            for idx in range(1, len(spec.mediators) + 1):
                name = f"indirect_m{idx}"
                lines.append(f"{name} := a{idx}*b{idx}")
                indirect_names.append(name)
        else:
            for idx in range(1, len(spec.mediators) + 1):
                name = "indirect" if len(spec.mediators) == 1 else f"indirect_m{idx}"
                lines.append(f"{name} := a{idx}*b{idx}")
                indirect_names.append(name)
        total_indirect_expr = " + ".join(indirect_names)
        lines.append(f"total_indirect := {total_indirect_expr}")
        if include_direct:
            lines.append("total := c_prime + total_indirect")
        else:
            lines.append("total := total_indirect")

    return "\n".join(lines)


def _default_temporal_lags(
    lags: Sequence[int],
    mediator_count: int,
    warnings: List[str],
) -> MediationTemporalLags:
    if any(lag >= 2 for lag in lags):
        return MediationTemporalLags(
            treatment=2,
            mediators=[1] * mediator_count,
            outcome=0,
        )
    warnings.append(
        "Only lag 1 is available; temporal mediation uses treatment at lag 1 "
        "with mediator and outcome at the current wave."
    )
    return MediationTemporalLags(
        treatment=1,
        mediators=[0] * mediator_count,
        outcome=0,
    )


def _resolve_temporal_column(
    variable: str,
    lag: int,
    mapping: Dict[str, Dict[str, str]],
) -> str:
    if variable not in mapping:
        return current_name(variable) if lag == 0 else lagged_name(variable, lag)
    key = "current" if lag == 0 else f"lag{lag}"
    if key not in mapping[variable]:
        raise ValueError(f"Temporal lag '{key}' not available for variable '{variable}'")
    return mapping[variable][key]


def _mediation_path_edges(spec: ResolvedMediationSpec) -> List[Tuple[str, str]]:
    if spec.original.mode == "parallel":
        return [
            *[(spec.treatment, mediator) for mediator in spec.mediators],
            *[(mediator, spec.outcome) for mediator in spec.mediators],
        ]
    edges = [(spec.treatment, spec.mediators[0])]
    edges.extend(zip(spec.mediators[:-1], spec.mediators[1:]))
    edges.append((spec.mediators[-1], spec.outcome))
    return edges


def _defined_by_name(output: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = output.get("defined_parameters") or []
    result = {}
    for row in rows:
        name = row.get("lhs") or row.get("label") or row.get("Parameter")
        if name:
            result[str(name)] = row
    return result


def _row_to_effect(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    estimate = _first_present(row, ["est", "Estimate", "Coefficient", "est.std"])
    return {
        "estimate": _as_float(estimate),
        "standardized_estimate": _as_float(
            _first_present(row, ["est.std", "Coefficient", "std.all"])
        ),
        "se": _as_float(_first_present(row, ["se", "SE"])),
        "z": _as_float(_first_present(row, ["z", "Z"])),
        "p_value": _as_float(_first_present(row, ["pvalue", "p-value", "p_value"])),
        "ci_lower": _as_float(_first_present(row, ["ci.lower", "ci_lower"])),
        "ci_upper": _as_float(_first_present(row, ["ci.upper", "ci_upper"])),
    }


def _structural_effect(
    output: Dict[str, Any],
    *,
    lhs: str,
    rhs: str,
    fallback_name: str,
) -> Optional[Dict[str, Any]]:
    for row in output.get("structural_model") or []:
        if row.get("LV") == lhs and row.get("Predictor") == rhs:
            effect = _row_to_effect(row)
            if effect is not None:
                effect["name"] = fallback_name
            return effect
    for row in output.get("unstandardized_parameter_estimates") or []:
        if row.get("lhs") == lhs and row.get("rhs") == rhs and row.get("op") == "~":
            effect = _row_to_effect(row)
            if effect is not None:
                effect["name"] = fallback_name
            return effect
    return None


def _effect_significant(effect: Optional[Dict[str, Any]], alpha: float) -> bool:
    if not effect:
        return False
    lower = effect.get("ci_lower")
    upper = effect.get("ci_upper")
    if lower is not None and upper is not None:
        return (lower > 0 and upper > 0) or (lower < 0 and upper < 0)
    p_value = effect.get("p_value")
    return bool(p_value is not None and p_value < alpha)


def _effect_value(effect: Optional[Dict[str, Any]]) -> Optional[float]:
    if not effect:
        return None
    value = effect.get("estimate")
    return _as_float(value)


def _graph_edge_status(
    graph: Optional[GeneralGraph],
    source: str,
    target: str,
) -> Dict[str, Any]:
    if graph is None:
        return {"present": False, "oriented_forward": False, "status": "graph_unavailable"}
    nodes = {node.get_name(): node for node in graph.get_nodes()}
    if source not in nodes or target not in nodes:
        return {"present": False, "oriented_forward": False, "status": "node_missing"}
    source_node = nodes[source]
    target_node = nodes[target]
    present = graph.is_adjacent_to(source_node, target_node)
    return {
        "present": bool(present),
        "oriented_forward": bool(
            present and graph.is_directed_from_to(source_node, target_node)
        ),
        "status": "present" if present else "missing",
    }


def _effect_method_values(
    causal_effects: Optional[Dict[str, Any]],
    columns: Sequence[str],
    source: str,
    target: str,
) -> Dict[str, Optional[float]]:
    if not causal_effects or source not in columns or target not in columns:
        return {}
    i = columns.index(source)
    j = columns.index(target)
    values = {}
    for method_name, matrix in causal_effects.items():
        try:
            label = getattr(method_name, "value", str(method_name))
            values[label] = float(np.asarray(matrix)[i, j])
        except Exception:
            continue
    return values


def _write_mediation_outputs(result: Dict[str, Any], out_dir: str) -> None:
    dump_json_to(result, os.path.join(out_dir, "mediation_results.json"))

    model_rows = result.get("model_comparison", {}).get("models", [])
    if model_rows:
        pd.DataFrame(model_rows).to_csv(
            os.path.join(out_dir, "model_comparison.csv"), index=False
        )

    coefficient_rows = []
    for model_name, rows in result.get("effects", {}).get("path_coefficients", {}).items():
        for row in rows:
            coefficient_rows.append({"model": model_name, **row})
    if coefficient_rows:
        pd.DataFrame(coefficient_rows).to_csv(
            os.path.join(out_dir, "path_coefficients.csv"), index=False
        )

    with open(os.path.join(out_dir, "mediation_summary.txt"), "w") as f:
        classification = result.get("classification", {})
        f.write(f"Classification: {classification.get('label')}\n")
        f.write(f"Confidence: {classification.get('confidence')}\n")
        f.write("\nResolved path:\n")
        spec = result.get("spec", {}).get("resolved", {})
        f.write(
            f"{spec.get('treatment')} -> {spec.get('mediators')} -> {spec.get('outcome')}\n"
        )


def _delta(
    lookup: Dict[str, Dict[str, Any]],
    model_a: str,
    model_b: str,
    key: str,
) -> Dict[str, Any]:
    a = lookup.get(model_a, {}).get(key)
    b = lookup.get(model_b, {}).get(key)
    if a is None or b is None:
        return {"metric": key, "delta": None}
    return {"metric": key, "delta": a - b, "preferred": model_a if a < b else model_b}


def _first_present(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_").lower()
