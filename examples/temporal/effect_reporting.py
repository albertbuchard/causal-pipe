"""Small reporting helpers for the temporal examples."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib
import numpy as np

from causal_pipe import CausalPipe
from causal_pipe.causal_discovery.static_causal_discovery import visualize_graph

matplotlib.use("Agg")


_LAG_RE = re.compile(r"^(?P<base>.+)__lag(?P<lag>\d+)$")


def _lag_order(name: str) -> Optional[int]:
    if name.endswith("__t"):
        return 0
    match = _LAG_RE.match(name)
    if match:
        return int(match.group("lag"))
    return None


def _temporal_pair_label(left: str, right: str) -> str:
    left_lag = _lag_order(left)
    right_lag = _lag_order(right)
    if left_lag is None or right_lag is None or left_lag == right_lag:
        return f"{left} <-> {right}"
    if left_lag > right_lag:
        return f"{left} -> {right}"
    return f"{right} -> {left}"


def _ranked_nonzero_pairs(
    matrix: np.ndarray,
    columns: Sequence[str],
    top_n: int,
    only_temporal: bool,
) -> list[tuple[str, float]]:
    pairs = []
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            right = columns[j]
            value = float(matrix[i, j])
            if np.isclose(value, 0.0):
                continue
            if only_temporal and _lag_order(left) == _lag_order(right):
                continue
            pairs.append((_temporal_pair_label(left, right), value))
    pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    return pairs[:top_n]


def _expected_edge_value(
    matrix: np.ndarray,
    columns: Sequence[str],
    edge: str,
) -> Optional[float]:
    parts = [part.strip() for part in edge.split("->")]
    if len(parts) != 2 or parts[0] not in columns or parts[1] not in columns:
        return None
    return float(matrix[columns.index(parts[0]), columns.index(parts[1])])


def print_causal_effect_summary(
    pipe: CausalPipe,
    expected_edges: Iterable[str],
    top_n: int = 8,
    only_temporal: bool = True,
) -> None:
    """Print readable summaries from CausalPipe's effect matrices."""

    print("\nCausal effect estimation summary")
    print("--------------------------------")
    print(
        "These are graph-conditioned effect/association estimates on the "
        "lag-expanded variables. Pearson and Spearman are signed; MI is "
        "non-negative."
    )

    if not pipe.causal_effects:
        print("No causal effect estimates were produced.")
        return

    columns = list(pipe.preprocessed_data.columns)
    expected_edge_list = list(expected_edges)
    for method_name, matrix in pipe.causal_effects.items():
        method_label = getattr(method_name, "value", str(method_name))
        matrix = np.asarray(matrix)
        print(f"\nMethod: {method_label}")
        for label, value in _ranked_nonzero_pairs(
            matrix,
            columns=columns,
            top_n=top_n,
            only_temporal=only_temporal,
        ):
            print(f"  {label}: {value:.3f}")

        print("  Expected teaching edges:")
        for edge in expected_edge_list:
            value = _expected_edge_value(matrix, columns, edge)
            if value is None:
                print(f"    {edge}: unavailable")
            elif np.isclose(value, 0.0):
                print(f"    {edge}: 0.000 (not adjacent in the learned graph)")
            else:
                print(f"    {edge}: {value:.3f}")

        method_dir = os.path.join(pipe.output_path, "causal_effect", method_label)
        print(f"  Full matrix: {os.path.join(method_dir, method_label + '_results.json')}")
        print(f"  Plot: {os.path.join(method_dir, method_label + '_result.png')}")


def save_true_graph(graph, title: str, output_path: str, filename: str) -> str:
    """Save a teaching true-graph PNG without opening an interactive window."""

    figure_dir = Path(output_path) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / filename
    visualize_graph(graph, title=title, output_path=str(path), show=False)
    return str(path)
