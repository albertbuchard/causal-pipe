"""Bootstrap edge stability for the FAS algorithm."""

from typing import Dict, Tuple, Optional, List, Any, Set
import copy
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from causallearn.utils.cit import CIT
from causallearn.utils.FAS import fas
from causallearn.graph.GeneralGraph import GeneralGraph

from causal_pipe.utilities.graph_utilities import get_nodes_from_node_names
from .bootstrap_utils import make_graph
from .static_causal_discovery import visualize_graph

_fas_bootstrap_data = None
_fas_bootstrap_node_names = None
_fas_bootstrap_ci_method = None
_fas_bootstrap_kwargs = None
_fas_bootstrap_n = None


def _init_fas_bootstrap(data, node_names, ci_method, fas_kwargs):
    """Initializer to share data across FAS bootstrap worker processes."""
    global _fas_bootstrap_data, _fas_bootstrap_node_names
    global _fas_bootstrap_ci_method, _fas_bootstrap_kwargs, _fas_bootstrap_n
    _fas_bootstrap_data = data
    _fas_bootstrap_node_names = node_names
    _fas_bootstrap_ci_method = ci_method
    _fas_bootstrap_kwargs = fas_kwargs
    _fas_bootstrap_n = data.shape[0]


def _fas_bootstrap_worker(seed: int):
    """Run a single FAS bootstrap iteration."""
    sample = _fas_bootstrap_data.sample(
        n=_fas_bootstrap_n, replace=True, random_state=seed
    )
    nodes = get_nodes_from_node_names(node_names=_fas_bootstrap_node_names)
    cit = CIT(data=sample.values, method=_fas_bootstrap_ci_method)
    g, sepsets, _ = fas(
        data=sample.values,
        nodes=nodes,
        independence_test_method=cit,
        **_fas_bootstrap_kwargs,
    )

    edges_repr = []
    for edge in g.get_graph_edges():
        n1 = edge.get_node1().get_name()
        n2 = edge.get_node2().get_name()
        if n1 <= n2:
            pair = (n1, n2)
        else:
            pair = (n2, n1)
        edges_repr.append(pair)

    return edges_repr, sepsets


def bootstrap_fas_edge_stability(
    data: pd.DataFrame,
    resamples: int,
    *,
    random_state: Optional[int] = None,
    fas_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    n_jobs: Optional[int] = 1,
) -> Tuple[
    Dict[Tuple[str, str], float],
    Optional[
        Tuple[
            float,
            GeneralGraph,
            Dict[Tuple[str, str], float],
            Dict[Tuple[int, int], Set[int]],
        ]
    ],
]:
    """Estimate edge presence probabilities via bootstrapped FAS runs."""

    if resamples <= 0:
        return {}, None

    rng = np.random.RandomState(random_state)
    counts: Dict[Tuple[str, str], int] = {}
    graph_counts: Dict[
        Tuple[Tuple[str, str], ...], Tuple[int, List[Tuple[str, str]], Dict[Tuple[int, int], Set[int]]]
    ] = {}
    fas_kwargs = fas_kwargs or {}

    node_names = list(data.columns)
    ci_method = fas_kwargs.pop("conditional_independence_method", "fisherz")

    if n_jobs is None or n_jobs <= 0:
        n_jobs = 1
    n_jobs = min(n_jobs, resamples)

    seeds = rng.randint(0, 2**32, size=resamples)
    _init_fas_bootstrap(data, node_names, ci_method, fas_kwargs)
    if n_jobs == 1:
        results = [_fas_bootstrap_worker(s) for s in seeds]
    else:
        with mp.Pool(
            processes=n_jobs,
            initializer=_init_fas_bootstrap,
            initargs=(data, node_names, ci_method, fas_kwargs),
        ) as pool:
            results = pool.map(_fas_bootstrap_worker, seeds)

    for edges_repr, sepsets in results:
        for pair in edges_repr:
            counts[pair] = counts.get(pair, 0) + 1
        key = tuple(sorted(edges_repr))
        if key in graph_counts:
            graph_counts[key] = (
                graph_counts[key][0] + 1,
                graph_counts[key][1],
                graph_counts[key][2],
            )
        else:
            graph_counts[key] = (1, list(edges_repr), copy.deepcopy(sepsets))

    probs = {edge: c / resamples for edge, c in counts.items()}

    if probs:
        print("Edge presence probabilities from FAS bootstrap:")
        for (a, b), p in probs.items():
            print(f"  {a} -- {b}: {p:.2f}")

    best_graph_with_bootstrap = None
    graph_probs: List[Tuple[float, GeneralGraph, Dict[Tuple[int, int], Set[int]]]] = []
    if graph_counts:
        for edges_repr, (count, edges_list, seps) in graph_counts.items():
            prob = count / resamples
            graph_obj = make_graph(
                node_names, [(a, b, "TAIL", "TAIL") for a, b in edges_list]
            )
            graph_probs.append((prob, graph_obj, seps))

        if graph_probs:
            best_prob, best_graph, best_sepsets = max(
                graph_probs, key=lambda x: x[0]
            )
            best_graph_with_bootstrap = (
                best_prob,
                copy.deepcopy(best_graph),
                probs,
                copy.deepcopy(best_sepsets),
            )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            top_graphs = sorted(graph_probs, key=lambda x: x[0], reverse=True)[:3]
            for idx, (prob, graph_obj, _) in enumerate(top_graphs, start=1):
                title = f"Bootstrap Graph {idx} (p={prob:.2f})"
                out_path = os.path.join(output_dir, f"graph_{idx}.png")
                visualize_graph(
                    graph_obj, title=title, show=False, output_path=out_path
                )

    return probs, best_graph_with_bootstrap

