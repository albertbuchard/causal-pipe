"""Bootstrap edge orientation stability for the FCI algorithm."""

from typing import Dict, Tuple, Optional, List, Any
import copy
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from bcsl.fci import fci_orient_edges_from_graph_node_sepsets
from causallearn.graph.GeneralGraph import GeneralGraph

from causal_pipe.utilities.graph_utilities import (
    get_nodes_from_node_names,
    copy_graph,
    visualize_graph,
)
from .bootstrap_utils import format_oriented_edge, make_graph


_fci_bootstrap_data = None
_fci_bootstrap_graph = None
_fci_bootstrap_node_names = None
_fci_bootstrap_sepsets = None
_fci_bootstrap_kwargs = None
_fci_bootstrap_n = None


def _init_fci_bootstrap(data, graph, node_names, sepsets, fci_kwargs):
    """Initializer to share data across FCI bootstrap worker processes."""
    global _fci_bootstrap_data, _fci_bootstrap_graph
    global _fci_bootstrap_node_names, _fci_bootstrap_sepsets
    global _fci_bootstrap_kwargs, _fci_bootstrap_n
    _fci_bootstrap_data = data
    _fci_bootstrap_graph = graph
    _fci_bootstrap_node_names = node_names
    _fci_bootstrap_sepsets = sepsets
    _fci_bootstrap_kwargs = fci_kwargs
    _fci_bootstrap_n = data.shape[0]


def _fci_bootstrap_worker(seed: int):
    """Run a single FCI bootstrap iteration."""
    sample = _fci_bootstrap_data.sample(
        n=_fci_bootstrap_n, replace=True, random_state=seed
    )
    nodes = get_nodes_from_node_names(node_names=_fci_bootstrap_node_names)
    g, _ = fci_orient_edges_from_graph_node_sepsets(
        data=sample.values,
        graph=copy_graph(_fci_bootstrap_graph),
        nodes=nodes,
        sepsets=_fci_bootstrap_sepsets,
        **_fci_bootstrap_kwargs,
    )

    edges_repr = []
    for edge in g.get_graph_edges():
        n1 = edge.get_node1().get_name()
        n2 = edge.get_node2().get_name()
        e1 = edge.endpoint1
        e2 = edge.endpoint2
        if n1 <= n2:
            edges_repr.append((n1, n2, e1.name, e2.name))
        else:
            edges_repr.append((n2, n1, e2.name, e1.name))
    return edges_repr


def bootstrap_fci_edge_stability(
    data: pd.DataFrame,
    resamples: int,
    *,
    graph: GeneralGraph,
    nodes,
    sepsets: Dict[Tuple[int, int], Any],
    random_state: Optional[int] = None,
    fci_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    n_jobs: Optional[int] = 1,
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, float]],
    Optional[Tuple[float, GeneralGraph, Dict[Tuple[str, str], Dict[str, float]]]],
]:
    """Estimate edge orientation probabilities via bootstrapped FCI runs."""

    if resamples <= 0:
        return {}, None

    rng = np.random.RandomState(random_state)
    counts: Dict[Tuple[str, str], Dict[str, int]] = {}
    graph_counts: Dict[
        Tuple[Tuple[str, str, str, str], ...], Tuple[int, List[Tuple[str, str, str, str]]]
    ] = {}
    fci_kwargs = fci_kwargs or {}

    node_names = [node.get_name() for node in nodes]

    if n_jobs is None or n_jobs <= 0:
        n_jobs = 1
    n_jobs = min(n_jobs, resamples)

    seeds = rng.randint(0, 2**32, size=resamples)
    _init_fci_bootstrap(data, graph, node_names, sepsets, fci_kwargs)
    if n_jobs == 1:
        results = [_fci_bootstrap_worker(s) for s in seeds]
    else:
        with mp.Pool(
            processes=n_jobs,
            initializer=_init_fci_bootstrap,
            initargs=(data, graph, node_names, sepsets, fci_kwargs),
        ) as pool:
            results = pool.map(_fci_bootstrap_worker, seeds)

    for edges_repr in results:
        edges_repr_list = []
        for n1, n2, e1, e2 in edges_repr:
            pair = (n1, n2)
            orient = f"{e1}-{e2}"
            orient_counts = counts.setdefault(pair, {})
            orient_counts[orient] = orient_counts.get(orient, 0) + 1
            edges_repr_list.append((n1, n2, e1, e2))

        key = tuple(sorted(edges_repr_list))
        if key in graph_counts:
            graph_counts[key] = (graph_counts[key][0] + 1, graph_counts[key][1])
        else:
            graph_counts[key] = (1, list(edges_repr_list))

    probs = {
        edge: {o: c / resamples for o, c in orient_counts.items()}
        for edge, orient_counts in counts.items()
    }

    if probs:
        print("Edge orientation probabilities from FCI bootstrap:")
        for (a, b), orient_probs in probs.items():
            for orient, p in orient_probs.items():
                edge_str = format_oriented_edge(a, b, orient)
                print(f"  {edge_str}: {p:.2f}")

    best_graph_with_bootstrap = None
    graph_probs: List[Tuple[float, GeneralGraph]] = []
    if graph_counts:
        for edges_repr, (count, edges_list) in graph_counts.items():
            prob = count / resamples
            graph_obj = make_graph(node_names, edges_list)
            graph_probs.append((prob, graph_obj))

        if graph_probs:
            best_prob, best_graph = max(graph_probs, key=lambda x: x[0])
            best_graph_with_bootstrap = (
                best_prob,
                copy.deepcopy(best_graph),
                probs,
            )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            top_graphs = sorted(graph_probs, key=lambda x: x[0], reverse=True)[:3]
            for idx, (prob, graph_obj) in enumerate(top_graphs, start=1):
                title = f"Bootstrap Graph {idx} (p={prob:.2f})"
                out_path = os.path.join(output_dir, f"graph_{idx}.png")
                visualize_graph(
                    graph_obj, title=title, show=False, output_path=out_path
                )

    return probs, best_graph_with_bootstrap

