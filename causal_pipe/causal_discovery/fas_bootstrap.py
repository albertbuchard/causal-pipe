"""Bootstrap edge stability for the FAS algorithm."""

from typing import Dict, Tuple, Optional, List, Any, Set, FrozenSet
import os
import multiprocessing as mp
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from causallearn.utils.cit import CIT
from causallearn.utils.FAS import fas
from causallearn.graph.GeneralGraph import GeneralGraph

from causal_pipe.utilities.graph_utilities import get_nodes_from_node_names
from .bootstrap_utils import make_graph
from .static_causal_discovery import visualize_graph

# Limit BLAS thread usage in child processes to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_fas_bootstrap_data = None
_fas_bootstrap_nodes = None
_fas_bootstrap_ci_method = None
_fas_bootstrap_kwargs = None
_fas_bootstrap_n = None


def _init_fas_bootstrap(data, node_names, ci_method, fas_kwargs):
    """Initializer to share data across FAS bootstrap worker processes."""
    global _fas_bootstrap_data, _fas_bootstrap_nodes
    global _fas_bootstrap_ci_method, _fas_bootstrap_kwargs, _fas_bootstrap_n
    _fas_bootstrap_data = data
    _fas_bootstrap_nodes = get_nodes_from_node_names(node_names=node_names)
    _fas_bootstrap_ci_method = ci_method
    _fas_bootstrap_kwargs = fas_kwargs
    _fas_bootstrap_n = data.shape[0]


def _to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to a numeric matrix based on CI test method."""
    method = (_fas_bootstrap_ci_method or "").lower()
    if method in {"gsq", "chisq", "g2"}:

        def _enc(s):
            if pd.api.types.is_categorical_dtype(s):
                return s.cat.codes
            if pd.api.types.is_object_dtype(s):
                return s.astype("category").cat.codes
            return s.astype("int64")

        return df.apply(_enc).to_numpy(copy=False)
    return df.astype("float64").to_numpy(copy=False)


def _fas_bootstrap_worker(seed: int):
    """Run a single FAS bootstrap iteration."""
    sample = _fas_bootstrap_data.sample(
        n=_fas_bootstrap_n, replace=True, random_state=seed
    )
    sample_matrix = _to_matrix(sample)
    cit = CIT(data=sample_matrix, method=_fas_bootstrap_ci_method)
    g, sepsets, _ = fas(
        data=sample_matrix,
        nodes=_fas_bootstrap_nodes,
        independence_test_method=cit,
        **_fas_bootstrap_kwargs,
    )

    edges_repr = []
    for edge in g.get_graph_edges():
        a = edge.get_node1().get_name()
        b = edge.get_node2().get_name()
        edges_repr.append((a, b))

    return edges_repr, sepsets


def bootstrap_fas_edge_stability(
    data: pd.DataFrame,
    resamples: int,
    *,
    random_state: Optional[int] = None,
    fas_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    n_jobs: Optional[int] = 1,
    edge_threshold: Optional[float] = None,
) -> Tuple[
    Dict[Tuple[str, str], float],
    Optional[
        Tuple[
            float,
            GeneralGraph,
            Dict[Tuple[str, str], float],
            Dict[FrozenSet[str], Set[str]],
        ]
    ],
]:
    """Estimate edge presence probabilities via bootstrapped FAS runs.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with variables as columns.
    resamples : int
        Number of bootstrap resamples.
    random_state : Optional[int], default None
        Seed for bootstrap sampling.
    fas_kwargs : Optional[Dict[str, Any]], default None
        Additional keyword arguments passed to ``fas``.
    output_dir : Optional[str], default None
        Directory to save the top bootstrap graphs.
    n_jobs : Optional[int], default 1
        Number of worker processes for parallel execution.
    edge_threshold : Optional[float], default None
        If provided, edges in the best bootstrap graph with probability below
        this threshold are removed before returning.
    """

    if resamples <= 0:
        return {}, None

    rng = np.random.default_rng(random_state)
    counts: Dict[FrozenSet[str], int] = defaultdict(int)
    graph_counts: Dict[FrozenSet[FrozenSet[str]], Tuple[int, List[Tuple[str, str]]]] = {}
    sepset_counts: Dict[FrozenSet[str], Counter[FrozenSet[str]]] = defaultdict(Counter)

    fas_kwargs = dict(fas_kwargs or {})
    node_names = list(data.columns)
    ci_method = fas_kwargs.pop("conditional_independence_method", "fisherz")

    if n_jobs in (None, 0, -1):
        n_jobs = os.cpu_count() or 1
    n_jobs = min(n_jobs, resamples)

    seeds = rng.integers(0, 2**32, size=resamples, dtype=np.uint32).tolist()

    # initialise globals for single-process path
    _init_fas_bootstrap(data, node_names, ci_method, fas_kwargs)

    def _iter_results():
        if n_jobs == 1:
            for s in seeds:
                yield _fas_bootstrap_worker(int(s))
        else:
            chunksize = max(1, len(seeds) // (n_jobs * 4))
            with mp.Pool(
                processes=n_jobs,
                initializer=_init_fas_bootstrap,
                initargs=(data, node_names, ci_method, fas_kwargs),
                maxtasksperchild=250,
            ) as pool:
                for r in pool.imap_unordered(
                    _fas_bootstrap_worker, seeds, chunksize=chunksize
                ):
                    yield r

    names = node_names
    for edges_repr, sepsets in _iter_results():
        for (a, b) in edges_repr:
            counts[frozenset((a, b))] += 1

        graph_key = frozenset(frozenset((a, b)) for (a, b) in edges_repr)
        if graph_key in graph_counts:
            graph_counts[graph_key] = (
                graph_counts[graph_key][0] + 1,
                graph_counts[graph_key][1],
            )
        else:
            graph_counts[graph_key] = (1, list(edges_repr))

        for (i, j), S in sepsets.items():
            ai, aj = names[i], names[j]
            key = frozenset((ai, aj))
            S_names = frozenset(names[k] for k in S)
            sepset_counts[key][S_names] += 1

    probs_unordered = {k: c / resamples for k, c in counts.items()}

    probs: Dict[Tuple[str, str], float] = {}
    for key_set, p in probs_unordered.items():
        a, b = tuple(key_set)
        probs[(a, b)] = p
        probs[(b, a)] = p

    best_graph_with_bootstrap = None
    if graph_counts:
        prob_graphs = sorted(
            (
                (cnt / resamples, edges_list)
                for _k, (cnt, edges_list) in graph_counts.items()
            ),
            key=lambda x: x[0],
            reverse=True,
        )
        best_prob, best_edges_display = prob_graphs[0]
        if edge_threshold is None:
            filtered_edges_display = best_edges_display
        else:
            filtered_edges_display = [
                (a, b)
                for (a, b) in best_edges_display
                if probs_unordered.get(frozenset((a, b)), 0.0) >= edge_threshold
            ]
        graph_obj = make_graph(
            node_names,
            [(a, b, "TAIL", "TAIL") for (a, b) in filtered_edges_display],
        )
        best_sepsets = {
            k: set(max(cnt.items(), key=lambda x: x[1])[0])
            for k, cnt in sepset_counts.items()
        }
        best_graph_with_bootstrap = (
            best_prob,
            graph_obj,
            probs,
            best_sepsets,
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for idx, (p, edges) in enumerate(prob_graphs[:3], start=1):
                g = make_graph(
                    node_names,
                    [(a, b, "TAIL", "TAIL") for (a, b) in edges],
                )
                visualize_graph(
                    g,
                    title=f"Bootstrap Graph {idx} (p={p:.2f})",
                    show=False,
                    output_path=os.path.join(output_dir, f"graph_{idx}.png"),
                )

    return probs, best_graph_with_bootstrap
