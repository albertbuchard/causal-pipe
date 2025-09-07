"""Bootstrap edge stability for the FAS algorithm."""

from typing import Dict, Tuple, Optional, List, Any, Set
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

    rng = np.random.default_rng(random_state)
    counts = defaultdict(int)
    graph_counts: Dict[Tuple[Tuple[str, str], ...], Tuple[int, List[Tuple[str, str]]]] = {}
    sepset_counts = defaultdict(Counter)

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
                for r in pool.imap_unordered(_fas_bootstrap_worker, seeds, chunksize=chunksize):
                    yield r

    for edges_repr, sepsets in _iter_results():
        for pair in edges_repr:
            counts[pair] += 1
        key = tuple(sorted(edges_repr))
        if key in graph_counts:
            graph_counts[key] = (graph_counts[key][0] + 1, graph_counts[key][1])
        else:
            graph_counts[key] = (1, list(edges_repr))
        for (i, j), S in sepsets.items():
            if i > j:
                i, j = j, i
            sepset_counts[(i, j)][frozenset(S)] += 1

    probs = {edge: c / resamples for edge, c in counts.items()}

    best_graph_with_bootstrap = None
    if graph_counts:
        prob_graphs = sorted(
            (
                (cnt / resamples, edges_list)
                for _edges_key, (cnt, edges_list) in graph_counts.items()
            ),
            reverse=True,
        )
        best_prob, best_edges = prob_graphs[0]
        graph_obj = make_graph(
            node_names, [(a, b, "TAIL", "TAIL") for a, b in best_edges]
        )
        best_sepsets = {
            k: set(max(cnt.items(), key=lambda x: x[1])[0])
            for k, cnt in sepset_counts.items()
        }
        best_graph_with_bootstrap = (best_prob, graph_obj, probs, best_sepsets)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for idx, (p, edges) in enumerate(prob_graphs[:3], start=1):
                g = make_graph(
                    node_names, [(a, b, "TAIL", "TAIL") for a, b in edges]
                )
                visualize_graph(
                    g,
                    title=f"Bootstrap Graph {idx} (p={p:.2f})",
                    show=False,
                    output_path=os.path.join(output_dir, f"graph_{idx}.png"),
                )

    return probs, best_graph_with_bootstrap

