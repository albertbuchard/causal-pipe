"""Utility helpers for bootstrap-based causal discovery routines."""

from typing import List, Tuple

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode


def format_oriented_edge(a: str, b: str, orient: str) -> str:
    """Return a human-readable string for an oriented edge."""
    e1, e2 = orient.split("-")
    if e1 == "TAIL" and e2 == "ARROW":
        return f"{a} -> {b}"
    if e1 == "ARROW" and e2 == "TAIL":
        return f"{a} <- {b}"
    if e1 == "ARROW" and e2 == "ARROW":
        return f"{a} <-> {b}"
    if e1 == "TAIL" and e2 == "TAIL":
        return f"{a} -- {b}"
    if e1 == "CIRCLE" and e2 == "ARROW":
        return f"{a} o-> {b}"
    if e1 == "ARROW" and e2 == "CIRCLE":
        return f"{a} <-o {b}"
    if e1 == "CIRCLE" and e2 == "TAIL":
        return f"{a} o- {b}"
    if e1 == "TAIL" and e2 == "CIRCLE":
        return f"{a} -o {b}"
    if e1 == "CIRCLE" and e2 == "CIRCLE":
        return f"{a} o-o {b}"
    return f"{a} {e1}-{e2} {b}"


def make_graph(
    node_names: List[str],
    edges_repr: List[Tuple[str, str, str, str]],
) -> GeneralGraph:
    """Construct a ``GeneralGraph`` from edge representations."""

    names = set(node_names)
    for n1, n2, _, _ in edges_repr:
        names.add(n1)
        names.add(n2)

    name_to_node = {name: GraphNode(name) for name in names}
    g = GeneralGraph(list(name_to_node.values()))
    for n1, n2, e1, e2 in edges_repr:
        g.add_edge(Edge(name_to_node[n1], name_to_node[n2], Endpoint[e1], Endpoint[e2]))
    return g

