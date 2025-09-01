from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint


def get_directed_edge(node1, node2):
    return Edge(node1, node2, Endpoint.TAIL, Endpoint.ARROW)


def get_bidirected_edge(node1, node2):
    return Edge(node1, node2, Endpoint.ARROW, Endpoint.ARROW)


def get_undirected_edge(node1, node2):
    return Edge(node1, node2, Endpoint.TAIL, Endpoint.TAIL)


def get_nondirected_edge(node1, node2):
    return Edge(node1, node2, Endpoint.TAIL, Endpoint.TAIL)
