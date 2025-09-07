from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

def get_nondirected_edge(n1, n2):
    return Edge(n1, n2, Endpoint.CIRCLE, Endpoint.CIRCLE)

def get_undirected_edge(n1, n2):
    return Edge(n1, n2, Endpoint.TAIL, Endpoint.TAIL)

def get_directed_edge(n1, n2):
    return Edge(n1, n2, Endpoint.TAIL, Endpoint.ARROW)

def get_bidirected_edge(n1, n2):
    return Edge(n1, n2, Endpoint.ARROW, Endpoint.ARROW)
