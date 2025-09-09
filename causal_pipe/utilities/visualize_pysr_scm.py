from typing import Dict, Any

import pydot

def graph_with_pysr_scm(pysr_results: Dict[str, Any],
                       show: bool = True,
                       title=None,
                       output_path=None,
                       labels=None
                       ) -> pydot.Dot:
    """
    Visualize the structural causal model (SCM) derived from PySR results using NetworkX and Matplotlib.

    :param pysr_results:
    :param show:
    :param title:
    :param output_path:
    :param labels:
    :return:
    """

    graph = pysr_results.get("final_graph", None)
    if graph is None:
        raise ValueError("No final graph found in PySR results.")

    scm = pysr_results.get("structural_equations", {})
    if not scm:
        raise ValueError("No structural equations found in PySR results.")

    G = pydot.Dot(title if title else "SCM from PySR", graph_type="digraph", fontsize=18)
    G.obj_dict["attributes"]["dpi"] = 300

    #     structural_equations[target_name] = {
    #             "equation": best["sympy_format"] if "sympy_format" in best else str(best),
    #             "best": best,
    #             "r2": r2,
    #             "parents": pnames,
    #         }
    nodes = list(scm.keys())
    if labels is not None:
        assert len(labels) == len(nodes), "Length of labels must match number of nodes."

    # Add nodes to the pydot graph
    for i, node in enumerate(nodes):
        node_label = labels[i] if labels is not None else node
        pydot_node = pydot.Node(str(i), label=node_label, shape="ellipse")
        G.add_node(pydot_node)

    node_to_id = {node: i for i, node in enumerate(nodes)}
    # Add edges based on structural equations
    for target, details in scm.items():
        target_id = node_to_id[target]
        parents = details.get("parents", [])
        for parent in parents:
            if parent not in node_to_id:
                continue
            parent_id = node_to_id[parent]
            edge = pydot.Edge(str(parent_id), str(target_id), arrowhead="normal", dir="forward")
            edge_equation = details.get("equation", "")
            r2 = details.get("r2", None)
            label_parts = []
            if edge_equation:
                label_parts.append(edge_equation)
            if r2 is not None:
                label_parts.append(f"R²={r2:.2f}")
            if label_parts:
                edge.set_label("\n".join(label_parts))
                edge.set_fontsize("10")
            G.add_edge(edge)

    if output_path is not None:
        G.write_png(output_path)
        print(f"SCM graph saved to {output_path}")

    return G

