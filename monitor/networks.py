"""Networks as plain data the dashboard can draw ({"type": "network", ...}), from the
arrays of a genome's condensed graph.

A run logs every member's network each generation (monitor.client.Networks), packed
into one .npz: per genome, its connections and its nodes, concatenated.

    ids              (n,)  genome ids
    num_connections  (n,)  how many of the connections below are each genome's
    from_nodes, to_nodes, weights        the connections, genome after genome
    num_nodes        (n,)  how many of the nodes below are each genome's
    activation_ids, biases               the nodes, genome after genome
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def describe_network(
    from_nodes: Sequence[int],
    to_nodes: Sequence[int],
    weights: Sequence[float],
    activation_ids: Sequence[int],
    biases: Sequence[float],
    input_size: int,
    output_size: int,
    activation_names: Dict[int, str],
    input_labels: Optional[List[str]] = None,
    output_labels: Optional[List[str]] = None,
) -> Dict:
    """Describe a condensed graph (connections in topological order) for the monitor
    to draw. Nodes are indexed by id: inputs, then outputs, then hidden nodes."""
    num_io = input_size + output_size
    input_labels = input_labels or [str(i) for i in range(input_size)]
    output_labels = output_labels or [str(i) for i in range(output_size)]
    from_nodes = [int(i) for i in from_nodes]
    to_nodes = [int(i) for i in to_nodes]

    node_ids = sorted(set(range(num_io)) | set(from_nodes) | set(to_nodes))
    nodes = []
    for i in node_ids:
        if i < input_size:
            kind, label = "input", input_labels[i]
        elif i < num_io:
            kind, label = "output", output_labels[i - input_size]
        else:
            kind, label = "hidden", None
        nodes.append(
            {
                "id": i,
                "kind": kind,
                "label": label,
                "activation": activation_names.get(int(activation_ids[i]), "?"),
                "bias": float(biases[i]),
            }
        )
    edges = [
        {"from": f, "to": t, "weight": float(w)}
        for f, t, w in zip(from_nodes, to_nodes, weights)
    ]
    return {"type": "network", "nodes": nodes, "edges": edges}


def unpack_network(npz: Mapping[str, Any], genome_id: int) -> Optional[Dict]:
    """One genome's arrays out of a generation's packed networks, or None if it
    isn't there."""
    where = np.flatnonzero(npz["ids"] == genome_id)
    if not where.size:
        return None
    i = int(where[0])
    conn_end = np.cumsum(npz["num_connections"])
    node_end = np.cumsum(npz["num_nodes"])
    conns = slice(int(conn_end[i] - npz["num_connections"][i]), int(conn_end[i]))
    nodes = slice(int(node_end[i] - npz["num_nodes"][i]), int(node_end[i]))
    return {
        "from_nodes": npz["from_nodes"][conns],
        "to_nodes": npz["to_nodes"][conns],
        "weights": npz["weights"][conns],
        "activation_ids": npz["activation_ids"][nodes],
        "biases": npz["biases"][nodes],
    }
