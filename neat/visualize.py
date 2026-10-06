from typing import Dict, List, Optional

from neat.genome import Genome


def genome_to_network(
    genome: Genome,
    activation_map: Dict,
    input_labels: Optional[List[str]] = None,
    output_labels: Optional[List[str]] = None,
) -> Dict:
    """Describe a genome's condensed graph as plain data, for the monitor to draw."""
    num_io = genome.input_size + genome.output_size
    input_labels = input_labels or [str(i) for i in range(genome.input_size)]
    output_labels = output_labels or [str(i) for i in range(genome.output_size)]
    size = genome.condensed_size.item()
    from_nodes = genome.graph.from_nodes[:size].tolist()
    to_nodes = genome.graph.to_nodes[:size].tolist()
    weights = genome.graph.weights[:size].tolist()
    activation_ids = genome.node_activation_ids.tolist()
    biases = genome.node_biases.tolist()

    node_ids = sorted(set(range(num_io)) | set(from_nodes) | set(to_nodes))
    nodes = []
    for i in node_ids:
        if i < genome.input_size:
            kind, label = "input", input_labels[i]
        elif i < num_io:
            kind, label = "output", output_labels[i - genome.input_size]
        else:
            kind, label = "hidden", None
        nodes.append(
            {
                "id": i,
                "kind": kind,
                "label": label,
                "activation": activation_map[activation_ids[i]],
                "bias": biases[i],
            }
        )
    edges = [
        {"from": f, "to": t, "weight": w}
        for f, t, w in zip(from_nodes, to_nodes, weights)
    ]
    return {"type": "network", "nodes": nodes, "edges": edges}
