from typing import Dict, List, Optional

from monitor.networks import describe_network
from neat.genome import Genome


def genome_to_network(
    genome: Genome,
    activation_map: Dict,
    input_labels: Optional[List[str]] = None,
    output_labels: Optional[List[str]] = None,
) -> Dict:
    """Describe a genome's condensed graph as plain data, for the monitor to draw."""
    size = genome.condensed_size.item()
    return describe_network(
        from_nodes=genome.graph.from_nodes[:size].tolist(),
        to_nodes=genome.graph.to_nodes[:size].tolist(),
        weights=genome.graph.weights[:size].tolist(),
        activation_ids=genome.node_activation_ids.tolist(),
        biases=genome.node_biases.tolist(),
        input_size=genome.input_size,
        output_size=genome.output_size,
        activation_names=activation_map,
        input_labels=input_labels,
        output_labels=output_labels,
    )
