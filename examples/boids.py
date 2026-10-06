"""Evolve a flocking policy for EvoJAX's boids task.

    uv run examples/boids.py --generations 100
    uv run examples/boids.py --monitor

Start the monitor first with `uv run turbo-neat-monitor`.
"""

import argparse

from evojax.task.flocking import FlockingTask

import neat.activations as act
from monitor import DEFAULT_URL, Episode, Monitor
from neat.config import GenomeConfig, MutationConfig, NEATConfig, SelectionConfig
from neat.fitness import make_fitness_fn
from neat.neat import NEAT
from neat.species import make_improvement_stagnation_fn

NUM_STEPS = 100


def get_input_label(idx: int) -> str:
    neighbor_num = idx // 3
    if idx % 3 == 0:
        return f"Neighbor {neighbor_num}: X"
    elif idx % 3 == 1:
        return f"Neighbor {neighbor_num}: Y"
    else:
        return f"Neighbor {neighbor_num}: Theta"


def get_output_label(idx: int) -> str:
    if idx == 0:
        return "d_theta"
    else:
        return "d_speed"


def make_config(input_size: int, output_size: int) -> NEATConfig:
    mutation_config = MutationConfig(
        add_connection_prob=0.05,
        add_node_prob=0.03,
        disable_connection_prob=0.05,
        disable_node_prob=0.03,
        mutate_weight_std=0.1,
        mutate_weight_prob=0.8,
        mutate_activation_prob=0.0,
        mutate_bias_prob=0.0,
        mutate_bias_std=0.0,
    )

    genome_config = GenomeConfig(
        input_size=input_size,
        output_size=output_size,
        initial_capacity=100,
        init_weight_mean=0.0,
        init_weight_std=1.0,
        activation_fns=[act.iden, act.sigmoid, act.tanh, act.relu],
        # identity on every input
        input_activation_ids=[0 for _ in range(input_size)],
        # tanh on every output
        output_activation_ids=[2 for _ in range(output_size)],
        input_labels=[get_input_label(i) for i in range(input_size)],
        output_labels=[get_output_label(i) for i in range(output_size)],
        capacity_growth_strategy="linear",
        init_mode="full",
    )

    selection_config = SelectionConfig(
        population_size=100,
        cutoff_pct=0.1,
        speciation_threshold=1.0,
        compatibility_coefficients=(1.0, 0.9),
        elitism=0.1,
        stagnation_fn=make_improvement_stagnation_fn(max_stagnated_steps=20),
        max_transfer_age=1,
        maximum_species=5,
        selection_tournament_size=8,
        min_species_size=0,
        species_warmup_threshold=1,
    )

    return NEATConfig(
        mutation_config=mutation_config,
        genome_config=genome_config,
        selection_config=selection_config,
    )


def episode_fn(state):
    """One test episode as raw boid states (x, y, theta), for the monitor to draw."""
    return Episode("boids", boids=state.state)


def main():
    parser = argparse.ArgumentParser(
        description="Evolve a flocking policy for EvoJAX's boids task."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument(
        "--monitor",
        nargs="?",
        const=DEFAULT_URL,
        default=None,
        metavar="URL",
        help=f"log to a turbo-neat monitor (default {DEFAULT_URL})",
    )
    args = parser.parse_args()

    train_task = FlockingTask(NUM_STEPS, action_type=1)
    test_task = FlockingTask(NUM_STEPS, action_type=1)

    neat = NEAT(
        config=make_config(train_task.obs_shape[0], train_task.act_shape[0]),
        fitness_fn=make_fitness_fn(task=train_task, num_steps=NUM_STEPS),
        baseline_test_fn=make_fitness_fn(task=test_task, num_steps=NUM_STEPS),
        monitor=Monitor(args.monitor, project="boids") if args.monitor else None,
    )
    neat.run(seed=args.seed, num_generations=args.generations, episode_fn=episode_fn)


if __name__ == "__main__":
    main()
