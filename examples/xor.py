"""Backprop NEAT on XOR: classify points by whether x and y have the same sign.

Each generation, every genome's weights and biases take a few gradient steps on the
loss and are scored on fresh data; evolution then searches over topologies.

    uv run examples/xor.py --generations 100
    uv run examples/xor.py --monitor

Start the monitor first with `uv run turbo-neat-monitor`.
"""

import argparse
from dataclasses import replace
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

import neat.activations as act
from monitor import DEFAULT_URL, Episode, Monitor
from neat.backprop_neat import BackpropNEAT
from neat.config import GenomeConfig, MutationConfig, NEATConfig, SelectionConfig
from neat.fitness import Recording
from neat.genome import Genome, forward
from neat.species import make_remove_last_if_stagnant_and_full_stagnation_fn
from neat.testing import make_top_k_test

NUM_POINTS = 1000


def generate_xor_data(
    rng: jax.Array, num_points: int, noise: float = 0.5, r: float = 5
) -> Tuple[jax.Array, jax.Array]:
    data = jax.random.uniform(rng, (num_points, 2), minval=-r, maxval=r)
    data += jax.random.normal(rng, (num_points, 2)) * noise
    labels = jnp.logical_xor(data[:, 0] > 0, data[:, 1] > 0).astype(jnp.float32)
    return data, labels


def sigmoid_binary_cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return -labels * log_p - (1 - labels) * log_not_p


def make_backprop_fn(record: bool = False) -> Callable:
    """Fitness is the negative loss on a fresh batch; also returns the gradients of
    the loss with respect to each genome's weights and biases, and, if record, a
    Recording of each genome's batch with its predicted probabilities, shaped like
    the test's (genomes, points, 3) with one genome per episode."""

    def backprop_fn(rng, genome: Genome, **kwargs):
        rng_batch = jax.random.split(rng, genome.batch_size)
        data, labels = jax.vmap(partial(generate_xor_data, num_points=NUM_POINTS))(
            rng_batch
        )

        def loss_fn(weights, biases, gen: Genome, data, labels):
            gen = replace(
                gen, graph=replace(gen.graph, weights=weights), node_biases=biases
            )
            logits = forward(gen, data, **kwargs, diff_mode=True)
            loss = sigmoid_binary_cross_entropy(logits.squeeze(-1), labels).mean()
            return loss, logits

        value_and_grad = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
        (loss, logits), grads = jax.vmap(value_and_grad)(
            genome.graph.weights, genome.node_biases, genome, data, labels
        )
        if not record:
            return -loss, grads
        frames = jnp.concatenate([data, jax.nn.sigmoid(logits)], axis=-1)[:, None]
        players = jnp.arange(genome.batch_size)[:, None]
        return -loss, grads, Recording(frames, players)

    return backprop_fn


def make_test_fn() -> Callable:
    """Scores one genome on fresh data, and returns the points with their predicted
    class probabilities, shape (points, 3), for plotting."""

    def test_fn(rng, genome: Genome, **kwargs):
        data, labels = generate_xor_data(rng, num_points=NUM_POINTS)
        logits = forward(genome, data, **kwargs, diff_mode=False)
        loss = sigmoid_binary_cross_entropy(logits.squeeze(-1), labels).mean()
        return -loss, jnp.concatenate([data, jax.nn.sigmoid(logits)], axis=-1)

    return test_fn


def make_config() -> NEATConfig:
    mutation_config = MutationConfig(
        add_connection_prob=0.1,
        add_node_prob=0.06,
        disable_connection_prob=0.05,
        disable_node_prob=0.03,
        mutate_weight_std=0.0,
        mutate_weight_prob=0.0,
        mutate_activation_prob=0.0,
        mutate_bias_prob=0.0,
        mutate_bias_std=0.0,
        learning_rate=0.1,
        backprop_steps=5,
        max_grad_norm=1.0,
    )

    genome_config = GenomeConfig(
        input_size=2,
        output_size=1,
        initial_capacity=100,
        init_weight_mean=0.0,
        init_weight_std=1.0,
        activation_fns=[
            act.iden,
            act.sigmoid,
            act.tanh,
            act.relu,
            act.sin,
            act.cos,
            act.absl,
        ],
        input_activation_ids=[0, 0],
        # the output is a logit
        output_activation_ids=[0],
        input_labels=["x", "y"],
        output_labels=["class"],
        capacity_growth_strategy="linear",
        init_mode="full",
    )

    selection_config = SelectionConfig(
        population_size=100,
        cutoff_pct=0.1,
        speciation_threshold=2.0,
        compatibility_coefficients=(1.0, 0.5),
        elitism=0.1,
        selection_tournament_size=3,
        fitness_ema_period=5,
        maximum_species=5,
        stagnation_fn=make_remove_last_if_stagnant_and_full_stagnation_fn(
            max_stagnated_steps=20, full_target=5
        ),
    )

    return NEATConfig(
        genome_config=genome_config,
        mutation_config=mutation_config,
        selection_config=selection_config,
    )


def episode_fn(data):
    """Points with a genome's predicted probabilities, for the monitor to plot: the
    test's (points, 3), or the members' recorded batch, (members, 1, points, 3),
    which makes one plot each."""
    if data.ndim == 4:
        data = data[:, 0]
    return Episode("classification_2d", points=data[..., :2], probability=data[..., 2])


def main():
    parser = argparse.ArgumentParser(description="Backprop NEAT on XOR.")
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

    bneat = BackpropNEAT(
        make_config(),
        # record each member's scored batch for the monitor's member view
        backprop_fn=make_backprop_fn(record=args.monitor is not None),
        # every 10th generation's 10 fittest, on the same fresh data
        test_fn=make_top_k_test(make_test_fn()),
        monitor=Monitor(args.monitor, project="xor") if args.monitor else None,
        # each batch is a random draw of the data, so one would be a noisy fitness
        eval_batches=2,
    )
    bneat.run(seed=args.seed, num_generations=args.generations, episode_fn=episode_fn)


if __name__ == "__main__":
    main()
