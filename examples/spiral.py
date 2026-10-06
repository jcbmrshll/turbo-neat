"""Backprop NEAT on a 2D binary classification problem (two spirals, or XOR).

Each generation, every genome's weights and biases take a few gradient steps on the
loss and are scored on fresh data; evolution then searches over topologies.

    uv run examples/spiral.py --generations 300
    uv run examples/spiral.py --dataset xor
    uv run examples/spiral.py --wandb-project backprop-neat

Rendering the network after each improvement needs the graphviz `dot` binary.
"""

import argparse
from dataclasses import replace
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb

import neat_jax.activations as act
from neat_jax.backprop_neat import BackpropNEAT
from neat_jax.config import GenomeConfig, MutationConfig, NEATConfig, SelectionConfig
from neat_jax.genome import Genome, forward
from neat_jax.species import make_remove_last_if_stagnant_and_full_stagnation_fn
from neat_jax.utils import apply

NUM_POINTS = 1000

# maps (rng) to (points of shape (n, 2), labels of shape (n,))
DataFn = Callable[[jax.Array], Tuple[jax.Array, jax.Array]]


def generate_xor_data(
    rng: jax.Array, num_points: int, noise: float = 0.5, r: float = 5
) -> Tuple[jax.Array, jax.Array]:
    data = jax.random.uniform(rng, (num_points, 2), minval=-r, maxval=r)
    data += jax.random.normal(rng, (num_points, 2)) * noise
    labels = jnp.logical_xor(data[:, 0] > 0, data[:, 1] > 0).astype(jnp.float32)
    return data, labels


def generate_spiral_data(
    rng: jax.Array, num_points: int, noise: float = 0.5
) -> Tuple[jax.Array, jax.Array]:
    n = num_points // 2

    def gen_spiral(delta_t, label, rng):
        rng_p, rng_x, rng_y = jax.random.split(rng, 3)
        pts = jax.random.uniform(rng_p, (n,), minval=0, maxval=n)
        r = pts / n * 6.0
        t = 1.75 * pts / n * 2 * jnp.pi + delta_t
        x = (
            r * jnp.sin(t)
            + jax.random.uniform(rng_x, (n,), minval=-1, maxval=1) * noise
        )
        y = (
            r * jnp.cos(t)
            + jax.random.uniform(rng_y, (n,), minval=-1, maxval=1) * noise
        )
        return jnp.stack([x, y], axis=-1), jnp.full((n,), label)

    rng1, rng2 = jax.random.split(rng)
    pos_data, pos_labels = gen_spiral(0, 0, rng1)
    neg_data, neg_labels = gen_spiral(jnp.pi, 1, rng2)
    data = jnp.concatenate([pos_data, neg_data], axis=0)
    labels = jnp.concatenate([pos_labels, neg_labels], axis=0)
    return data, labels


DATASETS = {"spiral": generate_spiral_data, "xor": generate_xor_data}


def sigmoid_binary_cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return -labels * log_p - (1 - labels) * log_not_p


def make_backprop_fn(generate: Callable) -> Callable:
    """Fitness is the negative loss on a fresh batch; also returns the gradients of
    the loss with respect to each genome's weights and biases."""

    def backprop_fn(rng, genome: Genome, **kwargs):
        rng_batch = jax.random.split(rng, genome.batch_size)
        data, labels = jax.vmap(partial(generate, num_points=NUM_POINTS))(rng_batch)

        def loss_fn(weights, biases, gen: Genome, data, labels):
            gen = replace(
                gen, graph=replace(gen.graph, weights=weights), node_biases=biases
            )
            logits = forward(gen, data, **kwargs, diff_mode=True)
            return sigmoid_binary_cross_entropy(logits.squeeze(-1), labels).mean()

        loss, grads = jax.vmap(jax.value_and_grad(loss_fn, argnums=(0, 1)))(
            genome.graph.weights, genome.node_biases, genome, data, labels
        )
        return -loss, grads

    return backprop_fn


def make_test_fn(generate: Callable) -> Callable:
    """Scores the genomes on fresh data, and returns the points with their predicted
    class probabilities, shape (batch, points, 3), for plotting."""

    def test_fn(rng, genome: Genome, **kwargs):
        rng_batch = jax.random.split(rng, genome.batch_size)
        num_points = max(NUM_POINTS // genome.batch_size, 1)
        data, labels = jax.vmap(partial(generate, num_points=num_points))(rng_batch)
        logits = apply(genome, forward, data, **kwargs, diff_mode=False)
        loss = sigmoid_binary_cross_entropy(logits.squeeze(-1), labels).mean()
        return -loss, jnp.concatenate([data, jax.nn.sigmoid(logits)], axis=2)

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


def render_fn(data):
    """Scatter the test points, coloured by the class the best genome predicts."""
    data = data.reshape(-1, 3)
    predicted = (data[:, 2] > 0.5).tolist()
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=["y" if p else "b" for p in predicted])
    image = wandb.Image(fig)
    plt.close(fig)
    return image


def main():
    parser = argparse.ArgumentParser(
        description="Backprop NEAT on a 2D binary classification problem."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="spiral")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="log metrics and prediction plots to wandb",
    )
    args = parser.parse_args()

    generate = DATASETS[args.dataset]
    bneat = BackpropNEAT(
        make_config(),
        backprop_fn=make_backprop_fn(generate),
        test_fn=make_test_fn(generate),
        wandb_project=args.wandb_project,
    )
    bneat.run(seed=args.seed, num_generations=args.generations, render_fn=render_fn)


if __name__ == "__main__":
    main()
