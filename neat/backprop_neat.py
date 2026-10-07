from dataclasses import replace
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from monitor import Monitor
from neat.activations import ActivationSelector
from neat.config import NEATConfig
from neat.fitness import Recording
from neat.genome import Genome, prepare_for_inference
from neat.neat import NEAT, TestFn, evolve_one_generation, test_against_baseline
from neat.population import Population

# called like a FitnessFn on a fresh batch of data; returns fitness (-loss) and
# (weight grads, bias grads) of the loss for each genome, then, if it was made to
# record, a Recording of what each genome was scored on
BackpropFn = Callable[..., Tuple[Any, ...]]

# fitness given to a genome whose loss is no longer finite
DIVERGED_FITNESS = -1e6


def gradient_step(
    genome: Genome,
    weight_grads: jax.Array,
    bias_grads: jax.Array,
    learning_rate: float,
    max_grad_norm: Optional[float],
) -> Genome:
    """Take one gradient descent step on every genome in a batch."""
    weight_grads = jnp.nan_to_num(weight_grads)
    bias_grads = jnp.nan_to_num(bias_grads)
    if max_grad_norm is not None:
        # clip the gradient norm of each genome separately
        norm = jnp.sqrt(
            jnp.sum(weight_grads**2, axis=-1) + jnp.sum(bias_grads**2, axis=-1)
        )
        scale = jnp.minimum(1.0, max_grad_norm / (norm + 1e-8))[..., None]
        weight_grads = weight_grads * scale
        bias_grads = bias_grads * scale
    return replace(
        genome,
        graph=replace(
            genome.graph, weights=genome.graph.weights - learning_rate * weight_grads
        ),
        node_biases=genome.node_biases - learning_rate * bias_grads,
    )


def evaluate_and_train(
    rng: jax.Array,
    population: Population,
    backprop_fn: BackpropFn,
    learning_rate: float,
    num_steps: int,
    max_grad_norm: Optional[float],
    activation_selector: ActivationSelector,
) -> Tuple[Population, Optional[Recording]]:
    """Train the weights and biases of every genome by gradient descent, then score
    the trained genomes on a fresh batch.

    The trained weights are kept (and inherited), so the population's weights improve
    across generations as well as within them. Also returns what each genome was
    scored on, if backprop_fn records it."""
    population = replace(
        population, batched_genome=prepare_for_inference(population.batched_genome)
    )
    genome = population.batched_genome
    train_rng, eval_rng = jax.random.split(rng)

    def train_step(genome: Genome, rng: jax.Array) -> Tuple[Genome, None]:
        _, (weight_grads, bias_grads), *_ = backprop_fn(
            rng=rng, genome=genome, activation_selector=activation_selector
        )
        return gradient_step(
            genome, weight_grads, bias_grads, learning_rate, max_grad_norm
        ), None

    genome, _ = jax.lax.scan(train_step, genome, jax.random.split(train_rng, num_steps))
    fitnesses, _, *recorded = backprop_fn(
        rng=eval_rng, genome=genome, activation_selector=activation_selector
    )
    fitnesses = jnp.where(jnp.isfinite(fitnesses), fitnesses, DIVERGED_FITNESS)
    recording = recorded[0] if recorded else None
    return replace(
        population, batched_genome=replace(genome, fitness=fitnesses)
    ), recording


class BackpropNEAT(NEAT):
    """NEAT algorithm with backpropagation for network parameters"""

    def __init__(
        self,
        config: NEATConfig,
        backprop_fn: BackpropFn,
        test_fn: Optional[TestFn] = None,
        monitor: Optional[Monitor] = None,
    ):
        self.config = config
        self.activation_selector = config.genome_config.activation_selector
        self.monitor = monitor

        if config.mutation_config.mutate_weight_prob > 0:
            print(
                "Mutate weight probability is greater than 0. This is not supported by backprop NEAT. Setting mutate_weight_prob to 0."
            )
            self.config.mutation_config.mutate_weight_prob = 0.0
        if config.mutation_config.mutate_bias_prob > 0:
            print(
                "Mutate bias probability is greater than 0. This is not supported by backprop NEAT. Setting mutate_bias_prob to 0."
            )
            self.config.mutation_config.mutate_bias_prob = 0.0
        if config.mutation_config.learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0 for backprop NEAT")
        if config.mutation_config.backprop_steps <= 0:
            raise ValueError("backprop_steps must be greater than 0 for backprop NEAT")

        self.evaluate_population = jax.jit(
            partial(
                evaluate_and_train,
                backprop_fn=backprop_fn,
                learning_rate=config.mutation_config.learning_rate,
                num_steps=config.mutation_config.backprop_steps,
                max_grad_norm=config.mutation_config.max_grad_norm,
                activation_selector=self.activation_selector,
            )
        )
        self.evolve = jax.jit(partial(evolve_one_generation, config=config))
        if test_fn is not None:
            self.test_baseline = jax.jit(
                partial(
                    test_against_baseline,
                    test_fn=test_fn,
                    activation_selector=self.activation_selector,
                )
            )
        else:
            self.test_baseline = None

        self.test_champion = None
