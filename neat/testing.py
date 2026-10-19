"""Periodic tests of a population, run every few generations to see how good its
best members really are: training fitness on a noisy environment is mostly luck at
the top.

NEAT takes a population test, called with keyword arguments (rng, genome,
activation_selector), where genome is the generation's evaluated population (with
its training fitness), and returning a TestResult. The test names the metrics it
logs. make_top_k_test and make_field_test build the common ones."""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from neat.fitness import describe_stats
from neat.genome import Genome

# called with keyword arguments (rng, genome, activation_selector) for a single
# genome; returns its score and any data to show for it (e.g. the episode)
TestFn = Callable[..., Tuple[jax.Array, Any]]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TestResult:
    """What a test of the population found: metrics to log, by the names the test
    gives them, and optionally an episode for the monitor to show, and the genome ids
    playing the episode, one per seat, for it to name."""

    metrics: Dict[str, jax.Array]
    episode: Any = None
    players: Optional[jax.Array] = None


# called with keyword arguments (rng, genome, activation_selector) for a population
PopulationTestFn = Callable[..., TestResult]


def fittest(genome: Genome, k: int) -> Genome:
    """The k genomes of a population with the highest training fitness, fittest
    first."""
    _, idx = jax.lax.top_k(genome.fitness, k)
    return jax.tree.map(lambda x: x[idx], genome)


def top_k_test(
    rng: jax.Array,
    genome: Genome,
    test_fn: TestFn,
    k: int,
    name: str,
    **kwargs,
) -> TestResult:
    """Tests the k fittest genomes on the same episodes, logging the best and the
    mean of their scores as max_<name> and mean_<name>, and shows the best one's
    episode."""
    tested = fittest(genome, k)
    scores, data = jax.vmap(lambda g: test_fn(rng=rng, genome=g, **kwargs))(tested)
    best = jnp.argmax(scores)
    return TestResult(
        metrics={f"max_{name}": scores.max(), f"mean_{name}": scores.mean()},
        episode=jax.tree.map(lambda x: x[best], data),
        players=tested.genome_id[best][None],
    )


def field_test(
    rng: jax.Array,
    genome: Genome,
    test_fn: Callable[..., Tuple[jax.Array, Any, Dict[str, jax.Array]]],
    size: int,
    name: Optional[str],
    ranks: Optional[Tuple[int, ...]] = None,
    **kwargs,
) -> TestResult:
    """Tests the size fittest genomes against each other, fittest in the first seat,
    or with ranks, the genomes at those ranks by fitness (1 the fittest), logging the best and the mean of their scores as max_<name> and mean_<name>
    (not with no name), and the mean, least and most of each of the test's stats
    (see neat.fitness.describe_stats), with the test's episode."""
    if ranks is None:
        field = fittest(genome, size)
    else:
        top = fittest(genome, max(ranks))
        field = jax.tree.map(lambda x: x[jnp.array(ranks) - 1], top)
    scores, episode, stats = test_fn(rng=rng, genome=field, **kwargs)
    metrics = describe_stats(stats)
    if name is not None:
        metrics.update({f"max_{name}": scores.max(), f"mean_{name}": scores.mean()})
    return TestResult(
        metrics=metrics,
        episode=episode,
        players=field.genome_id,
    )


def make_top_k_test(
    test_fn: TestFn, k: int = 10, name: str = "fitness_against_baseline"
) -> PopulationTestFn:
    """A population test from a test of one genome, e.g. against a baseline (see
    neat.fitness.make_test_fn): tests the k fittest (see top_k_test)."""
    assert k > 0, "k must be greater than 0"
    return partial(top_k_test, test_fn=test_fn, k=k, name=name)


def make_field_test(
    test_fn: Callable[..., Tuple[jax.Array, Any, Dict[str, jax.Array]]],
    size: int,
    name: Optional[str] = "field",
    ranks: Optional[Sequence[int]] = None,
) -> PopulationTestFn:
    """A population test from a game between a batch of genomes, returning each
    one's score, an episode and each one's stats (see
    neat.fitness.make_mp_field_fn): the size fittest play it, or the size genomes
    at ranks by fitness, 1 being the fittest, for a field of mixed strength (see
    field_test). With name None, it logs no scores, just the stats."""
    assert size > 0, "a field test needs at least one genome"
    if ranks is not None:
        ranks = tuple(ranks)
        assert len(ranks) == size, "a field test needs a rank for every genome"
        assert len(set(ranks)) == size and min(ranks) >= 1, "ranks are distinct, from 1"
    return partial(field_test, test_fn=test_fn, size=size, name=name, ranks=ranks)
