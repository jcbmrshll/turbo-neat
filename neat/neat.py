import concurrent.futures
from dataclasses import replace
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from monitor import Members, Monitor, Networks
from neat.activations import ActivationSelector
from neat.config import GenomeConfig, NEATConfig
from neat.fitness import Recording
from neat.genome import (
    Genome,
    extend_capacity,
    get_hidden_node_counts,
    init_genome,
    is_almost_full,
    prepare_for_crossover,
    prepare_for_inference,
)
from neat.logging import log_generation
from neat.population import (
    Population,
    _init_population,
    create_next_generation,
)
from neat.species import SpeciesPCA, fill_prev_stats, get_species_stats, species_pca
from neat.testing import PopulationTestFn, TestResult
from neat.utils import apply, is_printable
from neat.visualize import genome_to_network

# called with keyword arguments (rng, genome, activation_selector)
# returns fitness and any auxiliary data (e.g. the episode, for the monitor); a
# fitness function made with record=True returns a Recording of every episode
FitnessFn = Callable[..., Tuple[jax.Array, Any]]


def init_population(
    rng: jax.Array,
    config: NEATConfig,
    generation: int = 0,
    initial_innovation_id: int = 0,
) -> Population:
    """Initialize a population of genomes"""
    batched_genome = init_genome(
        rng,
        batch_size=config.selection_config.population_size,
        capacity=config.genome_config.initial_capacity,
        input_size=config.genome_config.input_size,
        output_size=config.genome_config.output_size,
        input_activation_ids=config.genome_config.input_activation_ids,
        output_activation_ids=config.genome_config.output_activation_ids,
        weight_mean=config.genome_config.init_weight_mean,
        weight_std=config.genome_config.init_weight_std,
        mode=config.genome_config.init_mode,
        initial_innovation_id=initial_innovation_id,
    )

    next_innovation_id = batched_genome.graph.innovation_ids.max() + 1
    next_genome_id = batched_genome.genome_id.max() + 1

    return _init_population(
        batched_genome,
        config.mutation_config,
        config.selection_config,
        next_innovation_id,
        next_genome_id,
        generation,
    )


def resize_genome(population: Population, genome_config: GenomeConfig) -> Population:
    """
    Expand the capacity of the genome. This will cause jitted functions to be recompiled.
    So it should be called sparingly!
    """
    strategy = genome_config.capacity_growth_strategy
    if strategy == "linear":
        new_capacity = (
            population.batched_genome.capacity + genome_config.initial_capacity
        )
    elif strategy == "exponential":
        new_capacity = population.batched_genome.capacity * 2
    elif strategy == "constant":
        return population
    else:
        raise ValueError(f"unknown capacity_growth_strategy: {strategy}")

    amount = new_capacity - population.batched_genome.capacity
    extended_genome = apply(population.batched_genome, extend_capacity, amount=amount)
    extended_prev_genome = apply(
        population.prev_batched_genome, extend_capacity, amount=amount
    )
    return replace(
        population,
        batched_genome=extended_genome,
        prev_batched_genome=extended_prev_genome,
    )


def evaluate(
    rng: jax.Array,
    population: Population,
    fitness_fn: FitnessFn,
    activation_selector: ActivationSelector,
) -> Tuple[Population, Optional[Recording]]:
    """Evaluate the fitness of every genome; also returns the episodes they played,
    if the fitness function recorded them"""
    population = replace(
        population, batched_genome=prepare_for_inference(population.batched_genome)
    )
    genome = population.batched_genome
    # Evaluate fitness of population
    fitness_rng, rng = jax.random.split(rng)
    fitnesses, aux = fitness_fn(
        rng=fitness_rng, genome=genome, activation_selector=activation_selector
    )
    # store raw fitnesses in genome
    genome = replace(genome, fitness=fitnesses)
    recording = aux if isinstance(aux, Recording) else None
    return replace(population, batched_genome=genome), recording


def evolve_one_generation(
    rng: jax.Array, population: Population, config: NEATConfig
) -> Population:
    """Evolve the population by one generation"""
    # Prepare genomes for crossover
    population = replace(
        population, batched_genome=prepare_for_crossover(population.batched_genome)
    )
    # create next generation
    selection_rng, rng = jax.random.split(rng)
    population = create_next_generation(
        population,
        selection_rng,
        selection_config=config.selection_config,
        mutation_config=config.mutation_config,
        genome_config=config.genome_config,
    )
    return replace(population, generation=population.generation + 1)


def test_population(
    rng: jax.Array,
    population: Population,
    test_fn: PopulationTestFn,
    activation_selector: ActivationSelector,
) -> TestResult:
    """Test the evaluated population (see neat.testing)"""
    return test_fn(
        rng=rng,
        genome=population.batched_genome,
        activation_selector=activation_selector,
    )


class NEAT:
    """NEAT algorithm"""

    # jitted steps, bound to the config and fitness functions in __init__
    evaluate_population: Callable[
        [jax.Array, Population], Tuple[Population, Optional[Recording]]
    ]
    evolve: Callable[[jax.Array, Population], Population]
    test: Optional[Callable[[jax.Array, Population], TestResult]]
    map_species: Callable[[Genome], SpeciesPCA]

    def __init__(
        self,
        config: NEATConfig,
        fitness_fn: FitnessFn,
        test_fn: Optional[PopulationTestFn] = None,
        test_every: int = 10,
        monitor: Optional[Monitor] = None,
    ):
        """test_fn, if given, tests the population of every test_every-th generation
        (from the first), e.g. its fittest against a baseline: its metrics are
        logged, and its episode and network go to the monitor (see neat.testing)."""
        self.config = config
        self.activation_selector = config.genome_config.activation_selector
        self.monitor = monitor
        self.evaluate_population = jax.jit(
            partial(
                evaluate,
                fitness_fn=fitness_fn,
                activation_selector=self.activation_selector,
            )
        )
        self.evolve = jax.jit(partial(evolve_one_generation, config=config))
        self.map_species = jax.jit(
            partial(species_pca, num_species=config.selection_config.maximum_species)
        )
        self._init_test(test_fn, test_every)

    def _init_test(self, test_fn: Optional[PopulationTestFn], test_every: int) -> None:
        assert test_every > 0, "test_every must be greater than 0"
        self.test_every = test_every
        if test_fn is not None:
            self.test = jax.jit(
                partial(
                    test_population,
                    test_fn=test_fn,
                    activation_selector=self.activation_selector,
                )
            )
        else:
            self.test = None

    def run(
        self,
        seed: int,
        num_generations: int,
        population: Optional[Population] = None,
        episode_fn: Optional[Callable] = None,
        log_async: bool = True,
    ):
        """Run the NEAT algorithm"""
        if self.monitor is not None:
            self.monitor.start(
                config={
                    "run": {"seed": seed, "num_generations": num_generations},
                    **self.config.to_dict(),
                }
            )
        try:
            return self._run(seed, num_generations, population, episode_fn, log_async)
        except BaseException as e:
            if self.monitor is not None:
                interrupted = isinstance(e, KeyboardInterrupt)
                self.monitor.finish("stopped" if interrupted else "crashed")
            raise

    def _run(
        self,
        seed: int,
        num_generations: int,
        population: Optional[Population],
        episode_fn: Optional[Callable],
        log_async: bool,
    ):
        rng = jax.random.PRNGKey(seed)
        if population is None:
            rng_init, rng = jax.random.split(rng, 2)
            population = init_population(rng_init, self.config)

        prev_stats = dict()
        species_stats = get_species_stats(
            population.species_data, prev_stats=prev_stats
        )

        # this just functions as a logging job queue
        logger = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        for g in range(num_generations):
            results = {}
            rng, gen_rng, eval_rng, test_rng = jax.random.split(rng, 4)
            # evaluate population
            population, recording = self.evaluate_population(eval_rng, population)
            # adjust mutation noise
            num_unique_species = len(
                jnp.unique(population.batched_genome.species_id).tolist()
            )
            # store results
            results.update(
                {
                    "generation": population.generation,
                    "mean_fitness": population.batched_genome.fitness.mean(),
                    "max_fitness": population.batched_genome.fitness.max(),
                    "min_fitness": population.batched_genome.fitness.min(),
                    "mean_hidden_nodes": get_hidden_node_counts(
                        population.batched_genome, use_condensed=False
                    ).mean(),
                    "mean_condensed_hidden_nodes": get_hidden_node_counts(
                        population.batched_genome, use_condensed=True
                    ).mean(),
                    "mean_connections": population.batched_genome.num_enabled_connections.mean(),
                    "mean_condensed_connections": population.batched_genome.condensed_size.mean(),
                    "num_species": num_unique_species,
                }
            )
            new_species_stats = get_species_stats(
                population.species_data, prev_stats=prev_stats
            )
            species_stats, prev_stats = new_species_stats, species_stats
            prev_stats = fill_prev_stats(prev_stats=prev_stats, cur_stats=species_stats)

            # test the population every test_every generations, keeping its
            # episode, and the genome it picked, for the monitor
            test, shown = None, None
            if self.test is not None and g % self.test_every == 0:
                test = self.test(test_rng, population)
                results.update(test.metrics)
                if (idx := test.shown) is not None:
                    shown = jax.tree.map(lambda x: x[idx], population.batched_genome)

            # evolving re-sorts each graph by innovation, so keep the evaluated
            # population (graphs condensed, in topological order) for its networks
            evaluated = population.batched_genome
            # evolve population
            population = self.evolve(gen_rng, population)

            if self.monitor is not None:
                # evolving speciated this generation and kept it as prev_batched_genome,
                # so log its members from there, in the species they were assigned to
                members = population.prev_batched_genome
                results["members"] = Members(
                    ids=members.genome_id,
                    parents=members.parent_ids,
                    species=members.species_id,
                    fitness=members.fitness,
                )
                # after the members, so the monitor knows the genome to name it
                if shown is not None:
                    results["network"] = genome_to_network(
                        shown,
                        self.config.genome_config.activation_map,
                        input_labels=self.config.genome_config.input_labels,
                        output_labels=self.config.genome_config.output_labels,
                    )
                    results["network"]["players"] = [int(shown.genome_id)]
                results["networks"] = Networks(
                    ids=evaluated.genome_id,
                    from_nodes=evaluated.graph.from_nodes,
                    to_nodes=evaluated.graph.to_nodes,
                    weights=evaluated.graph.weights,
                    num_connections=evaluated.condensed_size,
                    activation_ids=evaluated.node_activation_ids,
                    biases=evaluated.node_biases,
                    num_nodes=evaluated.next_node_idx,
                )
                # logging is i/o bound (and copies the episode, members and networks
                # off the device), so it goes to a background thread
                log_args = (
                    self.monitor,
                    dict(prev_stats),
                    dict(results),
                    g,
                    None if test is None else test.episode,
                    episode_fn,
                    recording,
                    evaluated.genome_id,
                    # on the device: only the map itself is copied off it
                    self.map_species(members),
                    None if test is None else test.players,
                )
                with jax.default_device(jax.devices("cpu")[0]):
                    if log_async:
                        logger.submit(log_generation, *log_args)
                    else:
                        log_generation(*log_args)

            results.update(**species_stats)
            print({k: f"{v:.2f}" for k, v in results.items() if is_printable(v)})

            # check if genome needs more space allocated
            if is_almost_full(population.batched_genome).any():
                population = resize_genome(population, self.config.genome_config)
                print(f"Resized genome to {population.batched_genome.capacity} nodes")

        logger.shutdown(wait=True)
        if self.monitor is not None:
            self.monitor.finish()
        # return the population when done
        return population
