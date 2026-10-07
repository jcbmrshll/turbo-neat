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
from neat.utils import apply, is_printable
from neat.visualize import genome_to_network

# called with keyword arguments (rng, genome, activation_selector), or for head-to-head
# functions (rng, genome_1, genome_2, activation_selector)
# returns fitness and any auxiliary data (e.g. the episode, for the monitor); a
# fitness function made with record=True returns a Recording of every episode
FitnessFn = Callable[..., Tuple[jax.Array, Any]]
# called with keyword arguments (rng, genome, activation_selector) for a single
# genome; returns its score and any data to show for it (e.g. the episode)
TestFn = Callable[..., Tuple[jax.Array, Any]]


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
    extended_champion = extend_capacity(population.champion, amount=amount)
    return replace(
        population,
        batched_genome=extended_genome,
        prev_batched_genome=extended_prev_genome,
        champion=extended_champion,
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


def test_against_baseline(
    rng: jax.Array,
    population: Population,
    test_fn: TestFn,
    activation_selector: ActivationSelector,
) -> Tuple[jax.Array, Any]:
    """Test the most fit member of the population"""
    genomes: Genome = population.batched_genome
    most_fit = jax.tree.map(lambda x: x[jnp.argmax(genomes.fitness)], genomes)
    return test_fn(rng=rng, genome=most_fit, activation_selector=activation_selector)


def test_against_champion(
    rng: jax.Array,
    population: Population,
    h2h_fn: FitnessFn,
    activation_selector: ActivationSelector,
) -> Tuple[Population, jax.Array, jax.Array]:
    genome: Genome = population.batched_genome
    champion = population.champion
    challenger_idx = jnp.argmax(genome.fitness)
    challenger = jax.tree.map(lambda x: x[challenger_idx], genome)
    b_champion = jax.tree.map(
        lambda x, y: jnp.broadcast_to(x, y.shape), champion, genome
    )
    b_challenger = jax.tree.map(
        lambda x, y: jnp.broadcast_to(x, y.shape), challenger, genome
    )
    fitnesses, _ = h2h_fn(
        rng=rng,
        genome_1=b_champion,
        genome_2=b_challenger,
        activation_selector=activation_selector,
    )
    champion_fitness = fitnesses[0]
    challenger_fitness = fitnesses[1]
    new_champion = challenger_fitness > champion_fitness
    # a scalar mask broadcasts as is (mask_data would give scalar fields a batch axis)
    population = replace(
        population,
        champion=jax.tree.map(
            lambda x, y: jnp.where(new_champion, x, y), challenger, champion
        ),
    )
    return population, challenger_fitness, new_champion


def test_field(
    rng: jax.Array,
    population: Population,
    test_fn: FitnessFn,
    field_size: int,
    activation_selector: ActivationSelector,
) -> Tuple[jax.Array, Any]:
    """Test the field_size fittest genomes against each other: their scores, the
    test's episode, and their genome ids, fittest first"""
    genomes: Genome = population.batched_genome
    fittest = jnp.argsort(-genomes.fitness)[:field_size]
    field = jax.tree.map(lambda x: x[fittest], genomes)
    scores, data = test_fn(
        rng=rng, genome=field, activation_selector=activation_selector
    )
    return scores, data, field.genome_id


class NEAT:
    """NEAT algorithm"""

    # jitted steps, bound to the config and fitness functions in __init__
    evaluate_population: Callable[
        [jax.Array, Population], Tuple[Population, Optional[Recording]]
    ]
    evolve: Callable[[jax.Array, Population], Population]
    test_champion: Optional[
        Callable[[jax.Array, Population], Tuple[Population, jax.Array, jax.Array]]
    ]
    test_baseline: Optional[Callable[[jax.Array, Population], Tuple[jax.Array, Any]]]
    map_species: Callable[[Genome], SpeciesPCA]
    test_field: Optional[
        Callable[[jax.Array, Population], Tuple[jax.Array, Any, jax.Array]]
    ]

    def __init__(
        self,
        config: NEATConfig,
        fitness_fn: FitnessFn,
        h2h_test_fn: Optional[FitnessFn] = None,
        baseline_test_fn: Optional[TestFn] = None,
        monitor: Optional[Monitor] = None,
        field_test_fn: Optional[FitnessFn] = None,
        field_size: int = 0,
        field_every: int = 1,
    ):
        """field_test_fn, if given, tests the field_size fittest genomes of every
        field_every-th generation (from the first) against each other, returning
        each one's score and an episode, which goes to the monitor."""
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
        if h2h_test_fn is not None:
            self.test_champion = jax.jit(
                partial(
                    test_against_champion,
                    h2h_fn=h2h_test_fn,
                    activation_selector=self.activation_selector,
                )
            )
        else:
            self.test_champion = None
        self.evolve = jax.jit(partial(evolve_one_generation, config=config))
        self.map_species = jax.jit(
            partial(species_pca, num_species=config.selection_config.maximum_species)
        )
        if baseline_test_fn is not None:
            self.test_baseline = jax.jit(
                partial(
                    test_against_baseline,
                    test_fn=baseline_test_fn,
                    activation_selector=self.activation_selector,
                )
            )
        else:
            self.test_baseline = None
        self.field_every = field_every
        if field_test_fn is not None:
            assert field_size > 0, "a field test needs a field_size"
            assert field_every > 0, "field_every must be greater than 0"
            self.test_field = jax.jit(
                partial(
                    test_field,
                    test_fn=field_test_fn,
                    field_size=field_size,
                    activation_selector=self.activation_selector,
                )
            )
        else:
            self.test_field = None

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
        best_fitness = float("-inf")

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

            if self.test_champion:
                # test against champion
                test_rng, rng = jax.random.split(rng)
                population, challenger_fitness, improved = self.test_champion(
                    test_rng, population
                )

                results.update(
                    {
                        "challenger_fitness": challenger_fitness,
                        "improved": int(improved),
                    }
                )
            else:
                max_fitness = population.batched_genome.fitness.max().item()
                max_idx = jnp.argmax(population.batched_genome.fitness)
                improved = max_fitness > best_fitness
                best_fitness = max(max_fitness, best_fitness)
                if improved:
                    population = replace(
                        population,
                        champion=jax.tree.map(
                            lambda x: x[max_idx], population.batched_genome
                        ),
                    )

            # test against baseline, keeping the episode for the monitor
            episode_data, episode_players = None, None
            if improved:
                if self.monitor is not None:
                    results["network"] = genome_to_network(
                        population.champion,
                        self.config.genome_config.activation_map,
                        input_labels=self.config.genome_config.input_labels,
                        output_labels=self.config.genome_config.output_labels,
                    )

                if self.test_baseline:
                    test_rng, rng = jax.random.split(rng)
                    test_fitness, episode_data = self.test_baseline(
                        test_rng, population
                    )
                    results["fitness_against_baseline"] = test_fitness

            # the fittest playing each other, every field_every generations
            if self.test_field and g % self.field_every == 0:
                test_rng, rng = jax.random.split(rng)
                field_scores, field_episode, field_ids = self.test_field(
                    test_rng, population
                )
                results["field_mean"] = field_scores.mean()
                results["field_best"] = field_scores.max()
                # the baseline's episode, if there is one, takes the monitor's slot
                if episode_data is None:
                    episode_data, episode_players = field_episode, field_ids

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
                    champion=population.champion.genome_id,
                )
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
                    episode_data,
                    episode_fn,
                    recording,
                    evaluated.genome_id,
                    # on the device: only the map itself is copied off it
                    self.map_species(members),
                    episode_players,
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
