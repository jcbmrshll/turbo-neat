from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from neat.activations import (
    ActivationFn,
    ActivationSelector,
    iden,
    make_activation_selector_fn,
)

# one activation function for every node of a kind, or a list with one per node
Activations = Union[ActivationFn, Sequence[ActivationFn]]


@dataclass
class MutationConfig:
    """How offspring mutate each generation; elites never do.

    The structural mutations (adding or disabling a node or a connection) each
    happen to an offspring at most once, with their probability. The others happen
    to each of its connections or nodes separately."""

    # chance of adding a connection from a random node that isn't an output to a
    # random node that isn't an input, isn't already connected to it and wouldn't
    # make a cycle; a pair connected before by a now-disabled connection gets it back
    # with a new weight. Weights are drawn like a new genome's (see GenomeConfig)
    add_connection_prob: float
    # chance of disabling a random enabled connection
    disable_connection_prob: float
    # chance of splitting a random enabled connection with a new hidden node: the old
    # connection is disabled, the new node's incoming weight is 1 and its outgoing
    # weight the old connection's, and it gets a random activation from
    # activation_fns and a bias of 0
    add_node_prob: float
    # chance of removing a random hidden node and disabling its connections
    disable_node_prob: float
    # chance that each enabled connection's weight gets normal noise of this std
    mutate_weight_prob: float
    mutate_weight_std: float
    # chance that each hidden node's activation is redrawn from activation_fns
    mutate_activation_prob: float
    # chance that each hidden node's bias gets normal noise of this std
    mutate_bias_prob: float
    mutate_bias_std: float
    # also mutate the activations or biases of output or input nodes
    allow_output_act_mutation: bool = False
    allow_output_bias_mutation: bool = False
    allow_input_act_mutation: bool = False
    allow_input_bias_mutation: bool = False
    # only used for backprop neat, which trains weights and biases by gradient
    # descent instead of mutating them; trained weights are inherited
    learning_rate: float = 0.0
    backprop_steps: int = 1  # gradient steps per generation, before scoring
    max_grad_norm: Optional[float] = None  # per-genome gradient clipping, if set


@dataclass
class SelectionConfig:
    """How members are grouped into species, and how species reproduce.

    Each generation, every member joins a species, and the next generation's places
    are shared out among the species by their fitness: an EMA, over generations, of
    their members' mean fitness. In each species the fittest carry on unchanged as
    elites, and the rest of its places go to offspring: the crossover of two of its
    members, mutated."""

    population_size: int
    # the fraction of each species, rounded down, that doesn't reproduce: its least
    # fit members
    cutoff_pct: float
    # (c_connections, c_weights): the distance between two genomes is c_connections
    # times the number of connections only one of them has, over the larger genome's
    # number of connections, plus c_weights times the mean absolute difference of
    # the weights of the connections they share
    compatibility_coefficients: Tuple[float, float]
    # which species to remove for not improving, checked every generation (see the
    # make_*_stagnation_fn in neat.species); a removed species gets no places
    stagnation_fn: Callable
    # places are shared out by a softmax of the species' fitness, scaled to [0, 1],
    # over this temperature: at 1.0 the fittest species gets at most e times the
    # share of the least fit, and a lower temperature favors it more
    offspring_temperature: float = 1.0
    # the period, in generations, of the EMA of each species' fitness; a species
    # only counts as stagnating once it is this many generations old
    fitness_ema_period: int = 10
    # each parent is the fittest of this many members drawn at random from those of
    # its species that reproduce
    selection_tournament_size: int = 3
    # a member whose mean distance to every species' members is above this starts a
    # new species, if there are fewer than maximum_species; otherwise it joins the
    # nearest species
    speciation_threshold: float = 3.0
    # if set, new species only start while there are fewer than this, and the
    # threshold above is where it starts: after each generation's speciation it's
    # divided by speciation_threshold_step while there are fewer species than this
    # (never below min_speciation_threshold), and held otherwise, but never more
    # than a step above the furthest any member was from its nearest species, so
    # that a species that dies out is soon replaced by the members furthest from
    # the rest
    target_species: Optional[int] = None
    speciation_threshold_step: float = 1.05
    min_speciation_threshold: float = 0.05
    # the fraction of each species, rounded down, carried over unchanged as elites:
    # its fittest members, at most as many as the species has places
    elitism: float = 0.1
    # if set, a member only joins a species other than its fitter parent's if that
    # species is at most this many generations old
    max_transfer_age: Optional[int] = None
    maximum_species: int = 10


@dataclass
class GenomeConfig:
    """The shape of a genome and how it starts out."""

    input_size: int
    output_size: int
    # node slots and connection slots per genome: genomes are fixed-size arrays, so a
    # population can be vmapped. When a genome comes within 3 connections of full,
    # every genome grows by capacity_growth_strategy, which recompiles the jitted
    # steps
    initial_capacity: int
    # the normal distribution weights are drawn from, for a new genome's connections
    # and for connections added by mutation
    init_weight_mean: float
    init_weight_std: float
    # the activation functions hidden nodes choose from
    activation_fns: List[ActivationFn]
    # the activation of every input (or output) node, or a list with one per node;
    # they needn't be in activation_fns, and hidden nodes only take them if they are
    input_activation: Activations = iden
    output_activation: Activations = iden
    # node names for the monitor's network view
    input_labels: List[str] = field(default_factory=list)
    output_labels: List[str] = field(default_factory=list)
    # how genomes grow when one is nearly full: "linear" adds initial_capacity slots,
    # "exponential" doubles them, "constant" never grows (so initial_capacity has to
    # be big enough that no genome fills up)
    capacity_growth_strategy: str = "linear"
    # how a new genome is connected: "full" connects every input to every output,
    # "partial" each input to one random output, "empty" not at all (nodes are only
    # added by splitting connections, so the first must come from add_connection_prob)
    init_mode: str = "partial"
    # every activation function a node can have, indexed by a node's activation id:
    # activation_fns, then any input or output activation that isn't one of them
    activations: List[ActivationFn] = field(init=False)
    input_activation_ids: List[int] = field(init=False)
    output_activation_ids: List[int] = field(init=False)

    def __post_init__(self):
        self.activations = list(self.activation_fns)
        self.input_activation_ids = self._activation_ids(
            self.input_activation, self.input_size, "input"
        )
        self.output_activation_ids = self._activation_ids(
            self.output_activation, self.output_size, "output"
        )

    def _activation_ids(self, fns: Activations, size: int, kind: str) -> List[int]:
        if callable(fns):
            fns = [fns] * size
        if len(fns) != size:
            raise ValueError(
                f"{kind}_activation has {len(fns)} functions for {size} {kind}s"
            )
        for fn in fns:
            if fn not in self.activations:
                self.activations.append(fn)
        return [self.activations.index(fn) for fn in fns]

    @property
    def num_activation_functions(self) -> int:
        """How many activation functions hidden nodes choose from (the first of
        activations)."""
        return len(self.activation_fns)

    @property
    def activation_selector(self) -> ActivationSelector:
        return make_activation_selector_fn(self.activations)


@dataclass
class NEATConfig:
    # mutation
    mutation_config: MutationConfig
    # genome
    genome_config: GenomeConfig
    # selection
    selection_config: SelectionConfig

    def to_dict(self) -> Dict:
        return {
            "mutation_config": self.mutation_config.__dict__,
            "selection_config": self.selection_config.__dict__,
            "genome_config": self.genome_config.__dict__,
        }
