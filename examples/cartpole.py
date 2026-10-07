"""Evolve a policy for EvoJAX's cart-pole swing-up task.

The pole starts hanging below the cart; the policy pushes the cart left or right to
swing it up and balance it there; running off the end of the track ends the
episode. Each step scores up to 1 for a pole that's upright over a centered cart, so a perfect
1000-step episode scores just under 1000. Each genome's fitness is averaged over a
few episodes from different starts; on one, a lucky start can outscore a better
policy.

Expect the champion's average score over 1000 test episodes to pass 800 within the
first 100 generations, which take a few minutes on a GPU. This config isn't tuned
for --harder yet: there it stalls at about 400.

    uv run examples/cartpole.py --generations 100
    uv run examples/cartpole.py --harder        # random starts, anywhere on the track
    uv run examples/cartpole.py --monitor

Start the monitor first with `uv run turbo-neat-monitor`.
"""

import argparse

from evojax.task.cartpole import DELTA_T, CartPoleSwingUp

import neat.activations as act
from monitor import DEFAULT_URL, Episode, Monitor
from neat.config import GenomeConfig, MutationConfig, NEATConfig, SelectionConfig
from neat.fitness import make_fitness_fn
from neat.neat import NEAT
from neat.species import make_remove_last_if_stagnant_and_full_stagnation_fn

NUM_STEPS = 1000
NUM_EPISODES = 4
# keep every few steps of an episode: the monitor plays them back at 25 fps, so this
# is real time
FRAME_STRIDE = round(1 / (25 * DELTA_T))


def make_config(input_size: int, output_size: int) -> NEATConfig:
    mutation_config = MutationConfig(
        add_connection_prob=0.05,
        add_node_prob=0.03,
        disable_connection_prob=0.01,
        disable_node_prob=0.006,
        mutate_weight_std=0.5,
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
        # tanh on the output: the task clips the force to [-1, 1]
        output_activation_ids=[2 for _ in range(output_size)],
        input_labels=["x", "x_vel", "cos_theta", "sin_theta", "theta_vel"],
        output_labels=["force"],
        capacity_growth_strategy="linear",
        init_mode="full",
    )

    selection_config = SelectionConfig(
        population_size=1000,
        cutoff_pct=0.1,
        speciation_threshold=1.0,
        compatibility_coefficients=(1.0, 0.5),
        elitism=0.1,
        stagnation_fn=make_remove_last_if_stagnant_and_full_stagnation_fn(
            max_stagnated_steps=20, full_target=5
        ),
        maximum_species=5,
        selection_tournament_size=5,
        min_species_size=25,
        species_warmup_threshold=5,
    )

    return NEATConfig(
        mutation_config=mutation_config,
        genome_config=genome_config,
        selection_config=selection_config,
    )


def episode_fn(state):
    """An episode as raw cart states (x, x_vel, theta, theta_vel), for the monitor to
    draw; also packs the members' recorded episodes at once, with a leading episode
    axis."""
    return Episode("cartpole", state=state.state[..., ::FRAME_STRIDE, :])


def main():
    parser = argparse.ArgumentParser(
        description="Evolve a policy for EvoJAX's cart-pole swing-up task."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument(
        "--harder",
        action="store_true",
        help="start from random states anywhere on the track, not just hanging",
    )
    parser.add_argument(
        "--monitor",
        nargs="?",
        const=DEFAULT_URL,
        default=None,
        metavar="URL",
        help=f"log to a turbo-neat monitor (default {DEFAULT_URL})",
    )
    args = parser.parse_args()

    train_task = CartPoleSwingUp(NUM_STEPS, harder=args.harder)
    test_task = CartPoleSwingUp(NUM_STEPS, harder=args.harder, test=True)

    neat = NEAT(
        config=make_config(train_task.obs_shape[0], train_task.act_shape[0]),
        # record every member's first episode for the monitor's lineage view
        fitness_fn=make_fitness_fn(
            task=train_task,
            num_steps=NUM_STEPS,
            num_episodes=NUM_EPISODES,
            record=args.monitor is not None,
        ),
        baseline_test_fn=make_fitness_fn(task=test_task, num_steps=NUM_STEPS),
        monitor=Monitor(args.monitor, project="cartpole") if args.monitor else None,
    )
    neat.run(seed=args.seed, num_generations=args.generations, episode_fn=episode_fn)


if __name__ == "__main__":
    main()
