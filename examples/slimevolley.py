"""Evolve a SlimeVolley policy by self-play.

Genomes are paired off against each other for fitness. The best genome of each
generation then challenges the reigning champion head-to-head, and each new champion
is scored against EvoJAX's built-in baseline policy.

The population of 1000 wants a GPU; expect the score against the baseline to climb
from about -10 to above -1 within the first 50 generations.

    uv run examples/slimevolley.py --generations 100
    uv run examples/slimevolley.py --wandb-project turbo-neat

Rendering the network after each improvement needs the graphviz `dot` binary.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import wandb
from evojax.task import slimevolley
from evojax.task.slimevolley import Game, GameState, SlimeVolley

import neat_jax.activations as act
from neat_jax.config import GenomeConfig, MutationConfig, NEATConfig, SelectionConfig
from neat_jax.fitness import make_2p_fitness_fn, make_fitness_fn, make_h2h_fitness_fn
from neat_jax.neat import NEAT
from neat_jax.species import make_remove_last_if_stagnant_and_full_stagnation_fn

LOG_DIR = "./logs"


def get_random_ball_v(key: jax.Array) -> Tuple[jax.Array, jax.Array]:
    result = jax.random.uniform(key, shape=(2,)) * 2 - 1
    return result[0] * 20, result[1] * 7.5 + 17.5


# evojax indexes one past the end of `result` here, which draws both components of the
# serve from the same random number. Serve with independent components instead.
slimevolley.get_random_ball_v = get_random_ball_v


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class State2p:
    game_state: GameState
    obs_left: jax.Array
    obs_right: jax.Array
    reward_left: jax.Array
    reward_right: jax.Array
    steps: jax.Array
    key: jax.Array


class SlimeVolley2p:
    """SlimeVolley with both slimes controlled by the caller, for self-play."""

    obs_shape = (12,)
    act_shape = (3,)

    def __init__(self, max_steps: int = 3000):
        self.max_steps = max_steps

        def reset_fn(key: jax.Array) -> State2p:
            next_key, key = jax.random.split(key)
            game_state = slimevolley.get_init_game_state_fn(key)
            game = Game(game_state)
            return State2p(
                game_state=game_state,
                obs_left=game.agent_left.getObservation(),
                obs_right=game.agent_right.getObservation(),
                reward_left=jnp.zeros((), dtype=jnp.int32),
                reward_right=jnp.zeros((), dtype=jnp.int32),
                steps=jnp.zeros((), dtype=jnp.int32),
                key=next_key,
            )

        def step_fn(state: State2p, action_left: jax.Array, action_right: jax.Array):
            next_key, key = jax.random.split(state.key)
            game = Game(state.game_state)
            game.setLeftAction(action_left)
            game.setRightAction(action_right)
            game.agent_left.setAction(action_left)
            game.agent_right.setAction(action_right)
            # the game scores from the perspective of the slime on the right
            reward_right = game.step()
            reward_left = -reward_right
            game_state = slimevolley.update_state_for_new_match(
                game.getGameState(), reward_right, key
            )
            steps = state.steps + 1
            done = steps >= max_steps
            state = State2p(
                game_state=game_state,
                obs_left=game.agent_left.getObservation(),
                obs_right=game.agent_right.getObservation(),
                reward_left=reward_left,
                reward_right=reward_right,
                steps=jnp.where(done, 0, steps),
                key=next_key,
            )
            return state, reward_left, reward_right, done

        self._reset_fn = jax.jit(jax.vmap(reset_fn))
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jax.Array) -> State2p:
        return self._reset_fn(key)

    def step_2p(
        self, state: State2p, action_left: jax.Array, action_right: jax.Array
    ) -> Tuple[State2p, jax.Array, jax.Array, jax.Array]:
        return self._step_fn(state, action_left, action_right)


def make_config(input_size: int, output_size: int) -> NEATConfig:
    mutation_config = MutationConfig(
        add_connection_prob=0.05,
        add_node_prob=0.03,
        disable_connection_prob=0.01,
        disable_node_prob=0.006,
        mutate_weight_std=0.67,
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
        init_weight_std=6.7,
        activation_fns=[act.iden, act.sigmoid, act.tanh, act.relu, act.inv, act.exp],
        # identity on every input
        input_activation_ids=[0 for _ in range(input_size)],
        # tanh on every output
        output_activation_ids=[2 for _ in range(output_size)],
        input_labels=[
            "agent_x",
            "agent_y",
            "agent_vel_x",
            "agent_vel_y",
            "ball_x",
            "ball_y",
            "ball_vel_x",
            "ball_vel_y",
            "opp_x",
            "opp_y",
            "opp_vel_x",
            "opp_vel_y",
        ],
        output_labels=["forward", "backward", "jump"],
        capacity_growth_strategy="linear",
        init_mode="partial",
    )

    selection_config = SelectionConfig(
        population_size=1000,
        cutoff_pct=0.1,
        speciation_threshold=2.0,
        compatibility_coefficients=(1.0, 0.5),
        elitism=0.1,
        stagnation_fn=make_remove_last_if_stagnant_and_full_stagnation_fn(
            max_stagnated_steps=20, full_target=5
        ),
        maximum_species=5,
        selection_tournament_size=5,
        fitness_ema_period=5,
        offspring_temperature=0.65,
        min_species_size=25,
        species_warmup_threshold=5,
    )

    return NEATConfig(
        mutation_config=mutation_config,
        genome_config=genome_config,
        selection_config=selection_config,
    )


def render_fn(state):
    """Turn the frames of one game against the baseline policy into a gif for wandb."""
    os.makedirs(LOG_DIR, exist_ok=True)
    gif_file = os.path.join(LOG_DIR, "slimevolley.gif")
    screens = [
        SlimeVolley.render(jax.tree.map(lambda x: x[i], state))
        for i in range(state.obs.shape[0])
    ]
    screens[0].save(
        gif_file, save_all=True, append_images=screens[1:], duration=40, loop=0
    )
    return wandb.Video(gif_file)


def main():
    parser = argparse.ArgumentParser(
        description="Evolve a SlimeVolley policy by self-play."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument(
        "--wandb-project", default=None, help="log metrics and episode gifs to wandb"
    )
    args = parser.parse_args()

    selfplay = SlimeVolley2p()
    baseline = SlimeVolley()

    neat = NEAT(
        config=make_config(selfplay.obs_shape[0], selfplay.act_shape[0]),
        # a single round is too noisy a fitness signal to learn from
        fitness_fn=make_2p_fitness_fn(selfplay, steps_per_round=500, num_rounds=4),
        h2h_test_fn=make_h2h_fitness_fn(selfplay, num_steps=1000),
        baseline_test_fn=make_fitness_fn(baseline, num_steps=1000, frames_len=300),
        wandb_project=args.wandb_project,
    )
    neat.run(seed=args.seed, num_generations=args.generations, render_fn=render_fn)


if __name__ == "__main__":
    main()
