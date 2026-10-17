"""Evolve a SlimeVolley policy by self-play.

Genomes are paired off against each other for fitness. Every 10th generation, the
10 fittest are each scored against EvoJAX's built-in baseline policy.

The population of 1000 wants a GPU; expect the best score against the baseline to
climb from about -10 to above -1 within the first 50 generations.

    uv run examples/slimevolley.py --generations 100
    uv run examples/slimevolley.py --monitor

Start the monitor first with `uv run turbo-neat-monitor`.
"""

import argparse
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from evojax.task import slimevolley
from evojax.task.slimevolley import Game, GameState, SlimeVolley

import neat.activations as act
from monitor import DEFAULT_URL, Episode, Monitor
from neat.config import GenomeConfig, MutationConfig, NEATConfig, SelectionConfig
from neat.fitness import make_mp_fitness_fn, make_test_fn
from neat.neat import NEAT
from neat.species import make_remove_last_if_stagnant_and_full_stagnation_fn
from neat.testing import make_top_k_test


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
    # each slime's observation, left then right
    obs: jax.Array
    steps: jax.Array
    key: jax.Array


def observe(game: Game) -> jax.Array:
    return jnp.stack(
        [game.agent_left.getObservation(), game.agent_right.getObservation()]
    )


class SlimeVolley2p:
    """SlimeVolley with both slimes controlled by the caller, for self-play: a
    multi-player environment of two players, the left slime then the right."""

    num_players = 2
    obs_shape = (12,)
    act_shape = (3,)

    def __init__(self, max_steps: int = 3000):
        self.max_steps = max_steps

        def reset_fn(key: jax.Array) -> State2p:
            next_key, key = jax.random.split(key)
            game_state = slimevolley.get_init_game_state_fn(key)
            return State2p(
                game_state=game_state,
                obs=observe(Game(game_state)),
                steps=jnp.zeros((), dtype=jnp.int32),
                key=next_key,
            )

        def step_fn(state: State2p, actions: jax.Array):
            next_key, key = jax.random.split(state.key)
            action_left, action_right = actions
            game = Game(state.game_state)
            game.setLeftAction(action_left)
            game.setRightAction(action_right)
            game.agent_left.setAction(action_left)
            game.agent_right.setAction(action_right)
            # the game scores from the perspective of the slime on the right
            reward_right = game.step()
            game_state = slimevolley.update_state_for_new_match(
                game.getGameState(), reward_right, key
            )
            steps = state.steps + 1
            done = steps >= max_steps
            state = State2p(
                game_state=game_state,
                obs=observe(game),
                steps=jnp.where(done, 0, steps),
                key=next_key,
            )
            return state, jnp.stack([-reward_right, reward_right]), done

        self._reset_fn = jax.jit(jax.vmap(reset_fn))
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jax.Array) -> State2p:
        return self._reset_fn(key)

    def step_mp(
        self, state: State2p, actions: jax.Array
    ) -> Tuple[State2p, jax.Array, jax.Array]:
        return self._step_fn(state, actions)


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
        input_activation=act.iden,
        output_activation=act.tanh,
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
    )

    return NEATConfig(
        mutation_config=mutation_config,
        genome_config=genome_config,
        selection_config=selection_config,
    )


def episode_fn(state):
    """A game as raw positions, for the monitor to draw: the ball's x, y, r and each
    slime's x, y, r, direction and lives; and what each policy saw, for it to show
    their networks' activations. Packs the test game against the baseline policy,
    and the members' recorded self-play games at once (with a leading game axis)."""
    game = state.game_state
    agents = [
        jnp.stack([a.x, a.y, a.r, a.direction, a.life], axis=-1)
        for a in (game.agent_left, game.agent_right)
    ]
    ball = jnp.stack([game.ball.x, game.ball.y, game.ball.r], axis=-1)
    if isinstance(state, State2p):
        # both slimes, left then right
        obs = dict(obs=state.obs)
    else:
        # against the baseline, the policy is the right slime
        obs = dict(obs=state.obs[..., None, :], obs_seats=jnp.array([1]))
    return Episode("slimevolley", ball=ball, left=agents[0], right=agents[1], **obs)


def main():
    parser = argparse.ArgumentParser(
        description="Evolve a SlimeVolley policy by self-play."
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

    selfplay = SlimeVolley2p()
    baseline = SlimeVolley()

    neat = NEAT(
        config=make_config(selfplay.obs_shape[0], selfplay.act_shape[0]),
        # each round is one game per genome, and a few games are too noisy a fitness
        # signal to learn from; the first round's games are recorded for the
        # monitor's lineage view
        fitness_fn=make_mp_fitness_fn(
            selfplay,
            steps_per_round=500,
            num_rounds=8,
            record=args.monitor is not None,
            parallel_rounds=2,
        ),
        # every 10th generation's 10 fittest, over 1000 games each
        test_fn=make_top_k_test(
            make_test_fn(baseline, num_steps=1000, num_episodes=1000, frames_len=300)
        ),
        monitor=Monitor(args.monitor, project="slimevolley") if args.monitor else None,
    )
    neat.run(seed=args.seed, num_generations=args.generations, episode_fn=episode_fn)


if __name__ == "__main__":
    main()
