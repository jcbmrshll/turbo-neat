from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from evojax.task.base import VectorizedTask

from neat.genome import Genome, forward
from neat.utils import apply


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FitnessState:
    task_state: jax.Array
    reward: jax.Array


def fitness(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    genome: Genome,
    num_steps: int,
    frames_len: int,
    **kwargs,
) -> Tuple[jax.Array, Any]:
    task_state = reset_fn(jax.random.split(rng, genome.batch_size))
    state = FitnessState(
        task_state=task_state,
        reward=jnp.zeros(genome.batch_size, dtype=jnp.float32),
    )

    def game_step(state, _):
        actions = apply(genome, forward, state.task_state.obs, **kwargs)
        task_state, rewards, _ = step_fn(state.task_state, actions)
        return FitnessState(
            task_state=task_state, reward=state.reward + rewards
        ), task_state

    state, task_state_frames = jax.lax.scan(game_step, state, jnp.zeros(num_steps))
    return state.reward, jax.tree.map(lambda x: x[:frames_len, 0], task_state_frames)


class TwoPlayerTask(Protocol):
    """A task stepped with one action per side, e.g. a self-play environment."""

    def reset(self, key: jax.Array, /) -> Any: ...

    def step_2p(
        self, state: Any, action_left: jax.Array, action_right: jax.Array, /
    ) -> Tuple[Any, jax.Array, jax.Array, jax.Array]: ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FitnessState2p:
    task_state: jax.Array
    reward_left: jax.Array
    reward_right: jax.Array


def fitness_2p(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    genome: Genome,
    steps_per_round: int,
    num_rounds: int,
    **kwargs,
):
    fitnesses = jnp.zeros(genome.batch_size)

    def many_games(fitnesses, rng):
        rng_left, rng_right, rng_init = jax.random.split(rng, 3)
        ids_left = jax.random.permutation(rng_left, genome.batch_size)
        ids_right = jax.random.permutation(rng_right, genome.batch_size)
        idxd_left = jax.tree.map(lambda x: x[ids_left], genome)
        idxd_right = jax.tree.map(lambda x: x[ids_right], genome)

        task_state = reset_fn(jax.random.split(rng_init, genome.batch_size))
        state = FitnessState2p(
            task_state=task_state,
            reward_left=jnp.zeros(
                genome.batch_size, dtype=task_state.reward_left.dtype
            ),
            reward_right=jnp.zeros(
                genome.batch_size, dtype=task_state.reward_right.dtype
            ),
        )

        def game_step(_, state):
            action_left = apply(idxd_left, forward, state.task_state.obs_left, **kwargs)
            action_right = apply(
                idxd_right, forward, state.task_state.obs_right, **kwargs
            )
            task_state, rewards_left, rewards_right, _ = step_fn(
                state.task_state, action_left, action_right
            )

            return FitnessState2p(
                task_state=task_state,
                reward_left=state.reward_left + rewards_left,
                reward_right=state.reward_right + rewards_right,
            )

        state = jax.lax.fori_loop(0, steps_per_round, game_step, state)
        fitnesses = (
            fitnesses.at[ids_left]
            .add(state.reward_left)
            .at[ids_right]
            .add(state.reward_right)
        )
        return fitnesses, None

    fitnesses, _ = jax.lax.scan(
        many_games, fitnesses, xs=jax.random.split(rng, num_rounds)
    )
    return fitnesses, None


def fitness_h2h(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    genome_1: Genome,
    genome_2: Genome,
    num_steps: int,
    **kwargs,
):
    task_state = reset_fn(jax.random.split(rng, genome_1.batch_size))
    state = FitnessState2p(
        task_state=task_state,
        reward_left=jnp.zeros(genome_1.batch_size, dtype=task_state.reward_left.dtype),
        reward_right=jnp.zeros(
            genome_1.batch_size, dtype=task_state.reward_right.dtype
        ),
    )

    def game_step(_, state):
        action_left = apply(genome_1, forward, state.task_state.obs_left, **kwargs)
        action_right = apply(genome_2, forward, state.task_state.obs_right, **kwargs)
        task_state, rewards_left, rewards_right, _ = step_fn(
            state.task_state, action_left, action_right
        )

        return FitnessState2p(
            task_state=task_state,
            reward_left=state.reward_left + rewards_left,
            reward_right=state.reward_right + rewards_right,
        )

    state = jax.lax.fori_loop(0, num_steps, game_step, state)
    return jnp.array([state.reward_left.mean(), state.reward_right.mean()]), None


def make_fitness_fn(
    task: VectorizedTask, num_steps: int, frames_len: Optional[int] = None
) -> Callable:
    assert num_steps > 0, "num_steps must be greater than 0"
    if frames_len is None:
        frames_len = num_steps
    frames_len = min(frames_len, num_steps)
    return partial(
        fitness,
        step_fn=task.step,
        reset_fn=task.reset,
        num_steps=num_steps,
        frames_len=frames_len,
    )


def make_2p_fitness_fn(
    task: TwoPlayerTask, steps_per_round: int, num_rounds: int
) -> Callable:
    assert num_rounds > 0, "num_rounds must be greater than 0"
    assert steps_per_round > 0, "steps_per_round must be greater than 0"
    return partial(
        fitness_2p,
        step_fn=task.step_2p,
        reset_fn=task.reset,
        steps_per_round=steps_per_round,
        num_rounds=num_rounds,
    )


def make_h2h_fitness_fn(task: TwoPlayerTask, num_steps: int) -> Callable:
    assert num_steps > 0, "num_steps must be greater than 0"
    return partial(
        fitness_h2h, step_fn=task.step_2p, reset_fn=task.reset, num_steps=num_steps
    )
