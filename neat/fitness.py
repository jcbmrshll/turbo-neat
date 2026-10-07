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
class Recording:
    """What a fitness evaluation played, for the monitor: one episode per member (or
    per game, for two-player tasks).

    frames is the task state at every step, with leading (episode, step) axes, and
    players holds the batch index of each genome playing each episode, shape
    (episodes, players)."""

    frames: Any
    players: jax.Array


def _episode_major(frames: Any) -> Any:
    """Task states stacked by lax.scan, (step, episode, ...), as (episode, step, ...)."""
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), frames)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FitnessState:
    task_state: jax.Array
    reward: jax.Array
    # false once a genome's episode is done
    alive: jax.Array


def _keep_where(mask: jax.Array, new: Any, old: Any) -> Any:
    """new where mask, else old, for every leaf of a batched pytree."""
    return jax.tree.map(
        lambda n, o: jnp.where(mask.reshape(mask.shape + (1,) * (n.ndim - 1)), n, o),
        new,
        old,
    )


def fitness(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    genome: Genome,
    num_steps: int,
    frames_len: int,
    num_episodes: int = 1,
    record: bool = False,
    **kwargs,
) -> Tuple[jax.Array, Any]:
    """Returns the total reward of each genome, averaged over num_episodes episodes
    from different starts, and from the first of them either a Recording of every
    genome's episode (if record) or the first genome's first frames_len frames.

    An episode ends at the task's first done, like evojax's rollouts: the reward
    stops counting and the task state is held there, rather than playing on from
    the start the task resets to. So done is the task's to decide: a task that
    should be scored through its resets (a new round, say) shouldn't report done
    for them."""

    def episode(rng, record):
        task_state = reset_fn(jax.random.split(rng, genome.batch_size))
        state = FitnessState(
            task_state=task_state,
            reward=jnp.zeros(genome.batch_size, dtype=jnp.float32),
            alive=jnp.ones(genome.batch_size, dtype=bool),
        )

        def game_step(state, _):
            actions = apply(genome, forward, state.task_state.obs, **kwargs)
            task_state, rewards, done = step_fn(state.task_state, actions)
            reward = state.reward + jnp.where(state.alive, rewards, 0)
            alive = state.alive & ~done.astype(bool)
            # the step that ends an episode resets the task; keep its last state
            task_state = _keep_where(alive, task_state, state.task_state)
            # every genome's states if recording, else just the first genome's
            frame = task_state if record else jax.tree.map(lambda x: x[0], task_state)
            return FitnessState(
                task_state=task_state, reward=reward, alive=alive
            ), frame

        state, frames = jax.lax.scan(game_step, state, jnp.zeros(num_steps))
        return state.reward, frames

    reward, frames = episode(rng, record)
    if num_episodes > 1:
        # the rest of the episodes only count towards fitness
        rngs = jax.random.split(jax.random.fold_in(rng, 1), num_episodes - 1)
        rewards, _ = jax.vmap(partial(episode, record=False))(rngs)
        reward = (reward + rewards.sum(axis=0)) / num_episodes
    if record:
        players = jnp.arange(genome.batch_size)[:, None]
        return reward, Recording(_episode_major(frames), players)
    return reward, jax.tree.map(lambda x: x[:frames_len], frames)


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
    record: bool = False,
    **kwargs,
):
    """Returns each genome's total reward over num_rounds rounds of games against
    random opponents, and, if record, a Recording of the first round's games."""
    fitnesses = jnp.zeros(genome.batch_size)

    def many_games(fitnesses, rng, record=False):
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

        recording = None
        if record:
            # scan rather than loop, to keep every step's state
            def recorded_step(state, _):
                state = game_step(None, state)
                return state, state.task_state

            state, frames = jax.lax.scan(recorded_step, state, length=steps_per_round)
            players = jnp.stack([ids_left, ids_right], axis=-1)
            recording = Recording(_episode_major(frames), players)
        else:
            state = jax.lax.fori_loop(0, steps_per_round, game_step, state)
        fitnesses = (
            fitnesses.at[ids_left]
            .add(state.reward_left)
            .at[ids_right]
            .add(state.reward_right)
        )
        return fitnesses, recording

    rngs = jax.random.split(rng, num_rounds)
    recording = None
    if record:
        # one round's games are enough to show: every genome plays in two of them
        fitnesses, recording = many_games(fitnesses, rngs[0], record=True)
        rngs = rngs[1:]
    fitnesses, _ = jax.lax.scan(many_games, fitnesses, xs=rngs)
    return fitnesses, recording


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
    task: VectorizedTask,
    num_steps: int,
    frames_len: Optional[int] = None,
    num_episodes: int = 1,
    record: bool = False,
) -> Callable:
    """Fitness is the total reward over num_steps, averaged over num_episodes from
    different starts; see fitness() for the auxiliary data."""
    assert num_steps > 0, "num_steps must be greater than 0"
    assert num_episodes > 0, "num_episodes must be greater than 0"
    if frames_len is None:
        frames_len = num_steps
    frames_len = min(frames_len, num_steps)
    return partial(
        fitness,
        step_fn=task.step,
        reset_fn=task.reset,
        num_steps=num_steps,
        frames_len=frames_len,
        num_episodes=num_episodes,
        record=record,
    )


def make_2p_fitness_fn(
    task: TwoPlayerTask, steps_per_round: int, num_rounds: int, record: bool = False
) -> Callable:
    assert num_rounds > 0, "num_rounds must be greater than 0"
    assert steps_per_round > 0, "steps_per_round must be greater than 0"
    return partial(
        fitness_2p,
        step_fn=task.step_2p,
        reset_fn=task.reset,
        steps_per_round=steps_per_round,
        num_rounds=num_rounds,
        record=record,
    )


def make_h2h_fitness_fn(task: TwoPlayerTask, num_steps: int) -> Callable:
    assert num_steps > 0, "num_steps must be greater than 0"
    return partial(
        fitness_h2h, step_fn=task.step_2p, reset_fn=task.reset, num_steps=num_steps
    )
