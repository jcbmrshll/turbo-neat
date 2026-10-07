from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from evojax.task.base import VectorizedTask

from neat.genome import Genome, forward
from neat.utils import apply, mask_data


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Recording:
    """What a fitness evaluation played, for the monitor: one episode per member (or
    per game, for two-player tasks).

    frames is the task state at every recorded step, with leading (episode, step)
    axes, and
    players holds the batch index of each genome playing each episode, shape
    (episodes, players)."""

    frames: Any
    players: jax.Array


def _episode_major(frames: Any) -> Any:
    """Task states stacked by lax.scan, (step, episode, ...), as (episode, step, ...)."""
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), frames)


def _scan_every(
    step: Callable, carry: Any, num_steps: int, every: int, frame: Callable
) -> Tuple[Any, Any]:
    """step carry num_steps times, keeping frame(carry) after every `every` steps;
    the final carry, and the frames stacked on a leading axis."""

    def chunk(carry, _):
        carry = jax.lax.fori_loop(0, every, lambda _, c: step(c), carry)
        return carry, frame(carry)

    carry, frames = jax.lax.scan(chunk, carry, length=num_steps // every)
    carry = jax.lax.fori_loop(0, num_steps % every, lambda _, c: step(c), carry)
    return carry, frames


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FitnessState:
    task_state: jax.Array
    reward: jax.Array
    # how many times each episode's task has reset (reported done) while scoring
    resets: jax.Array


def _keep_where(mask: jax.Array, new: Any, old: Any) -> Any:
    """new where mask, else old, for every leaf of a batched pytree."""
    return jax.tree.map(
        lambda n, o: jnp.where(mask.reshape(mask.shape + (1,) * (n.ndim - 1)), n, o),
        new,
        old,
    )


def rollout(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    policy: Callable[[jax.Array], jax.Array],
    num_envs: int,
    num_steps: int,
    max_resets: Optional[int],
    keep: Optional[Callable[[Any], Any]] = None,
) -> Tuple[jax.Array, Any]:
    """Play num_envs episodes side by side for num_steps steps, with policy mapping
    their batched observations to actions. Returns each episode's total reward,
    and keep(task state) at every step (None without keep).

    Each done the task reports is a reset: the task starts over from a new state.
    Reward counts until the task has reset more than max_resets times, after which
    the task state is held where it was, rather than playing on from a new start.
    So 0 scores one episode, like evojax's rollouts; k lets an episode use k extra
    starts, like lives; None scores straight through every reset, for tasks whose
    done just starts a new round."""
    # None never runs out of resets
    limit = jnp.iinfo(jnp.int32).max if max_resets is None else max_resets
    state = FitnessState(
        task_state=reset_fn(jax.random.split(rng, num_envs)),
        reward=jnp.zeros(num_envs, dtype=jnp.float32),
        resets=jnp.zeros(num_envs, dtype=jnp.int32),
    )

    def game_step(state, _):
        task_state, rewards, done = step_fn(
            state.task_state, policy(state.task_state.obs)
        )
        # the step that resets the task still counts
        counted = state.resets <= limit
        reward = state.reward + jnp.where(counted, rewards, 0)
        resets = state.resets + (counted & done.astype(bool))
        # past the last counted reset, keep the task's last state
        task_state = _keep_where(resets <= limit, task_state, state.task_state)
        state = FitnessState(task_state=task_state, reward=reward, resets=resets)
        return state, None if keep is None else keep(task_state)

    state, frames = jax.lax.scan(game_step, state, length=num_steps)
    return state.reward, frames


def fitness(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    genome: Genome,
    num_steps: int,
    num_episodes: int,
    max_resets: Optional[int],
    record: bool,
    **kwargs,
) -> Tuple[jax.Array, Optional[Recording]]:
    """Scores a population: each genome's total reward, averaged over num_episodes
    episodes from different starts (see rollout for max_resets), and, if record, a
    Recording of every genome's first episode."""

    def policy(obs):
        return apply(genome, forward, obs, **kwargs)

    play = partial(
        rollout,
        step_fn,
        reset_fn,
        policy=policy,
        num_envs=genome.batch_size,
        num_steps=num_steps,
        max_resets=max_resets,
    )
    reward, frames = play(rng, keep=(lambda s: s) if record else None)
    if num_episodes > 1:
        # the rest of the episodes only count towards fitness
        rngs = jax.random.split(jax.random.fold_in(rng, 1), num_episodes - 1)
        rewards, _ = jax.vmap(play)(rngs)
        reward = (reward + rewards.sum(axis=0)) / num_episodes
    if not record:
        return reward, None
    players = jnp.arange(genome.batch_size)[:, None]
    return reward, Recording(_episode_major(frames), players)


def test(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    genome: Genome,
    num_steps: int,
    num_episodes: int,
    max_resets: Optional[int],
    frames_len: int,
    frames_every: int = 1,
    **kwargs,
) -> Tuple[jax.Array, Any]:
    """Scores one genome: its mean total reward over num_episodes episodes from
    different starts (see rollout for max_resets), and the task state over the first
    frames_len steps of the first episode, every frames_every steps, to show it."""

    def policy(obs):
        return forward(genome, obs, **kwargs)

    rewards, frames = rollout(
        step_fn,
        reset_fn,
        rng,
        policy,
        num_envs=num_episodes,
        num_steps=num_steps,
        max_resets=max_resets,
        keep=lambda s: jax.tree.map(lambda x: x[0], s),
    )
    shown = slice(frames_every - 1, frames_len, frames_every)
    return rewards.mean(), jax.tree.map(lambda x: x[shown], frames)


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


class MultiPlayerTask(Protocol):
    """A task where num_players genomes play each game, e.g. a race.

    The state's obs is (games, num_players, obs_size), and step_mp takes actions of
    (games, num_players, act_size) and returns rewards of (games, num_players)."""

    num_players: int

    def reset(self, key: jax.Array, /) -> Any: ...

    def step_mp(
        self, state: Any, actions: jax.Array, /
    ) -> Tuple[Any, jax.Array, jax.Array]: ...


def apply_seats(genome: Genome, obs: jax.Array, **kwargs) -> jax.Array:
    """Actions of a batch of genomes laid out as (games, seats), for obs of
    (games, seats, obs_size)."""
    games, seats = obs.shape[:2]
    actions = apply(genome, forward, obs.reshape(games * seats, -1), **kwargs)
    return actions.reshape(games, seats, -1)


def race(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    seated: Genome,
    num_games: int,
    num_steps: int,
    record: bool = False,
    record_every: int = 1,
    record_fn: Optional[Callable] = None,
    **kwargs,
) -> Tuple[jax.Array, Any]:
    """Total reward of each seat over num_games games of num_steps, where seated is
    a batch of genomes laid out as (games, seats) and flattened; and, if record,
    every game's task state every record_every steps (or what record_fn keeps of
    it), with leading (game, step) axes."""
    task_state = reset_fn(jax.random.split(rng, num_games))

    def game_step(carry):
        task_state, rewards = carry
        actions = apply_seats(seated, task_state.obs, **kwargs)
        task_state, step_rewards, _ = step_fn(task_state, actions)
        return task_state, rewards + step_rewards

    carry = (task_state, jnp.zeros(task_state.obs.shape[:2]))
    if record:
        keep = record_fn or (lambda task_state: task_state)
        (_, rewards), frames = _scan_every(
            game_step, carry, num_steps, record_every, lambda carry: keep(carry[0])
        )
        return rewards, _episode_major(frames)
    _, rewards = jax.lax.fori_loop(0, num_steps, lambda _, c: game_step(c), carry)
    return rewards, None


def fitness_mp(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    genome: Genome,
    num_players: int,
    steps_per_round: int,
    num_rounds: int,
    record: bool = False,
    record_every: int = 1,
    record_fn: Optional[Callable] = None,
    parallel_rounds: int = 1,
    **kwargs,
):
    """Each round shuffles the population into games of num_players. When the
    population doesn't divide evenly, the last game is filled with genomes that
    also play elsewhere, so fitness is the mean reward per game played. If record,
    also returns a Recording of the first round's games.

    Rounds run parallel_rounds at a time, side by side, which is faster as long as
    one round's games leave the accelerator room; the fitness is the same."""
    num_games = -(-genome.batch_size // num_players)
    num_seats = num_games * num_players

    def one_round(rng, record=False):
        """The genomes in each seat, their rewards, and the games if record."""
        rng_seats, rng_fill, rng_game = jax.random.split(rng, 3)
        ids = jnp.concatenate(
            [
                jax.random.permutation(rng_seats, genome.batch_size),
                jax.random.choice(
                    rng_fill,
                    genome.batch_size,
                    (num_seats - genome.batch_size,),
                    replace=False,
                ),
            ]
        )
        seated = jax.tree.map(lambda x: x[ids], genome)
        rewards, frames = race(
            step_fn,
            reset_fn,
            rng_game,
            seated,
            num_games,
            steps_per_round,
            record=record,
            record_every=record_every,
            record_fn=record_fn,
            **kwargs,
        )
        return ids, rewards.ravel(), frames

    def tally(carry, ids, rewards):
        totals, counts = carry
        return totals.at[ids].add(rewards), counts.at[ids].add(1)

    def some_rounds(carry, rngs):
        ids, rewards, _ = jax.vmap(one_round)(rngs)
        return tally(carry, ids.ravel(), rewards.ravel()), None

    zeros = jnp.zeros(genome.batch_size)
    carry, rngs, recording = (zeros, zeros), jax.random.split(rng, num_rounds), None
    if record:
        # one round's games are enough to show: every genome plays in one of them
        ids, rewards, frames = one_round(rngs[0], record=True)
        carry = tally(carry, ids, rewards)
        recording = Recording(frames, ids.reshape(num_games, num_players))
        rngs = rngs[1:]
    # the rest in batches of parallel_rounds, then any left over
    batch = max(1, min(parallel_rounds, len(rngs)))
    batches = len(rngs) // batch
    if batches:
        stacked = rngs[: batches * batch].reshape((batches, batch) + rngs.shape[1:])
        carry, _ = jax.lax.scan(some_rounds, carry, xs=stacked)
    if len(rngs) % batch:
        carry, _ = some_rounds(carry, rngs[batches * batch :])
    totals, counts = carry
    return totals / counts, recording


def fitness_mp_h2h(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    genome_1: Genome,
    genome_2: Genome,
    num_players: int,
    num_steps: int,
    **kwargs,
):
    """genome_1 against genome_2 with the seats alternating between them, shifted
    by one each game so that both start from every seat. genome_1 and genome_2 are
    each one genome broadcast over the batch; the games seat as many players as the
    batch holds. Returns each genome's mean reward per seat."""
    num_games = -(-genome_1.batch_size // num_players)
    games, seats = np.arange(num_games)[:, None], np.arange(num_players)[None]
    is_first = ((games + seats) % 2 == 0).ravel()

    def seat(x, y):
        shape = (num_games * num_players,) + x.shape[1:]
        return mask_data(
            jnp.broadcast_to(x[0], shape), jnp.broadcast_to(y[0], shape), is_first
        )

    seated = jax.tree.map(seat, genome_1, genome_2)
    rewards, _ = race(step_fn, reset_fn, rng, seated, num_games, num_steps, **kwargs)
    rewards = rewards.ravel()
    return jnp.array([rewards[is_first].mean(), rewards[~is_first].mean()]), None


def fitness_mp_field(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    genome: Genome,
    num_races: int,
    num_steps: int,
    record_every: int = 1,
    record_fn: Optional[Callable] = None,
    **kwargs,
) -> Tuple[jax.Array, Any]:
    """The batch of genomes all racing each other, num_races times: race r starts
    them r places round the grid from race 0, so that each starts from every place
    as often. Returns each genome's mean reward per race, and race 0's task state
    every record_every steps (or what record_fn keeps of it), genome i in seat i."""
    num_players = genome.batch_size
    games, seats = np.arange(num_races)[:, None], np.arange(num_players)[None]
    ids = (seats + games) % num_players
    seated = jax.tree.map(lambda x: x[ids.ravel()], genome)
    rewards, frames = race(
        step_fn,
        reset_fn,
        rng,
        seated,
        num_races,
        num_steps,
        record=True,
        record_every=record_every,
        record_fn=record_fn,
        **kwargs,
    )
    totals = jnp.zeros(num_players).at[ids.ravel()].add(rewards.ravel())
    return totals / num_races, jax.tree.map(lambda x: x[0], frames)


def make_fitness_fn(
    task: VectorizedTask,
    num_steps: int,
    num_episodes: int = 1,
    max_resets: Optional[int] = 0,
    record: bool = False,
) -> Callable:
    """A training fitness: scores every genome of a population on task (see
    fitness)."""
    assert num_steps > 0, "num_steps must be greater than 0"
    assert num_episodes > 0, "num_episodes must be greater than 0"
    assert max_resets is None or max_resets >= 0, "max_resets must be at least 0"
    return partial(
        fitness,
        step_fn=task.step,
        reset_fn=task.reset,
        num_steps=num_steps,
        num_episodes=num_episodes,
        max_resets=max_resets,
        record=record,
    )


def make_test_fn(
    task: VectorizedTask,
    num_steps: int,
    num_episodes: int,
    max_resets: Optional[int] = 0,
    frames_len: Optional[int] = None,
    frames_every: int = 1,
) -> Callable:
    """A test of one genome, e.g. the best of a generation, on task (see test)."""
    assert num_steps > 0, "num_steps must be greater than 0"
    assert num_episodes > 0, "num_episodes must be greater than 0"
    assert max_resets is None or max_resets >= 0, "max_resets must be at least 0"
    if frames_len is None:
        frames_len = num_steps
    return partial(
        test,
        step_fn=task.step,
        reset_fn=task.reset,
        num_steps=num_steps,
        num_episodes=num_episodes,
        max_resets=max_resets,
        frames_len=min(frames_len, num_steps),
        frames_every=frames_every,
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


def make_mp_fitness_fn(
    task: MultiPlayerTask,
    steps_per_round: int,
    num_rounds: int,
    record: bool = False,
    record_every: int = 1,
    record_fn: Optional[Callable] = None,
    parallel_rounds: int = 1,
) -> Callable:
    assert num_rounds > 0, "num_rounds must be greater than 0"
    assert steps_per_round > 0, "steps_per_round must be greater than 0"
    assert parallel_rounds > 0, "parallel_rounds must be greater than 0"
    return partial(
        fitness_mp,
        step_fn=task.step_mp,
        reset_fn=task.reset,
        num_players=task.num_players,
        steps_per_round=steps_per_round,
        num_rounds=num_rounds,
        record=record,
        record_every=record_every,
        record_fn=record_fn,
        parallel_rounds=parallel_rounds,
    )


def make_mp_h2h_fitness_fn(task: MultiPlayerTask, num_steps: int) -> Callable:
    assert task.num_players > 1, "a head-to-head needs at least two players"
    assert num_steps > 0, "num_steps must be greater than 0"
    return partial(
        fitness_mp_h2h,
        step_fn=task.step_mp,
        reset_fn=task.reset,
        num_players=task.num_players,
        num_steps=num_steps,
    )


def make_mp_field_fn(
    task: MultiPlayerTask,
    num_races: int,
    num_steps: int,
    record_every: int = 1,
    record_fn: Optional[Callable] = None,
) -> Callable:
    assert num_races > 0, "num_races must be greater than 0"
    assert num_steps > 0, "num_steps must be greater than 0"
    return partial(
        fitness_mp_field,
        step_fn=task.step_mp,
        reset_fn=task.reset,
        num_races=num_races,
        num_steps=num_steps,
        record_every=record_every,
        record_fn=record_fn,
    )
