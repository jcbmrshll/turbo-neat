from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from neat.genome import Genome, forward
from neat.utils import apply, mask_data


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Recording:
    """What a fitness evaluation played, for the monitor: one episode per member (or
    per game, for multi-player environments).

    frames is the environment state at every recorded step, with leading
    (episode, step) axes, and players holds the batch index of each genome playing
    each episode, shape (episodes, players)."""

    frames: Any
    players: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Scores:
    """What a fitness evaluation found besides fitness: every genome's reward in
    each episode it played, (episodes, genomes), which fitness sums up, and the
    Recording of what it played, if it was made to record."""

    returns: jax.Array
    recording: Optional[Recording] = None


def describe_stats(stats: Dict[str, jax.Array]) -> Dict[str, Any]:
    """Metrics of stats by name, a value per genome (NaN where it doesn't apply):
    their mean, least and most over the genomes they apply to, as mean_<name>,
    min_<name> and max_<name>, NaN where they apply to none."""
    metrics = {}
    for name, values in stats.items():
        for agg, fn in (
            ("mean", jnp.nanmean),
            ("min", jnp.nanmin),
            ("max", jnp.nanmax),
        ):
            metrics[f"{agg}_{name}"] = fn(values)
    return metrics


def _episode_major(frames: Any) -> Any:
    """Environment states stacked by lax.scan, (step, episode, ...), as
    (episode, step, ...)."""
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
    env_state: jax.Array
    # per episode, or per player in each episode
    reward: jax.Array
    # how many times each episode's environment has reset (reported done) while scoring
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
    keep_every: int = 1,
    replicate_rng: bool = False,
) -> Tuple[jax.Array, Any]:
    """Each episode's total reward, and what keep kept (see play)."""
    state, frames = play(
        step_fn,
        reset_fn,
        rng,
        policy,
        num_envs,
        num_steps,
        max_resets,
        keep=keep,
        keep_every=keep_every,
        replicate_rng=replicate_rng,
    )
    return state.reward, frames


def play(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    policy: Callable[[jax.Array], jax.Array],
    num_envs: int,
    num_steps: int,
    max_resets: Optional[int],
    keep: Optional[Callable[[Any], Any]] = None,
    keep_every: int = 1,
    replicate_rng: bool = False,
) -> Tuple[FitnessState, Any]:
    """Play num_envs episodes side by side for num_steps steps, with policy mapping
    their batched observations to actions. Returns the final FitnessState, with each
    episode's total reward and environment state, and keep(environment state) every
    keep_every steps (None without keep), stacked on a leading step axis.

    step_fn's rewards are per episode, (num_envs,), or per player in each episode,
    (num_envs, players) for a multi-player environment; the totals keep the same
    shape.

    Each done the environment reports is a reset: it starts over from a new state.
    Reward counts until the environment has reset more than max_resets times, after
    which its state is held where it was, rather than playing on from a new start.
    So 0 scores one episode, like evojax's rollouts; k lets an episode use k extra
    starts, like lives; None scores straight through every reset, for environments
    whose done just starts a new round.

    With replicate_rng, every episode resets with the same rng, so starts from the
    same state."""
    # None never runs out of resets
    limit = jnp.iinfo(jnp.int32).max if max_resets is None else max_resets
    if replicate_rng:
        keys = jnp.broadcast_to(rng, (num_envs,) + rng.shape)
    else:
        keys = jax.random.split(rng, num_envs)
    env_state = reset_fn(keys)
    rewards = jax.eval_shape(lambda s: step_fn(s, policy(s.obs))[1], env_state)
    state = FitnessState(
        env_state=env_state,
        reward=jnp.zeros(rewards.shape, dtype=jnp.float32),
        resets=jnp.zeros(num_envs, dtype=jnp.int32),
    )

    def game_step(state):
        env_state, rewards, done = step_fn(state.env_state, policy(state.env_state.obs))
        # the step that resets the environment still counts
        counted = state.resets <= limit
        reward = state.reward + mask_data(rewards, 0, counted)
        resets = state.resets + (counted & done.astype(bool))
        # past the last counted reset, keep the environment's last state
        env_state = _keep_where(resets <= limit, env_state, state.env_state)
        return FitnessState(env_state=env_state, reward=reward, resets=resets)

    if keep is None:
        state = jax.lax.fori_loop(0, num_steps, lambda _, s: game_step(s), state)
        return state, None
    return _scan_every(
        game_step, state, num_steps, keep_every, lambda s: keep(s.env_state)
    )


def fitness(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    genome: Genome,
    num_steps: int,
    num_episodes: int,
    max_resets: Optional[int],
    record: bool,
    replicate_rng: bool = False,
    **kwargs,
) -> Tuple[jax.Array, Scores]:
    """Scores a population: each genome's total reward, averaged over num_episodes
    episodes from different starts (see play for max_resets), and Scores with
    every episode's total and, if record, a Recording of every genome's first
    episode.

    With replicate_rng, every genome gets the same rng for each episode, so plays
    from the same num_episodes starts: one can't outscore another just by drawing
    easier ones."""

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
        replicate_rng=replicate_rng,
    )
    reward, frames = play(rng, keep=(lambda s: s) if record else None)
    returns = reward[None]
    if num_episodes > 1:
        # the rest of the episodes only count towards fitness
        rngs = jax.random.split(jax.random.fold_in(rng, 1), num_episodes - 1)
        rewards, _ = jax.vmap(play)(rngs)
        returns = jnp.concatenate([returns, rewards])
        reward = (reward + rewards.sum(axis=0)) / num_episodes
    if not record:
        return reward, Scores(returns)
    players = jnp.arange(genome.batch_size)[:, None]
    return reward, Scores(returns, Recording(_episode_major(frames), players))


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
    different starts (see play for max_resets), and the environment state over the
    first frames_len steps of the first episode, every frames_every steps, to show
    it."""

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


class Environment(Protocol):
    """An environment the genomes of a batch each play on their own, like evojax's.

    reset takes a key per episode, (episodes, 2), and returns the state of every
    episode, whose obs is (episodes, obs_size); step takes actions of
    (episodes, act_size) and returns the next state, rewards of (episodes,) and
    whether each episode is done, (episodes,)."""

    def reset(self, key: jax.Array, /) -> Any: ...

    def step(
        self, state: Any, action: jax.Array, /
    ) -> Tuple[Any, jax.Array, jax.Array]: ...


class MultiPlayerEnvironment(Protocol):
    """An environment where num_players genomes play each game together.

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


def play_seated(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    seated: Genome,
    num_games: int,
    num_steps: int,
    max_resets: Optional[int],
    record: bool = False,
    record_every: int = 1,
    record_fn: Optional[Callable] = None,
    stats_fn: Optional[Callable] = None,
    **kwargs,
) -> Tuple[jax.Array, Any, Dict[str, jax.Array]]:
    """Plays genomes against each other in num_games games of a multi-player
    environment: seated holds the genome in each seat, laid out as (games, seats)
    and flattened. Returns each seat's total reward over num_steps (see play for
    max_resets); if record, every game's environment state every record_every
    steps (or what record_fn keeps of it), with leading (game, step) axes; and
    stats_fn of the games' final environment state, stats by name of each seat,
    (games, seats) (none without stats_fn)."""

    def policy(obs):
        return apply_seats(seated, obs, **kwargs)

    state, frames = play(
        step_fn,
        reset_fn,
        rng,
        policy,
        num_envs=num_games,
        num_steps=num_steps,
        max_resets=max_resets,
        keep=(record_fn or (lambda env_state: env_state)) if record else None,
        keep_every=record_every,
    )
    stats = {} if stats_fn is None else stats_fn(state.env_state)
    frames = frames if frames is None else _episode_major(frames)
    return state.reward, frames, stats


def _mean_per_genome(ids: jax.Array, values: jax.Array, num_genomes: int) -> jax.Array:
    """Each genome's mean of the finite values of the seats it played, NaN for a
    genome with none: a stat that doesn't apply to some games (a lap time without a
    lap) is the mean of the games it does."""
    finite = jnp.isfinite(values)
    zeros = jnp.zeros(num_genomes)
    return zeros.at[ids].add(jnp.where(finite, values, 0.0)) / zeros.at[ids].add(finite)


def fitness_mp(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    genome: Genome,
    num_players: int,
    steps_per_round: int,
    num_rounds: int,
    max_resets: Optional[int],
    record: bool = False,
    record_every: int = 1,
    record_fn: Optional[Callable] = None,
    parallel_rounds: int = 1,
    **kwargs,
):
    """Each round shuffles the population into games of num_players. When the
    population doesn't divide evenly, the last game is filled with genomes that
    also play elsewhere, so fitness is the mean reward per game played (see play
    for max_resets). Also returns Scores with each genome's mean reward in each
    round (its one game, unless it filled a seat too) and, if record, a Recording
    of the first round's games.

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
        rewards, frames, _ = play_seated(
            step_fn,
            reset_fn,
            rng_game,
            seated,
            num_games,
            steps_per_round,
            max_resets,
            record=record,
            record_every=record_every,
            record_fn=record_fn,
            **kwargs,
        )
        return ids, rewards.ravel(), frames

    zeros = jnp.zeros(genome.batch_size)

    def tally(ids, rewards):
        """Each genome's total reward and games played in one round."""
        return zeros.at[ids].add(rewards), zeros.at[ids].add(1)

    def some_rounds(carry, rngs):
        """carry plus some rounds' totals and games, and each round's returns."""
        ids, rewards, _ = jax.vmap(one_round)(rngs)
        totals, counts = jax.vmap(tally)(ids, rewards)
        carry = (carry[0] + totals.sum(axis=0), carry[1] + counts.sum(axis=0))
        return carry, totals / counts

    carry, rngs = (zeros, zeros), jax.random.split(rng, num_rounds)
    returns, recording = [], None
    if record:
        # one round's games are enough to show: every genome plays in one of them
        ids, rewards, frames = one_round(rngs[0], record=True)
        totals, counts = tally(ids, rewards)
        carry = (carry[0] + totals, carry[1] + counts)
        returns.append((totals / counts)[None])
        recording = Recording(frames, ids.reshape(num_games, num_players))
        rngs = rngs[1:]
    # the rest in batches of parallel_rounds, then any left over
    batch = max(1, min(parallel_rounds, len(rngs)))
    batches = len(rngs) // batch
    if batches:
        stacked = rngs[: batches * batch].reshape((batches, batch) + rngs.shape[1:])
        carry, batched = jax.lax.scan(some_rounds, carry, xs=stacked)
        returns.append(batched.reshape(-1, genome.batch_size))
    if len(rngs) % batch:
        carry, rest = some_rounds(carry, rngs[batches * batch :])
        returns.append(rest)
    totals, counts = carry
    return totals / counts, Scores(jnp.concatenate(returns), recording)


def fitness_mp_field(
    step_fn: Callable,
    reset_fn: Callable,
    rng: jax.Array,
    genome: Genome,
    num_games: int,
    num_steps: int,
    max_resets: Optional[int],
    record_every: int = 1,
    record_fn: Optional[Callable] = None,
    stats_fn: Optional[Callable] = None,
    **kwargs,
) -> Tuple[jax.Array, Any, Dict[str, jax.Array]]:
    """The batch of genomes all playing each other, num_games times: game g seats
    them g seats along from game 0, so that each plays from every seat as often
    (see play for max_resets). Returns each genome's mean reward per game, game
    0's environment state every record_every steps (or what record_fn keeps of
    it), genome i in seat i, and each genome's mean of each stat stats_fn finds in
    the games (see play_seated)."""
    num_players = genome.batch_size
    games, seats = np.arange(num_games)[:, None], np.arange(num_players)[None]
    ids = (seats + games) % num_players
    seated = jax.tree.map(lambda x: x[ids.ravel()], genome)
    rewards, frames, stats = play_seated(
        step_fn,
        reset_fn,
        rng,
        seated,
        num_games,
        num_steps,
        max_resets,
        record=True,
        record_every=record_every,
        record_fn=record_fn,
        stats_fn=stats_fn,
        **kwargs,
    )
    totals = jnp.zeros(num_players).at[ids.ravel()].add(rewards.ravel())
    stats = {
        k: _mean_per_genome(ids.ravel(), v.ravel(), num_players)
        for k, v in stats.items()
    }
    return totals / num_games, jax.tree.map(lambda x: x[0], frames), stats


def make_fitness_fn(
    env: Environment,
    num_steps: int,
    num_episodes: int = 1,
    max_resets: Optional[int] = 0,
    record: bool = False,
    replicate_rng: bool = False,
) -> Callable:
    """A training fitness: scores every genome of a population on env (see
    fitness)."""
    assert num_steps > 0, "num_steps must be greater than 0"
    assert num_episodes > 0, "num_episodes must be greater than 0"
    assert max_resets is None or max_resets >= 0, "max_resets must be at least 0"
    return partial(
        fitness,
        step_fn=env.step,
        reset_fn=env.reset,
        num_steps=num_steps,
        num_episodes=num_episodes,
        max_resets=max_resets,
        record=record,
        replicate_rng=replicate_rng,
    )


def make_test_fn(
    env: Environment,
    num_steps: int,
    num_episodes: int,
    max_resets: Optional[int] = 0,
    frames_len: Optional[int] = None,
    frames_every: int = 1,
) -> Callable:
    """A test of one genome, e.g. the best of a generation, on env (see test)."""
    assert num_steps > 0, "num_steps must be greater than 0"
    assert num_episodes > 0, "num_episodes must be greater than 0"
    assert max_resets is None or max_resets >= 0, "max_resets must be at least 0"
    if frames_len is None:
        frames_len = num_steps
    return partial(
        test,
        step_fn=env.step,
        reset_fn=env.reset,
        num_steps=num_steps,
        num_episodes=num_episodes,
        max_resets=max_resets,
        frames_len=min(frames_len, num_steps),
        frames_every=frames_every,
    )


def make_mp_fitness_fn(
    env: MultiPlayerEnvironment,
    steps_per_round: int,
    num_rounds: int,
    max_resets: Optional[int] = 0,
    record: bool = False,
    record_every: int = 1,
    record_fn: Optional[Callable] = None,
    parallel_rounds: int = 1,
) -> Callable:
    assert num_rounds > 0, "num_rounds must be greater than 0"
    assert steps_per_round > 0, "steps_per_round must be greater than 0"
    assert max_resets is None or max_resets >= 0, "max_resets must be at least 0"
    assert parallel_rounds > 0, "parallel_rounds must be greater than 0"
    return partial(
        fitness_mp,
        step_fn=env.step_mp,
        reset_fn=env.reset,
        num_players=env.num_players,
        steps_per_round=steps_per_round,
        num_rounds=num_rounds,
        max_resets=max_resets,
        record=record,
        record_every=record_every,
        record_fn=record_fn,
        parallel_rounds=parallel_rounds,
    )


def make_mp_field_fn(
    env: MultiPlayerEnvironment,
    num_games: int,
    num_steps: int,
    max_resets: Optional[int] = 0,
    record_every: int = 1,
    record_fn: Optional[Callable] = None,
    stats_fn: Optional[Callable] = None,
) -> Callable:
    """A game of a batch of genomes against each other, for make_field_test (see
    fitness_mp_field)."""
    assert num_games > 0, "num_games must be greater than 0"
    assert num_steps > 0, "num_steps must be greater than 0"
    assert max_resets is None or max_resets >= 0, "max_resets must be at least 0"
    return partial(
        fitness_mp_field,
        step_fn=env.step_mp,
        reset_fn=env.reset,
        num_games=num_games,
        num_steps=num_steps,
        max_resets=max_resets,
        record_every=record_every,
        record_fn=record_fn,
        stats_fn=stats_fn,
    )
