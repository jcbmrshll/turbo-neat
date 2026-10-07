"""Race F1TENTH cars by self-play.

Genomes are shuffled into races of --cars cars on one of F1TENTH's tracks, and each
scores the distance it covers before the clock runs out. A car that touches a wall
or another car is out of the race, and stays on the track as a wreck for the others
to avoid. Each car sees a lidar sweep, its own speed, steering and heading, and the
cars around it.

The best genome of each generation races the reigning champion, the two taking
alternate places on the grid, and each new champion races the field of pure
pursuit bots that follow the track's raceline. The score against the baseline is
the distance the champion covers in that race, in metres; the bots' own distance
is printed at the start for reference.

    uv run examples/f1tenth.py --generations 100
    uv run examples/f1tenth.py --cars 6 --monitor

Start the monitor first with `uv run turbo-neat-monitor`.
"""

import argparse
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from f1tenth_gym_jax import make
from f1tenth_gym_jax.envs.collision_models import collision, get_vertices
from f1tenth_gym_jax.envs.utils import State

import neat.activations as act
from monitor import DEFAULT_URL, Episode, Monitor
from neat.config import GenomeConfig, MutationConfig, NEATConfig, SelectionConfig
from neat.fitness import make_fitness_fn, make_mp_fitness_fn, make_mp_h2h_fitness_fn
from neat.neat import NEAT
from neat.species import make_remove_last_if_stagnant_and_full_stagnation_fn

# The env's own lidar and wall check are too slow for a population: its lidar ray
# marches in a while loop, which under vmap runs every ray for as long as the
# slowest, and its wall check tests every car against every wall pixel. Both are
# done here against the map's distance transform instead.

# the lidar sweeps 270 degrees ahead, sphere tracing each ray for a fixed number of
# steps, which is enough to reach a wall in all but the most glancing directions
LIDAR_FOV, LIDAR_RANGE, LIDAR_STEPS = 1.5 * np.pi, 10.0, 24
# walls are checked at this many poses along each step, so a fast car can't jump one
COLLISION_SUBSTEPS = 4
# the starting grid: metres between places, and either side of the centerline
GRID_SPACING, GRID_OFFSET = 1.5, 0.4
# a car this far off the centerline has left the track (Spielberg is 2.2m wide)
MAX_LATERAL_ERROR = 1.5


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RaceState:
    env_state: State
    obs: jax.Array
    key: jax.Array


class F1TenthRace:
    """num_players cars racing on one track, the actions of each being a steering
    angle and a target speed in [-1, 1]."""

    def __init__(
        self,
        num_cars: int,
        map_name: str = "Spielberg",
        num_beams: int = 16,
        timestep_ratio: int = 5,
        max_speed: float = 10.0,
    ):
        self.num_players = num_cars
        self.max_speed = max_speed
        self.env = env = make(
            f"{map_name}_{num_cars}_noscan_nocollision_progress"
            f"_velocity+steeringangle_{timestep_ratio}_v0",
            observe_others=False,
            # the race is timed, so it never ends on laps
            max_num_laps=1_000_000,
        )
        self.obs_size = num_beams + 6 + 4 * (num_cars - 1)
        self.act_size = 2
        params = env.params
        distance_transform = jnp.asarray(env.distance_transform)

        def wall_distance(xy):
            col = ((xy[..., 0] - env.orig_x) / env.resolution).astype(jnp.int32)
            row = ((xy[..., 1] - env.orig_y) / env.resolution).astype(jnp.int32)
            on_map = (row >= 0) & (row < env.height) & (col >= 0) & (col < env.width)
            return jnp.where(on_map, distance_transform[row, col], 0.0)

        beam_angles = jnp.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, num_beams)

        def lidar(poses):
            angles = poses[:, 2:3] + beam_angles
            directions = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)

            def march(_, ranges):
                hits = poses[:, None, :2] + ranges[..., None] * directions
                return ranges + wall_distance(hits)

            ranges = jax.lax.fori_loop(0, LIDAR_STEPS, march, jnp.zeros(angles.shape))
            return jnp.minimum(ranges, LIDAR_RANGE) / LIDAR_RANGE

        def vertices(poses):
            return jax.vmap(
                lambda p: get_vertices(p, length=params.length, width=params.width)
            )(poses)

        def collisions(state: State) -> jax.Array:
            last = state.last_cartesian_states[:, [0, 1, 4]]
            pose = state.cartesian_states[:, [0, 1, 4]]
            fractions = jnp.linspace(0, 1, COLLISION_SUBSTEPS + 1)[1:]
            turn = jnp.angle(jnp.exp(1j * (pose[:, 2] - last[:, 2])))
            sweep = jax.vmap(
                lambda f: vertices(
                    jnp.column_stack(
                        (
                            last[:, :2] + f * (pose[:, :2] - last[:, :2]),
                            last[:, 2] + f * turn,
                        )
                    )
                )
            )(fractions)
            hit_wall = (wall_distance(sweep) < env.resolution / 2).any(axis=(0, 2))
            off_track = jnp.abs(state.frenet_states[:, 1]) > MAX_LATERAL_ERROR
            # every pair of cars, by the separating axis theorem
            corners = vertices(pose)
            i, j = np.triu_indices(num_cars, 1)
            pairs = jax.vmap(collision)(jnp.concatenate((corners[i], corners[j]), -1))
            hit_car = jnp.zeros(num_cars, bool).at[i].max(pairs).at[j].max(pairs)
            return state.collisions | hit_wall | off_track | hit_car

        def observe(state: State) -> jax.Array:
            x, y, steer, speed, yaw, yaw_rate, slip = state.cartesian_states.T
            own = jnp.column_stack(
                (
                    speed / max_speed,
                    steer / params.s_max,
                    yaw_rate / 5.0,
                    slip,
                    state.frenet_states[:, 1] / MAX_LATERAL_ERROR,
                    state.frenet_states[:, 2],
                )
            )
            scan = lidar(state.cartesian_states[:, [0, 1, 4]])
            # the other cars in each car's frame, nearest first: whether it is within
            # lidar range, where it is, and how much faster it is going
            dx, dy = x[None, :] - x[:, None], y[None, :] - y[:, None]
            cos, sin = jnp.cos(yaw)[:, None], jnp.sin(yaw)[:, None]
            ahead, left = dx * cos + dy * sin, -dx * sin + dy * cos
            dist = jnp.where(jnp.eye(num_cars, dtype=bool), jnp.inf, jnp.hypot(dx, dy))
            order = jnp.argsort(dist, axis=1)[:, : num_cars - 1]
            near = lambda a: jnp.take_along_axis(a, order, axis=1)
            visible = near(dist) < LIDAR_RANGE
            others = jnp.stack(
                (
                    visible,
                    visible * near(ahead) / LIDAR_RANGE,
                    visible * near(left) / LIDAR_RANGE,
                    visible * near(speed[None, :] - speed[:, None]) / max_speed,
                ),
                axis=-1,
            ).reshape(num_cars, -1)
            return jnp.concatenate((scan, own, others), axis=1)

        def reset_fn(key: jax.Array) -> RaceState:
            next_key, key = jax.random.split(key)
            _, env_state = env.reset(key)
            # the env lines the cars up nose to tail; start from a staggered grid
            # instead, from the place on the track it picked at random
            places = jnp.arange(num_cars)
            frenet = jnp.column_stack(
                (
                    env_state.frenet_states[0, 0] - GRID_SPACING * places,
                    jnp.where(places % 2 == 0, 1.0, -1.0) * GRID_OFFSET,
                    jnp.zeros(num_cars),
                )
            )
            poses = env.track.vmap_frenet_to_cartesian_jax(frenet)
            cartesian = env_state.cartesian_states.at[:, [0, 1, 4]].set(poses)
            env_state = env_state.replace(
                cartesian_states=cartesian,
                last_cartesian_states=cartesian,
                frenet_states=frenet,
                last_frenet_states=frenet,
            )
            return RaceState(env_state=env_state, obs=observe(env_state), key=next_key)

        def step_fn(state: RaceState, actions: jax.Array):
            next_key, key = jax.random.split(state.key)
            actions = jnp.clip(actions, -1.0, 1.0)
            controls = jnp.column_stack(
                (actions[:, 0] * params.s_max, (actions[:, 1] + 1) / 2 * max_speed)
            )
            _, env_state, rewards, _, _ = env.step_env(
                key, state.env_state, dict(zip(env.agents, controls))
            )
            # the car that crashes this step keeps the progress it made, and is
            # frozen in place from the next
            env_state = env_state.replace(collisions=collisions(env_state))
            rewards = jnp.stack([rewards[a] for a in env.agents])
            state = RaceState(env_state=env_state, obs=observe(env_state), key=next_key)
            return state, rewards, env_state.collisions.all()

        # one race, unbatched
        self.reset_one, self.step_one = reset_fn, step_fn
        self._reset_fn = jax.jit(jax.vmap(reset_fn))
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jax.Array) -> RaceState:
        return self._reset_fn(key)

    def step_mp(
        self, state: RaceState, actions: jax.Array
    ) -> Tuple[RaceState, jax.Array, jax.Array]:
        return self._step_fn(state, actions)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BotRaceState:
    race: RaceState
    seat: jax.Array
    obs: jax.Array


class RaceAgainstBots:
    """One car of an F1TenthRace driven by the policy and the rest by bots that
    pure-pursue the centerline, braking for corners. The policy's car starts from a
    random place on the grid."""

    def __init__(
        self,
        race: F1TenthRace,
        top_speed: float = 5.0,
        lateral_accel: float = 3.0,
        preview: float = 6.0,
    ):
        self.race = race
        env, params = race.env, race.env.params
        xs, ys = np.asarray(env.track.centerline.xs), np.asarray(env.track.centerline.ys)
        # the speed each waypoint allows, from the sharpest turn up to preview metres on
        dx, dy = np.gradient(xs), np.gradient(ys)
        curvature = np.abs(dx * np.gradient(dy) - dy * np.gradient(dx)) / np.hypot(dx, dy) ** 3
        spacing = np.hypot(dx, dy).mean()
        window = np.arange(int(preview / spacing))
        ahead_curvature = curvature[(np.arange(len(xs))[:, None] + window) % len(xs)].max(1)
        speeds = np.clip(np.sqrt(lateral_accel / ahead_curvature), 2.0, top_speed)
        waypoints = jnp.asarray(np.column_stack((xs, ys, speeds)))
        wheelbase = params.lf + params.lr

        def pure_pursuit(car):
            x, y, _, speed, yaw = car[:5]
            lookahead = jnp.maximum(1.0, 0.4 * speed)
            dx, dy = waypoints[:, 0] - x, waypoints[:, 1] - y
            ahead = dx * jnp.cos(yaw) + dy * jnp.sin(yaw)
            left = -dx * jnp.sin(yaw) + dy * jnp.cos(yaw)
            # the waypoint in front of the car nearest to the lookahead distance
            gap = jnp.where(ahead > 0, jnp.abs(jnp.hypot(dx, dy) - lookahead), jnp.inf)
            i = jnp.argmin(gap)
            steer = jnp.arctan(wheelbase * 2.0 * left[i] / lookahead**2)
            return steer, waypoints[i, 2]

        def drive(cars):
            """Actions for every car of a race, each keeping its distance from any
            car just ahead of it."""
            steer, speed = jax.vmap(pure_pursuit)(cars)
            x, y, yaw, car_speed = cars[:, 0], cars[:, 1], cars[:, 4], cars[:, 3]
            dx, dy = x[None, :] - x[:, None], y[None, :] - y[:, None]
            ahead = dx * jnp.cos(yaw)[:, None] + dy * jnp.sin(yaw)[:, None]
            left = -dx * jnp.sin(yaw)[:, None] + dy * jnp.cos(yaw)[:, None]
            in_lane = (ahead > 0) & (jnp.abs(left) < params.width * 2)
            # match the speed of the car ahead from 2m back, and stop at 1m
            follow = car_speed[None, :] * jnp.clip(ahead - 1.0, 0.0, 1.0)
            speed = jnp.minimum(speed, jnp.where(in_lane, follow, jnp.inf).min(axis=1))
            return jnp.column_stack(
                (steer / params.s_max, 2 * speed / race.max_speed - 1)
            )

        def reset_fn(key: jax.Array) -> BotRaceState:
            key_race, key_seat = jax.random.split(key)
            race_state = race.reset_one(key_race)
            seat = jax.random.randint(key_seat, (), 0, race.num_players)
            return BotRaceState(race=race_state, seat=seat, obs=race_state.obs[seat])

        def step_fn(state: BotRaceState, action: jax.Array):
            bots = drive(state.race.env_state.cartesian_states)
            actions = bots.at[state.seat].set(action)
            race_state, rewards, _ = race.step_one(state.race, actions)
            crashed = race_state.env_state.collisions[state.seat]
            state = BotRaceState(
                race=race_state, seat=state.seat, obs=race_state.obs[state.seat]
            )
            return state, rewards[state.seat], crashed

        self._reset_fn = jax.jit(jax.vmap(reset_fn))
        self._step_fn = jax.jit(jax.vmap(step_fn))
        self.drive = drive

    def reset(self, key: jax.Array) -> BotRaceState:
        return self._reset_fn(key)

    def step(
        self, state: BotRaceState, action: jax.Array
    ) -> Tuple[BotRaceState, jax.Array, jax.Array]:
        return self._step_fn(state, action)


def bot_distance(race: F1TenthRace, bots: RaceAgainstBots, num_steps: int) -> float:
    """Mean distance a car covers in num_steps when every car is a bot."""

    @jax.jit
    def run(key):
        state = race.reset(jax.random.split(key, 64))

        def step(_, carry):
            state, total = carry
            actions = jax.vmap(bots.drive)(
                state.env_state.cartesian_states
            )
            state, rewards, _ = race.step_mp(state, actions)
            return state, total + rewards

        _, total = jax.lax.fori_loop(
            0, num_steps, step, (state, jnp.zeros((64, race.num_players)))
        )
        return total.mean()

    return float(run(jax.random.PRNGKey(0)))


def make_config(input_size: int, output_size: int, num_cars: int) -> NEATConfig:
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

    input_labels = [f"lidar_{i}" for i in range(input_size - 6 - 4 * (num_cars - 1))]
    input_labels += ["speed", "steer", "yaw_rate", "slip", "lateral_error", "heading_error"]
    for i in range(num_cars - 1):
        input_labels += [f"car_{i}_{x}" for x in ("visible", "ahead", "left", "speed")]

    genome_config = GenomeConfig(
        input_size=input_size,
        output_size=output_size,
        initial_capacity=100,
        init_weight_mean=0.0,
        init_weight_std=1.0,
        activation_fns=[act.iden, act.sigmoid, act.tanh, act.relu],
        # identity on every input
        input_activation_ids=[0 for _ in range(input_size)],
        # tanh on every output
        output_activation_ids=[2 for _ in range(output_size)],
        input_labels=input_labels,
        output_labels=["steer", "speed"],
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


def make_episode_fn(race: F1TenthRace, pixel_size: float = 0.1):
    """The champion's race against the bots as raw car poses, for the monitor to
    draw, with the track's walls as a bitmap of pixel_size metres."""
    env = race.env
    occupied = np.asarray(env.track.occ_map) == 0
    # crop to the track, with a margin, and max-pool down so that thin walls survive
    k = max(1, round(pixel_size / env.resolution))
    xs, ys = np.asarray(env.track.centerline.xs), np.asarray(env.track.centerline.ys)
    margin = 3.0
    col0, col1 = ((np.array([xs.min(), xs.max()]) - env.orig_x + [-margin, margin]) / env.resolution).astype(int)
    row0, row1 = ((np.array([ys.min(), ys.max()]) - env.orig_y + [-margin, margin]) / env.resolution).astype(int)
    col0, row0 = max(col0, 0), max(row0, 0)
    occupied = occupied[row0:row1, col0:col1]
    h, w = occupied.shape[0] // k * k, occupied.shape[1] // k * k
    walls = occupied[:h, :w].reshape(h // k, k, w // k, k).any(axis=(1, 3))
    origin = np.array([env.orig_x + col0 * env.resolution, env.orig_y + row0 * env.resolution])

    def episode_fn(state: BotRaceState):
        env_state = state.race.env_state
        return Episode(
            "f1tenth",
            walls=walls,
            origin=origin,
            pixel_size=np.array(k * env.resolution),
            car_size=np.array([env.params.length, env.params.width]),
            poses=env_state.cartesian_states[..., [0, 1, 4]],
            crashed=env_state.collisions,
            champion=state.seat[0],
        )

    return episode_fn


def main():
    parser = argparse.ArgumentParser(description="Race F1TENTH cars by self-play.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--cars", type=int, default=4, help="cars in each race")
    parser.add_argument("--map", default="Spielberg", help="an F1TENTH track")
    parser.add_argument(
        "--race-steps",
        type=int,
        default=400,
        help="length of a race, in control steps of 0.05s",
    )
    parser.add_argument(
        "--rounds", type=int, default=2, help="races each genome runs per generation"
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

    race = F1TenthRace(args.cars, map_name=args.map)
    bots = RaceAgainstBots(race)
    print(
        f"bots cover {bot_distance(race, bots, args.race_steps):.1f}m"
        f" in a {args.race_steps}-step race"
    )

    neat = NEAT(
        config=make_config(race.obs_size, race.act_size, args.cars),
        fitness_fn=make_mp_fitness_fn(
            race, steps_per_round=args.race_steps, num_rounds=args.rounds
        ),
        # a lone car races the clock, so its best fitness so far is a fair test
        h2h_test_fn=(
            make_mp_h2h_fitness_fn(race, num_steps=args.race_steps)
            if args.cars > 1
            else None
        ),
        baseline_test_fn=make_fitness_fn(bots, num_steps=args.race_steps),
        monitor=Monitor(args.monitor, project="f1tenth") if args.monitor else None,
    )
    neat.run(
        seed=args.seed,
        num_generations=args.generations,
        episode_fn=make_episode_fn(race),
    )


if __name__ == "__main__":
    main()
