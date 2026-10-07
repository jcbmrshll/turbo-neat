"""Race F1TENTH cars by self-play.

Genomes are shuffled into races of --cars cars on one of F1TENTH's tracks, and each
scores the distance it covers before the clock runs out. A car that touches a wall
or another car is out of the race, and stays on the track as a wreck for the others
to avoid. Each car sees a lidar sweep, its own speed, steering and heading, and the
cars around it.

The best genome of each generation races the reigning champion, the two taking
alternate places on the grid. To see how good the best are, every --field-every
generations the --cars fittest race each other (--field-races races, starting from
every place on the grid in turn), and the monitor logs the field's mean and best
distance, in metres, with one of their races. With --test bots, each new champion races pure
pursuit bots that follow the track's centerline instead, scoring the distance it
covers.

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
from f1tenth_gym_jax.envs.track.utils import find_track_dir
from f1tenth_gym_jax.envs.utils import State

import neat.activations as act
from monitor import DEFAULT_URL, Episode, Monitor
from neat.config import GenomeConfig, MutationConfig, NEATConfig, SelectionConfig
from neat.fitness import (
    make_mp_field_fn,
    make_mp_fitness_fn,
    make_mp_h2h_fitness_fn,
    make_test_fn,
)
from neat.neat import NEAT
from neat.species import make_remove_last_if_stagnant_and_full_stagnation_fn

# The env's own lidar and wall check are too slow for a population: its lidar ray
# marches in a while loop, which under vmap runs every ray for as long as the
# slowest, and its wall check tests every car against every wall pixel. Both are
# done here against the map's distance transform instead.

# a car senses walls (by lidar) and other cars as far as it travels in this many
# seconds at top speed, and at least this many metres
SENSING_TIME, MIN_SENSING_RANGE = 1.0, 10.0
# the lidar sweeps 270 degrees ahead, sphere tracing each ray for a fixed number of
# steps per metre of range, which is enough to reach a wall in all but the most
# glancing directions
LIDAR_FOV, LIDAR_STEPS_PER_METRE = 1.5 * np.pi, 2.4
# walls are checked at poses this many metres apart along each step at top speed,
# so a fast car can't jump one (Spielberg's walls are about 0.1-0.2m thick)
WALL_CHECK_SPACING = 0.1
# races are recorded for the monitor at this many frames a second, whatever the
# control rate
RECORD_HZ = 20
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
    # each car's reward so far: metres along the track
    rewards: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RaceFrame:
    """What the monitor draws of a race at a moment: each car's x, y, steering
    angle, speed and heading, whether it has crashed, and its reward so far."""

    cars: jax.Array
    crashed: jax.Array
    rewards: jax.Array


class F1TenthRace:
    """num_players cars racing on one track, the actions of each being a steering
    angle and a target speed in [-1, 1]."""

    def __init__(
        self,
        num_cars: int,
        map_name: str = "Spielberg",
        num_beams: int = 16,
        timestep_ratio: int = 1,
        max_speed: float = 20.0,
        grid_spacing: float = GRID_SPACING,
    ):
        self.num_players = num_cars
        self.max_speed = max_speed
        self.env = env = make(
            f"{map_name}_{num_cars}_noscan_nocollision_progress"
            f"_velocity+steeringangle_{timestep_ratio}_v0",
            observe_others=False,
            # the env's own lidar isn't used, but it keeps a buffer of its beams in
            # every state: keep it as small as it goes
            num_beams=2,
            # the race is timed, so it never ends on laps
            max_num_laps=1_000_000,
        )
        self.obs_size = num_beams + 6 + 4 * (num_cars - 1)
        self.act_size = 2
        params = env.params
        # seconds per control step, and control steps per recorded frame
        self.dt = params.timestep * params.timestep_ratio
        self.record_every = max(1, round(1 / (RECORD_HZ * self.dt)))
        distance_transform = jnp.asarray(env.distance_transform)
        self.sensing_range = sensing = max(MIN_SENSING_RANGE, SENSING_TIME * max_speed)
        lidar_steps = int(np.ceil(LIDAR_STEPS_PER_METRE * sensing))
        step_length = max_speed * params.timestep * params.timestep_ratio
        wall_checks = max(4, int(np.ceil(step_length / WALL_CHECK_SPACING)))

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

            ranges = jax.lax.fori_loop(0, lidar_steps, march, jnp.zeros(angles.shape))
            return jnp.minimum(ranges, sensing) / sensing

        def vertices(poses):
            return jax.vmap(
                lambda p: get_vertices(p, length=params.length, width=params.width)
            )(poses)

        def settle(state: State) -> State:
            """Crashes: a car that hits a wall during this step goes back to the last
            place along it that was clear and stops there, and any car touching a
            wall or another car, or off the track, is out of the race."""
            last = state.last_cartesian_states[:, [0, 1, 4]]
            pose = state.cartesian_states[:, [0, 1, 4]]
            fractions = jnp.linspace(0, 1, wall_checks + 1)

            def along(f):
                turn = jnp.angle(jnp.exp(1j * (pose[:, 2] - last[:, 2])))
                return jnp.column_stack(
                    (
                        last[:, :2] + f * (pose[:, :2] - last[:, :2]),
                        last[:, 2] + f * turn,
                    )
                )

            # a corner within a pixel of a wall touches it: some walls are only a pixel
            # or two thick, which a car's corners could otherwise step across
            sweep = jax.vmap(lambda f: vertices(along(f)))(fractions[1:])
            touching = (wall_distance(sweep) <= env.resolution).any(axis=-1)
            hit_wall = touching.any(axis=0)
            clear = fractions[jnp.argmax(touching, axis=0)]
            back = jax.vmap(lambda f, i: along(f)[i])(clear, jnp.arange(num_cars))
            crashing = hit_wall & ~state.collisions
            cars = state.cartesian_states
            stopped = cars.at[:, [0, 1, 4]].set(back).at[:, 3].set(0.0)
            if cars.shape[1] > 5:
                stopped = stopped.at[:, 5:].set(0.0)  # no yaw rate or slip
            cars = jnp.where(crashing[:, None], stopped, cars)
            frenet = jnp.where(
                crashing[:, None],
                env.track.vmap_cartesian_to_frenet_jax(cars[:, [0, 1, 4]]),
                state.frenet_states,
            )
            state = state.replace(cartesian_states=cars, frenet_states=frenet)
            off_track = jnp.abs(frenet[:, 1]) > MAX_LATERAL_ERROR
            # every pair of cars, by the separating axis theorem
            corners = vertices(cars[:, [0, 1, 4]])
            i, j = np.triu_indices(num_cars, 1)
            pairs = jax.vmap(collision)(jnp.concatenate((corners[i], corners[j]), -1))
            hit_car = jnp.zeros(num_cars, bool).at[i].max(pairs).at[j].max(pairs)
            return state.replace(
                collisions=state.collisions | hit_wall | off_track | hit_car
            )

        def progress(state: State) -> jax.Array:
            """Metres each car advanced along the track this step, as the env's own
            progress reward, but from where crashes left the cars."""
            gain = jnp.mod(state.frenet_states[:, 0], env.track_length) - jnp.mod(
                state.last_frenet_states[:, 0], env.track_length
            )
            # across the start line, one way or the other
            half = env.track_length / 2
            return jnp.where(
                gain > half,
                gain - env.track_length,
                jnp.where(gain < -half, gain + env.track_length, gain),
            )

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

            def near(a):
                return jnp.take_along_axis(a, order, axis=1)

            visible = near(dist) < sensing
            others = jnp.stack(
                (
                    visible,
                    visible * near(ahead) / sensing,
                    visible * near(left) / sensing,
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
                    env_state.frenet_states[0, 0] - grid_spacing * places,
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
            return RaceState(
                env_state=env_state,
                obs=observe(env_state),
                key=next_key,
                rewards=jnp.zeros(num_cars),
            )

        def step_fn(state: RaceState, actions: jax.Array):
            next_key, key = jax.random.split(state.key)
            actions = jnp.clip(actions, -1.0, 1.0)
            controls = jnp.column_stack(
                (actions[:, 0] * params.s_max, (actions[:, 1] + 1) / 2 * max_speed)
            )
            _, env_state, _, _, _ = env.step_env(
                key, state.env_state, dict(zip(env.agents, controls))
            )
            # the car that crashes this step keeps the progress it made up to the
            # crash, and the env freezes it in place from the next
            env_state = settle(env_state)
            rewards = progress(env_state)
            state = RaceState(
                env_state=env_state,
                obs=observe(env_state),
                key=next_key,
                rewards=state.rewards + rewards,
            )
            return state, rewards, env_state.collisions.all()

        # one race, unbatched
        self.reset_one, self.step_one = reset_fn, step_fn
        self._reset_fn = jax.jit(jax.vmap(reset_fn))
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jax.Array) -> RaceState:
        return self._reset_fn(key)

    @staticmethod
    def record_frame(state: RaceState) -> RaceFrame:
        """What a recording keeps of a race's state, for the monitor."""
        cars = state.env_state.cartesian_states[..., :5]
        return RaceFrame(
            cars=cars, crashed=state.env_state.collisions, rewards=state.rewards
        )

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
        xs, ys = (
            np.asarray(env.track.centerline.xs),
            np.asarray(env.track.centerline.ys),
        )
        # the speed each waypoint allows, from the sharpest turn up to preview metres on
        dx, dy = np.gradient(xs), np.gradient(ys)
        curvature = (
            np.abs(dx * np.gradient(dy) - dy * np.gradient(dx)) / np.hypot(dx, dy) ** 3
        )
        spacing = np.hypot(dx, dy).mean()
        window = np.arange(int(preview / spacing))
        ahead_curvature = curvature[
            (np.arange(len(xs))[:, None] + window) % len(xs)
        ].max(1)
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

    def reset(self, key: jax.Array) -> BotRaceState:
        return self._reset_fn(key)

    def step(
        self, state: BotRaceState, action: jax.Array
    ) -> Tuple[BotRaceState, jax.Array, jax.Array]:
        return self._step_fn(state, action)


def make_config(
    input_size: int, output_size: int, num_cars: int, population_size: int = 1000
) -> NEATConfig:
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
    input_labels += [
        "speed",
        "steer",
        "yaw_rate",
        "slip",
        "lateral_error",
        "heading_error",
    ]
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
        population_size=population_size,
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


def track_edges(race: F1TenthRace) -> np.ndarray:
    """The track's two walls, (2, points, 2): its centerline offset to either side
    by the widths the map records alongside it."""
    track_dir = find_track_dir(race.env.params.map_name)
    # columns x, y, width to the right, width to the left
    x, y, right, left = np.loadtxt(
        track_dir / f"{track_dir.stem}_centerline.csv", delimiter=","
    ).T
    center = np.column_stack((x, y))
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return np.stack((center + left[:, None] * normal, center - right[:, None] * normal))


def make_episode_fn(race: F1TenthRace):
    """Races as raw car poses, speeds and steering, for the monitor to draw, with
    the track's walls. Packs the generation's fittest racing each other, or the
    champion's race against the bots, and the members' recorded races at once (with
    a leading race axis)."""
    edges = track_edges(race).astype(np.float32)
    car_size = np.array([race.env.params.length, race.env.params.width])
    # the top speed, the steering lock, and the speed lost between recorded frames
    # braking as hard as the car can, to scale the speed, steering and braking by
    full_braking = race.env.params.a_max * race.dt * race.record_every
    limits = np.array([race.max_speed, race.env.params.s_max, full_braking])

    def episode_fn(state):
        against_bots = isinstance(state, BotRaceState)
        if against_bots:
            env_state = state.race.env_state
            cars, crashed = env_state.cartesian_states, env_state.collisions
            rewards = state.race.rewards
        else:
            cars, crashed, rewards = state.cars, state.crashed, state.rewards
        cars = np.asarray(cars)
        poses = cars[..., [0, 1, 4]]
        # every recorded race carries its own copy of the track, as the monitor
        # draws each one on its own
        races = poses.shape[:-3]
        data = dict(
            edges=np.broadcast_to(edges, races + edges.shape),
            car_size=np.broadcast_to(car_size, races + car_size.shape),
            limits=np.broadcast_to(limits, races + limits.shape),
            poses=poses,
            speed=cars[..., 3],
            steer=cars[..., 2],
            crashed=np.asarray(crashed),
            # each car's reward so far, at every frame
            rewards=np.asarray(rewards),
        )
        if against_bots:
            data["champion"] = np.asarray(state.seat[0])
        # the generation's fittest racing each other, as against a race of members
        # (several, with a leading race axis) or of the champion and the bots
        field = not against_bots and races == ()
        return Episode("f1tenth_field" if field else "f1tenth", **data)

    return episode_fn


def main():
    parser = argparse.ArgumentParser(description="Race F1TENTH cars by self-play.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--cars", type=int, default=4, help="cars in each race")
    parser.add_argument("--map", default="Spielberg", help="an F1TENTH track")
    parser.add_argument(
        "--race-seconds", type=float, default=20.0, help="length of a race"
    )
    parser.add_argument(
        "--test-seconds",
        type=float,
        default=None,
        help="length of the champion's race against the bots (the one the monitor "
        "shows), if not --race-seconds",
    )
    parser.add_argument(
        "--rounds", type=int, default=2, help="races each genome runs per generation"
    )
    parser.add_argument(
        "--parallel-rounds",
        type=int,
        default=1,
        help="rounds raced side by side, --population x this many cars at once: "
        "worth raising until that's around 6000-12000 cars (where an RTX 5080 "
        "is fully busy)",
    )
    parser.add_argument(
        "--test",
        choices=["field", "bots"],
        default="field",
        help="each generation's --cars fittest race each other, or each new champion "
        "races the bots",
    )
    parser.add_argument(
        "--field-races",
        type=int,
        default=16,
        help="races the fittest run against each other (or the best against the "
        "bots, with --test bots), each time",
    )
    parser.add_argument(
        "--field-every",
        type=int,
        default=10,
        help="generations between the fittest racing each other",
    )
    parser.add_argument("--population", type=int, default=1000, help="population size")
    parser.add_argument(
        "--max-speed",
        type=float,
        default=20.0,
        help="the cars' top speed in m/s; they sense a second of travel ahead",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=GRID_SPACING,
        help="metres between places on the starting grid",
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

    race = F1TenthRace(
        args.cars,
        map_name=args.map,
        max_speed=args.max_speed,
        grid_spacing=args.grid_spacing,
    )
    race_steps = round(args.race_seconds / race.dt)
    # a race is done when every car is out, but it doesn't start over: the wrecks
    # just stand there, so score straight through and keep the last crash in replays
    max_resets = None
    test_steps = round((args.test_seconds or args.race_seconds) / race.dt)
    tests = {}
    if args.test == "bots":
        tests["baseline_test_fn"] = make_test_fn(
            RaceAgainstBots(race),
            num_steps=test_steps,
            num_episodes=args.field_races,
            frames_every=race.record_every,
        )
    else:
        tests["field_test_fn"] = make_mp_field_fn(
            race,
            args.field_races,
            test_steps,
            max_resets=max_resets,
            record_every=race.record_every,
            record_fn=race.record_frame,
        )
        tests["field_size"] = args.cars
        tests["field_every"] = args.field_every

    neat = NEAT(
        config=make_config(race.obs_size, race.act_size, args.cars, args.population),
        # the first round's races are recorded for the monitor's lineage view
        fitness_fn=make_mp_fitness_fn(
            race,
            steps_per_round=race_steps,
            num_rounds=args.rounds,
            max_resets=max_resets,
            record=args.monitor is not None,
            record_every=race.record_every,
            record_fn=race.record_frame,
            parallel_rounds=args.parallel_rounds,
        ),
        # a lone car races the clock, so its best fitness so far is a fair test
        h2h_test_fn=(
            make_mp_h2h_fitness_fn(race, num_steps=race_steps, max_resets=max_resets)
            if args.cars > 1
            else None
        ),
        monitor=Monitor(args.monitor, project="f1tenth") if args.monitor else None,
        **tests,
    )
    neat.run(
        seed=args.seed,
        num_generations=args.generations,
        episode_fn=make_episode_fn(race),
    )


if __name__ == "__main__":
    main()
