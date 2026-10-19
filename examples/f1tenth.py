"""Race F1TENTH cars by self-play.

Genomes are shuffled into races of --cars cars on one of F1TENTH's tracks, and each
scores the distance it covers before the clock runs out. A car that touches a wall
or another car is out of the race, and stays on the track as a wreck for the others
to avoid. Each car sees a lidar sweep, its own speed, steering and heading, and the
cars around it.

To see how good the best are, every --test-every generations the --cars fittest
race each other (--field-races races, starting from every place on the grid in
turn), and the monitor shows one of their races and logs five stats of each car in
them: its average lap time in seconds, over the laps it completed (a lap being the
track's length from its place on the grid), how many times it passed a car that was
still racing, whether it crashed, and its average and top speed in m/s while it was
still racing. Each is the car's mean over the races (lap time over those it
completed a lap in), and the monitor charts the mean, least and most over the cars
(lap_time, passes, crashes, avg_speed, top_speed). With --test bots, the 10 fittest
each race pure pursuit bots that follow the track's centerline instead, scoring the
distance they cover, and the monitor shows the best one's race.

The defaults race 11 cars, 8 m apart on the grid, 20 races per genome each
generation, in 20-second races, and test the field in 120-second races. Self-play
takes a while to get going: in three runs of them, the cars only learned to race
past the first corner after about 130, 180 and 550 generations. Two of those
fields went on to cover more than a lap of Spielberg (about 343 m) in a test race,
the faster by generation 350. 1000 generations take about 5.5 hours on an RTX
5080.

    uv run examples/f1tenth.py
    uv run examples/f1tenth.py --monitor

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
    make_test_fn,
)
from neat.neat import NEAT
from neat.species import make_relative_fitness_stagnation_fn
from neat.testing import make_field_test, make_top_k_test

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
GRID_SPACING, GRID_OFFSET = 8.0, 0.4
# a car this far off the centerline has left the track (Spielberg is 2.2m wide)
MAX_LATERAL_ERROR = 1.5
# a pass counts once a car is this many car lengths ahead of the other, and it has
# to drop as far behind before it can pass that car again
PASS_MARGIN = 1.0
# seconds over which a car's acceleration noise (with --accel-noise) drifts
ACCEL_NOISE_TIME = 0.5


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RaceState:
    env_state: State
    obs: jax.Array
    key: jax.Array
    # each car's reward so far: metres along the track
    rewards: jax.Array
    # control steps raced
    steps: jax.Array
    # where each car started, in metres along the track, so that start + rewards
    # is how far round it is, counting laps
    start: jax.Array
    # each car's complete laps (a track's length from where it started), and the
    # step it completed the last one on
    laps: jax.Array
    lap_steps: jax.Array
    # how many times each car passed (or lapped) a car still in the race, and
    # (cars, cars) how many laps each car counts as ahead of each other one
    passes: jax.Array
    laps_ahead: jax.Array
    # each car's steps still racing, the sum of its speeds over them, and the
    # fastest of them, in m/s
    racing_steps: jax.Array
    speed_sum: jax.Array
    top_speed: jax.Array
    # whether each car is out from touching another car (racing or a wreck), or
    # from anything else: a wall, or leaving the track
    car_crashes: jax.Array
    wall_crashes: jax.Array
    # each car's acceleration error this step, in m/s^2
    accel_noise: jax.Array
    # each car's score for its place in the race so far, 1 leading to 0 last
    place: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RaceFrame:
    """What the monitor draws of a race at a moment: each car's x, y, steering
    angle, speed and heading, whether it has crashed, and its reward so far; and
    what each car saw, for it to show the networks' activations."""

    cars: jax.Array
    crashed: jax.Array
    rewards: jax.Array
    obs: jax.Array


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
        crash_penalty: float = 0.0,
        grid_spacing_jitter: float = 0.0,
        cars_per_row: int = 1,
        accel_noise: float = 0.0,
        reward: str = "distance",
        crashed_last: bool = False,
        distance_weight: float = 0.0,
        lidar_cars: bool = False,
        lidar_rays: int = 1,
    ):
        assert reward in ("distance", "placement"), f"no reward {reward!r}"
        self.num_players = num_cars
        self.max_speed = max_speed
        self.crashed_last = crashed_last
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
        self.dt = dt = params.timestep * params.timestep_ratio
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
        # each beam reads what its own ray hits; with lidar_cars and lidar_rays > 1, it
        # also sweeps its sector (the angle between beams) with that many more rays,
        # and reads the nearest car any of them hits before a wall, if nearer, so
        # that a car can't hide between two beams (walls still read along the beam)
        sector = LIDAR_FOV / max(num_beams - 1, 1)
        sweep = lidar_rays if lidar_cars and lidar_rays > 1 else 0
        ray_offsets = jnp.concatenate(
            (jnp.zeros(1), ((jnp.arange(sweep) + 0.5) / max(sweep, 1) - 0.5) * sector)
        )
        ray_angles = (beam_angles[:, None] + ray_offsets[None, :]).ravel()
        half_size = jnp.array([params.length / 2, params.width / 2])

        def car_distance(points, poses):
            """Each car's rays' points' distance to the nearest other car's body,
            (cars, rays): to the rectangle around its pose, 0 inside it."""
            offset = points[:, :, None, :] - poses[None, None, :, :2]
            cos, sin = jnp.cos(poses[:, 2]), jnp.sin(poses[:, 2])
            local = jnp.stack(
                (
                    offset[..., 0] * cos + offset[..., 1] * sin,
                    -offset[..., 0] * sin + offset[..., 1] * cos,
                ),
                axis=-1,
            )
            outside = jnp.abs(local) - half_size
            distance = jnp.linalg.norm(
                jnp.maximum(outside, 0.0), axis=-1
            ) + jnp.minimum(outside.max(axis=-1), 0.0)
            # not its own body, which every ray starts inside
            own = jnp.eye(num_cars, dtype=bool)[:, None, :]
            return jnp.where(own, jnp.inf, jnp.maximum(distance, 0.0)).min(axis=-1)

        def lidar(poses):
            angles = poses[:, 2:3] + ray_angles
            directions = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)

            def ends(ranges):
                return poses[:, None, :2] + ranges[..., None] * directions

            def march(_, ranges):
                hits = ends(ranges)
                step = wall_distance(hits)
                if lidar_cars:
                    step = jnp.minimum(step, car_distance(hits, poses))
                return ranges + step

            ranges = jax.lax.fori_loop(0, lidar_steps, march, jnp.zeros(angles.shape))
            ranges = ranges.reshape(num_cars, num_beams, 1 + sweep)
            if sweep:
                # the sweep's rays that stopped at a car, not a wall
                hits = ends(ranges.reshape(num_cars, -1))
                at_car = car_distance(hits, poses) < wall_distance(hits)
                at_car = at_car.reshape(ranges.shape)[..., 1:]
                swept = jnp.where(at_car, ranges[..., 1:], jnp.inf).min(axis=-1)
                ranges = jnp.minimum(ranges[..., 0], swept)
            else:
                ranges = ranges[..., 0]
            return jnp.minimum(ranges, sensing) / sensing

        def vertices(poses):
            return jax.vmap(
                lambda p: get_vertices(p, length=params.length, width=params.width)
            )(poses)

        def settle(state: State) -> Tuple[State, jax.Array]:
            """Crashes: a car that hits a wall during this step goes back to the last
            place along it that was clear and stops there, and any car touching a
            wall or another car, or off the track, is out of the race. Also returns
            which cars are touching another car."""
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
            # a car out of the race, whatever put it out, is a standing wreck: no
            # speed, yaw rate or slip, so the others sense it standing still
            out = state.collisions | hit_wall | off_track | hit_car
            cars = state.cartesian_states
            # columns x, y, steer, speed, yaw, then yaw rate and slip
            column = jnp.arange(cars.shape[1])
            moving = (column == 3) | (column >= 5)
            cars = jnp.where(out[:, None] & moving, 0.0, cars)
            state = state.replace(collisions=out, cartesian_states=cars)
            return state, hit_car

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

        margin = PASS_MARGIN * params.length

        def passes(laps_ahead, position, racing):
            """Each car's laps ahead of each other one at position along the track
            (counting laps), going up a lap once it's margin past it and down once
            it's margin short of it, and how many times each car went up on a car
            still racing: passed it, or lapped it."""
            gap = position[:, None] - position[None, :]
            lap = env.track_length
            up = gap >= (laps_ahead + 1) * lap + margin
            down = gap < laps_ahead * lap - margin
            laps_ahead = laps_ahead + up - down
            return laps_ahead, (up & racing[None, :]).sum(axis=1).astype(jnp.int32)

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

        num_rows = -(-num_cars // cars_per_row)

        def reset_fn(key: jax.Array) -> RaceState:
            next_key, key = jax.random.split(key)
            gap_key, noise_key = jax.random.split(jax.random.fold_in(key, 1))
            _, env_state = env.reset(key)
            # the env lines the cars up nose to tail; start from a staggered grid
            # instead, from the place on the track it picked at random: rows of
            # cars_per_row side by side, each gap between rows grid_spacing_jitter
            # either side of grid_spacing
            places = jnp.arange(num_cars)
            gaps = (
                grid_spacing
                + jax.random.uniform(gap_key, (num_rows - 1,), minval=-1.0, maxval=1.0)
                * grid_spacing_jitter
            )
            rows_behind = jnp.concatenate((jnp.zeros(1), jnp.cumsum(gaps)))
            behind = rows_behind[places // cars_per_row]
            frenet = jnp.column_stack(
                (
                    env_state.frenet_states[0, 0] - behind,
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
            zeros = jnp.zeros(num_cars, jnp.int32)
            return RaceState(
                env_state=env_state,
                obs=observe(env_state),
                key=next_key,
                rewards=jnp.zeros(num_cars),
                steps=jnp.zeros((), jnp.int32),
                start=frenet[:, 0],
                laps=zeros,
                lap_steps=zeros,
                passes=zeros,
                laps_ahead=jnp.floor(
                    (frenet[:, None, 0] - frenet[None, :, 0]) / env.track_length
                ).astype(jnp.int32),
                racing_steps=zeros,
                speed_sum=jnp.zeros(num_cars),
                top_speed=jnp.zeros(num_cars),
                car_crashes=jnp.zeros(num_cars, bool),
                wall_crashes=jnp.zeros(num_cars, bool),
                accel_noise=accel_noise * jax.random.normal(noise_key, (num_cars,)),
                place=jnp.zeros(num_cars),
            )

        # the acceleration noise drifts (an Ornstein-Uhlenbeck process), keeping a
        # standard deviation of accel_noise
        noise_decay = np.exp(-self.dt / ACCEL_NOISE_TIME)
        noise_kick = accel_noise * np.sqrt(1 - noise_decay**2)

        def place_score(position):
            """Each car's score for its place, by how far round the track it is from
            the grid: 1 for the leader down to 0 for the last car."""
            place = jnp.argsort(jnp.argsort(-position))
            return (num_cars - 1 - place) / max(num_cars - 1, 1)

        def step_fn(state: RaceState, actions: jax.Array):
            next_key, key = jax.random.split(state.key)
            noise_key = jax.random.fold_in(key, 1)
            actions = jnp.clip(actions, -1.0, 1.0)
            controls = jnp.column_stack(
                (actions[:, 0] * params.s_max, (actions[:, 1] + 1) / 2 * max_speed)
            )
            _, env_state, _, _, _ = env.step_env(
                key, state.env_state, dict(zip(env.agents, controls))
            )
            # the cars still racing speed up or slow down by their noise, which the
            # speed controller then works against
            noise = noise_decay * state.accel_noise + noise_kick * jax.random.normal(
                noise_key, (num_cars,)
            )
            if accel_noise > 0:
                speed = jnp.clip(
                    env_state.cartesian_states[:, 3] + noise * self.dt,
                    params.v_min,
                    params.v_max,
                )
                speed = jnp.where(
                    env_state.collisions, env_state.cartesian_states[:, 3], speed
                )
                env_state = env_state.replace(
                    cartesian_states=env_state.cartesian_states.at[:, 3].set(speed)
                )
            # the car that crashes this step keeps the progress it made up to the
            # crash, and the env freezes it in place from the next
            env_state, hit_car = settle(env_state)
            distance = progress(env_state)
            total = state.rewards + distance
            # the reward is the distance, or with reward "placement", the change in
            # the car's place score, so that a race's rewards add up to the score of
            # the place it finishes in; less crash_penalty on the step a car crashes
            position = state.start + total
            if crashed_last:
                # every car out of the race places behind every car still in it
                position = jnp.where(env_state.collisions, position - 1e6, position)
            place = place_score(position)
            # placement can add distance_weight per metre, a little for not stopping
            progress_reward = (
                distance
                if reward == "distance"
                else place - state.place + distance_weight * distance
            )
            crashed = env_state.collisions & ~state.env_state.collisions
            rewards = progress_reward - crash_penalty * crashed
            steps = state.steps + 1
            # driving back over the line and on again doesn't count a lap twice
            laps = jnp.maximum(
                state.laps, (total // env.track_length).astype(jnp.int32)
            )
            # a step the car crashed on, or any after, isn't racing
            racing = ~env_state.collisions
            laps_ahead, passed = passes(state.laps_ahead, state.start + total, racing)
            speed = jnp.where(racing, env_state.cartesian_states[:, 3], 0.0)
            state = RaceState(
                env_state=env_state,
                obs=observe(env_state),
                key=next_key,
                rewards=total,
                steps=steps,
                start=state.start,
                laps=laps,
                lap_steps=jnp.where(laps > state.laps, steps, state.lap_steps),
                passes=state.passes + passed,
                laps_ahead=laps_ahead,
                racing_steps=state.racing_steps + racing,
                speed_sum=state.speed_sum + speed,
                top_speed=jnp.maximum(state.top_speed, speed),
                car_crashes=state.car_crashes | (crashed & hit_car),
                wall_crashes=state.wall_crashes | (crashed & ~hit_car),
                accel_noise=noise,
                place=place,
            )
            return state, rewards, env_state.collisions.all()

        def stats(state: RaceState) -> dict:
            """Each car's distance covered in metres, its average lap time in
            seconds over the laps it completed (NaN without one), how many times it
            passed a car still racing (by a car length), whether it crashed, and
            whether into another car or anything else, and its average and
            top speed while it was still racing (NaN if it never was): of a race,
            or a batch of races."""
            raced = state.racing_steps > 0
            return dict(
                distance=state.rewards,
                lap_time=jnp.where(
                    state.laps > 0,
                    state.lap_steps * dt / jnp.maximum(state.laps, 1),
                    jnp.nan,
                ),
                passes=state.passes.astype(jnp.float32),
                crashes=state.env_state.collisions.astype(jnp.float32),
                car_crashes=state.car_crashes.astype(jnp.float32),
                wall_crashes=state.wall_crashes.astype(jnp.float32),
                avg_speed=jnp.where(
                    raced,
                    state.speed_sum / jnp.maximum(state.racing_steps, 1),
                    jnp.nan,
                ),
                top_speed=jnp.where(raced, state.top_speed, jnp.nan),
            )

        # one race, unbatched
        self.reset_one, self.step_one = reset_fn, step_fn
        self.stats = stats
        self._reset_fn = jax.jit(jax.vmap(reset_fn))
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jax.Array) -> RaceState:
        return self._reset_fn(key)

    @staticmethod
    def record_frame(state: RaceState) -> RaceFrame:
        """What a recording keeps of a race's state, for the monitor."""
        cars = state.env_state.cartesian_states[..., :5]
        return RaceFrame(
            cars=cars,
            crashed=state.env_state.collisions,
            rewards=state.rewards,
            obs=state.obs,
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
    input_size: int,
    output_size: int,
    num_cars: int,
    population_size: int = 1000,
    speciation_threshold: float = 2.0,
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
        input_activation=act.iden,
        output_activation=act.tanh,
        input_labels=input_labels,
        output_labels=["steer", "speed"],
        capacity_growth_strategy="linear",
        init_mode="partial",
    )

    maximum_species = 5
    selection_config = SelectionConfig(
        population_size=population_size,
        cutoff_pct=0.1,
        speciation_threshold=speciation_threshold,
        compatibility_coefficients=(1.0, 0.5),
        elitism=0.1,
        # self-play scores are relative, so a species' fitness can stop rising while
        # it still improves: only remove stagnant species that are also below the
        # mean, and only once there are as many species as there can be
        stagnation_fn=make_relative_fitness_stagnation_fn(
            minimum_stagnation=20, full_target=maximum_species
        ),
        maximum_species=maximum_species,
        selection_tournament_size=5,
        fitness_ema_period=5,
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
    best tested genome's race against the bots, and the members' recorded races at
    once (with a leading race axis)."""
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
            # only the policy's car has a network
            obs = np.asarray(state.obs)[..., None, :]
        else:
            cars, crashed, rewards = state.cars, state.crashed, state.rewards
            obs = np.asarray(state.obs)
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
            obs=obs,
        )
        if race.crashed_last and not against_bots:
            # the monitor ranks crashed cars behind every car still racing, as
            # the reward does
            data["crashed_last"] = np.ones(races + (1,), np.float32)
        if against_bots:
            data["policy"] = np.asarray(state.seat[0])
            data["obs_seats"] = np.asarray(state.seat[:1])
        # the generation's fittest racing each other, as against a race of members
        # (several, with a leading race axis) or of the policy and the bots
        field = not against_bots and races == ()
        return Episode("f1tenth_field" if field else "f1tenth", **data)

    return episode_fn


def main():
    parser = argparse.ArgumentParser(description="Race F1TENTH cars by self-play.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--cars", type=int, default=6, help="cars in each race")
    parser.add_argument("--map", default="Spielberg", help="an F1TENTH track")
    parser.add_argument(
        "--race-seconds", type=float, default=60.0, help="length of a race"
    )
    parser.add_argument(
        "--test-seconds",
        type=float,
        default=120.0,
        help="length of the test races (the ones the monitor shows), if not "
        "--race-seconds",
    )
    parser.add_argument(
        "--rounds", type=int, default=7, help="races each genome runs per generation"
    )
    parser.add_argument(
        "--parallel-rounds",
        type=int,
        default=4,
        help="rounds raced side by side, --population x this many cars at once: "
        "worth raising until that's around 6000-12000 cars (where an RTX 5080 "
        "is fully busy)",
    )
    parser.add_argument(
        "--test",
        choices=["field", "bots"],
        default="field",
        help="every --test-every generations, the --cars fittest race each other, or "
        "the 10 fittest each race the bots",
    )
    parser.add_argument(
        "--field-ranks",
        type=int,
        nargs="+",
        default=None,
        help="the field test's cars by their rank in fitness (1 the fittest), one "
        "per car, for a field of mixed strength; if not, the --cars fittest",
    )
    parser.add_argument(
        "--field-races",
        type=int,
        default=16,
        help="races the fittest run against each other (or each against the bots, "
        "with --test bots), each time",
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=10,
        help="generations between tests",
    )
    parser.add_argument("--population", type=int, default=1500, help="population size")
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
        "--cars-per-row",
        type=int,
        choices=(1, 2),
        default=1,
        help="cars side by side in each row of the starting grid: 1 for single "
        "file, 2 for pairs as on an F1 grid; rows are --grid-spacing apart",
    )
    parser.add_argument(
        "--grid-spacing-jitter",
        type=float,
        default=0.0,
        help="metres either side of --grid-spacing each gap on the grid is drawn "
        "from, in training races",
    )
    parser.add_argument(
        "--accel-noise",
        type=float,
        default=0.0,
        help="standard deviation of the noise in each car's acceleration, in m/s^2, "
        f"drifting over {ACCEL_NOISE_TIME} s, in training races",
    )
    parser.add_argument(
        "--test-grid-spacing",
        type=float,
        default=None,
        help="metres between places on the test races' starting grid, if not "
        "--grid-spacing",
    )
    parser.add_argument(
        "--reward",
        choices=("distance", "placement"),
        default="distance",
        help="what a car scores in a race: the distance it covers, or its place at "
        "the finish (1 for first down to 0 for last)",
    )
    parser.add_argument(
        "--distance-bonus",
        type=float,
        default=0.0,
        help="with --reward placement, also this for covering --max-speed x "
        "--race-seconds, and as much less for less (0.1 is half a place's "
        "worth at the most, with 6 cars), so the leader has a reason not to stop",
    )
    parser.add_argument(
        "--lidar-cars",
        action="store_true",
        help="the lidar sees the other cars (and wrecks) as well as the walls",
    )
    parser.add_argument(
        "--lidar-rays",
        type=int,
        default=1,
        help="with --lidar-cars, rays each lidar beam also sweeps its sector with, "
        "reading the nearest car any of them hits, so that a car can't hide between "
        "beams (walls still read along the beam)",
    )
    parser.add_argument(
        "--crashed-last",
        action="store_true",
        help="with --reward placement, every car that crashed places behind every "
        "car still racing (the crashed ones by how far they got)",
    )
    parser.add_argument(
        "--balance-grid",
        action="store_true",
        help="start every genome from every place on the grid equally often over "
        "its races (exactly so when --rounds is a multiple of --cars)",
    )
    parser.add_argument(
        "--crash-penalty",
        type=float,
        default=0.0,
        help="taken off a car's score when it crashes: metres with --reward "
        "distance; with --reward placement, 1 / (cars - 1) is one place",
    )
    parser.add_argument(
        "--speciation-threshold",
        type=float,
        default=2.0,
        help="genome distance beyond which a genome starts a new species",
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
    if args.field_ranks is not None and (
        len(args.field_ranks) != args.cars or max(args.field_ranks) > args.population
    ):
        parser.error("--field-ranks needs one rank per car, none above --population")
    if not 0 <= args.grid_spacing_jitter < args.grid_spacing:
        parser.error(
            "--grid-spacing-jitter must be at least 0 and below --grid-spacing"
        )

    race = F1TenthRace(
        args.cars,
        map_name=args.map,
        max_speed=args.max_speed,
        grid_spacing=args.grid_spacing,
        crash_penalty=args.crash_penalty,
        reward=args.reward,
        crashed_last=args.crashed_last,
        distance_weight=args.distance_bonus / (args.max_speed * args.race_seconds),
        lidar_cars=args.lidar_cars,
        lidar_rays=args.lidar_rays,
        grid_spacing_jitter=args.grid_spacing_jitter,
        cars_per_row=args.cars_per_row,
        accel_noise=args.accel_noise,
    )
    race_steps = round(args.race_seconds / race.dt)
    # a race is done when every car is out, but it doesn't start over: the wrecks
    # just stand there, so score straight through and keep the last crash in replays
    max_resets = None
    test_steps = round((args.test_seconds or args.race_seconds) / race.dt)
    # the test races start from an even grid, without noise
    test_race = race
    if (
        args.test_grid_spacing is not None
        or args.grid_spacing_jitter
        or args.accel_noise
    ):
        test_race = F1TenthRace(
            args.cars,
            map_name=args.map,
            max_speed=args.max_speed,
            grid_spacing=args.test_grid_spacing or args.grid_spacing,
            cars_per_row=args.cars_per_row,
            lidar_cars=args.lidar_cars,
            lidar_rays=args.lidar_rays,
        )
    if args.test == "bots":
        test_fn = make_top_k_test(
            make_test_fn(
                RaceAgainstBots(test_race),
                num_steps=test_steps,
                num_episodes=args.field_races,
                frames_every=race.record_every,
            ),
            name="distance_against_bots",
        )
    else:
        test_fn = make_field_test(
            make_mp_field_fn(
                test_race,
                args.field_races,
                test_steps,
                max_resets=max_resets,
                record_every=race.record_every,
                record_fn=race.record_frame,
            ),
            size=args.cars,
            # just the race for the monitor: the stats come from training's races
            name=None,
            ranks=args.field_ranks,
            # from a grid in no order of fitness, so the fast can catch the slow
            shuffle=True,
        )

    neat = NEAT(
        config=make_config(
            race.obs_size,
            race.act_size,
            args.cars,
            args.population,
            args.speciation_threshold,
        ),
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
            # lap time, passes, crashes and speeds, from every race
            stats_fn=race.stats,
            balance_seats=args.balance_grid,
        ),
        test_fn=test_fn,
        test_every=args.test_every,
        monitor=Monitor(args.monitor, project="f1tenth") if args.monitor else None,
    )
    neat.run(
        seed=args.seed,
        num_generations=args.generations,
        episode_fn=make_episode_fn(race),
    )


if __name__ == "__main__":
    main()
