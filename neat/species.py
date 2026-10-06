from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpeciesData:
    next_idx: jax.Array
    next_species_id: jax.Array
    species_id: jax.Array
    cumulative_fitness: jax.Array
    ema_fitness: jax.Array
    best_fitness: jax.Array
    species_size: jax.Array
    stagnation: jax.Array
    species_age: jax.Array
    mask: jax.Array


def next_gen(species_data: SpeciesData, ema_period: int) -> SpeciesData:
    new_mean_fitness = jnp.where(
        species_data.species_size > 0,
        species_data.cumulative_fitness / species_data.species_size,
        0,
    )
    ema_fitness = (new_mean_fitness * (2 / (ema_period + 1))) + (
        species_data.ema_fitness * (1 - (2 / (ema_period + 1)))
    )

    best_fitness = jnp.maximum(species_data.best_fitness, ema_fitness)
    new_best = best_fitness > species_data.best_fitness
    # only count stagnation/new best when the ema period is full
    ema_full = species_data.species_age >= ema_period
    return replace(
        species_data,
        best_fitness=jnp.where(ema_full, best_fitness, species_data.best_fitness),
        ema_fitness=ema_fitness,
        stagnation=jnp.where(
            new_best | (~species_data.mask) | (~ema_full),
            0,
            species_data.stagnation + 1,
        ),
        species_age=jnp.where(species_data.mask, species_data.species_age + 1, 0),
    )


def reset_counts(species_data: SpeciesData) -> SpeciesData:
    return replace(
        species_data,
        cumulative_fitness=jnp.zeros_like(species_data.cumulative_fitness),
        species_size=jnp.zeros_like(species_data.species_size),
    )


def remove_stagnant_species(
    species_data: SpeciesData, stagnation_fn: StagnationFn
) -> SpeciesData:
    stagnated = stagnation_fn(species_data)
    new_mask = (~stagnated) & species_data.mask
    species_data = replace(
        species_data,
        species_id=jnp.where(new_mask, species_data.species_id, -1),
        stagnation=jnp.where(new_mask, species_data.stagnation, 0),
        mask=new_mask,
        next_idx=new_mask.sum(),
        best_fitness=jnp.where(
            new_mask, species_data.best_fitness, jnp.finfo(jnp.float32).min
        ),
        ema_fitness=jnp.where(new_mask, species_data.ema_fitness, 0),
    )

    sorted_idxs = jnp.argsort(species_data.species_size, stable=True, descending=True)
    return replace(
        species_data,
        species_id=species_data.species_id[sorted_idxs],
        cumulative_fitness=species_data.cumulative_fitness[sorted_idxs],
        best_fitness=species_data.best_fitness[sorted_idxs],
        species_size=species_data.species_size[sorted_idxs],
        stagnation=species_data.stagnation[sorted_idxs],
        mask=species_data.mask[sorted_idxs],
        species_age=species_data.species_age[sorted_idxs],
        ema_fitness=species_data.ema_fitness[sorted_idxs],
    )


def remove_extinct_species(species_data: SpeciesData) -> SpeciesData:
    new_mask = species_data.species_size > 0
    species_data = replace(
        species_data,
        species_id=jnp.where(new_mask, species_data.species_id, -1),
        stagnation=jnp.where(new_mask, species_data.stagnation, 0),
        mask=new_mask,
        next_idx=new_mask.sum(),
        best_fitness=jnp.where(
            new_mask, species_data.best_fitness, jnp.finfo(jnp.float32).min
        ),
    )

    sorted_idxs = jnp.argsort(species_data.species_size, stable=True, descending=True)
    return replace(
        species_data,
        species_id=species_data.species_id[sorted_idxs],
        cumulative_fitness=species_data.cumulative_fitness[sorted_idxs],
        ema_fitness=species_data.ema_fitness[sorted_idxs],
        best_fitness=species_data.best_fitness[sorted_idxs],
        species_size=species_data.species_size[sorted_idxs],
        stagnation=species_data.stagnation[sorted_idxs],
        species_age=species_data.species_age[sorted_idxs],
        mask=species_data.mask[sorted_idxs],
    )


def assign_species(
    species_data: SpeciesData,
    distances: jax.Array,
    cur_species_id: jax.Array,
    fitness: jax.Array,
    min_distance_threshold: Optional[float] = None,
    max_transfer_age: Optional[int] = None,
) -> Tuple[SpeciesData, jax.Array]:
    transfer_age_limit = jnp.inf if max_transfer_age is None else max_transfer_age

    species_distances = jnp.where(species_data.mask, distances, jnp.inf)
    min_distances = jnp.min(species_distances)
    min_idx = jnp.argmin(species_distances)

    create_new = False
    if min_distance_threshold is not None:
        create_new = (min_distances > min_distance_threshold) & (
            species_data.next_idx < species_data.species_id.shape[0]
        )
    # check if the species is too old to be transferred to
    assigned_species_id = jnp.where(
        species_data.species_age[min_idx] <= transfer_age_limit,
        species_data.species_id[min_idx],
        cur_species_id,
    )

    return jax.lax.cond(
        create_new,
        lambda _: create_new_species(species_data, fitness),
        lambda _: update_species(species_data, assigned_species_id, fitness),
        operand=None,
    )


def update_species(
    species_data: SpeciesData, species_id: jax.Array, fitness: jax.Array
) -> Tuple[SpeciesData, jax.Array]:
    idx = (species_data.species_id == species_id).argmax()
    return replace(
        species_data,
        cumulative_fitness=species_data.cumulative_fitness.at[idx].add(fitness),
        species_size=species_data.species_size.at[idx].add(1),
    ), species_id


def create_new_species(
    species_data: SpeciesData, fitness: jax.Array
) -> Tuple[SpeciesData, jax.Array]:
    new_species_id = species_data.next_species_id
    new_idx = species_data.next_idx
    return replace(
        species_data,
        next_idx=new_idx + 1,
        next_species_id=new_species_id + 1,
        species_id=species_data.species_id.at[new_idx].set(new_species_id),
        cumulative_fitness=species_data.cumulative_fitness.at[new_idx].add(fitness),
        species_size=species_data.species_size.at[new_idx].add(1),
        mask=species_data.mask.at[new_idx].set(True),
        species_age=species_data.species_age.at[new_idx].set(0),
    ), new_species_id


def get_species_stats(species_data: SpeciesData, prev_stats) -> Dict:
    species_dict = {}
    pop_size = species_data.species_size.sum()
    for i in range(pop_size):
        dom_str = f"species_dominance/s{species_data.species_id[i]}"
        fit_str = f"species_fitness/s{species_data.species_id[i]}"
        if species_data.mask[i]:
            species_dict[dom_str] = float(species_data.species_size[i] / pop_size)
            species_dict[fit_str] = float(species_data.ema_fitness[i])
    for k, v in prev_stats.items():
        if k not in species_dict and "species_dominance" in k and v > 0.0:
            species_dict[k] = 0.0
    return species_dict


def fill_prev_stats(prev_stats, cur_stats) -> Dict:
    for k, _ in cur_stats.items():
        if "species_dominance" in k and k not in prev_stats:
            prev_stats[k] = 0.0
    return prev_stats


def init_species_data(maximum_species: int, next_species_id: int = 0) -> SpeciesData:
    return SpeciesData(
        next_idx=jnp.int32(0),
        next_species_id=jnp.int32(next_species_id),
        species_id=jnp.full(maximum_species, -1, dtype=jnp.int32),
        cumulative_fitness=jnp.zeros(maximum_species, dtype=jnp.float32),
        best_fitness=jnp.full(
            maximum_species, jnp.finfo(jnp.float32).min, dtype=jnp.float32
        ),
        ema_fitness=jnp.zeros(maximum_species, dtype=jnp.float32),
        species_size=jnp.zeros(maximum_species, dtype=jnp.int32),
        stagnation=jnp.zeros(maximum_species, dtype=jnp.int32),
        mask=jnp.zeros(maximum_species, dtype=jnp.bool_),
        species_age=jnp.zeros(maximum_species, dtype=jnp.int32),
    )


StagnationFn = Callable[[SpeciesData], jax.Array]


def make_improvement_stagnation_fn(max_stagnated_steps: int) -> StagnationFn:
    def stagnated(species_data: SpeciesData) -> jax.Array:
        return species_data.stagnation >= max_stagnated_steps

    return stagnated


def make_relative_fitness_stagnation_fn(minimum_stagnation: int) -> StagnationFn:
    def stagnated(species_data: SpeciesData) -> jax.Array:
        fitness_mean = jnp.mean(species_data.ema_fitness, where=species_data.mask)
        return (species_data.stagnation >= minimum_stagnation) & (
            species_data.ema_fitness < fitness_mean
        )

    return stagnated


def make_remove_last_if_stagnant_and_full_stagnation_fn(
    max_stagnated_steps: int, full_target: int
) -> StagnationFn:
    def stagnated(species_data: SpeciesData) -> jax.Array:
        min_ema = jnp.min(
            species_data.ema_fitness, where=species_data.mask, initial=jnp.inf
        )
        min_ema_mask = species_data.ema_fitness == min_ema
        stagnated_mask = species_data.stagnation >= max_stagnated_steps
        full_mask = species_data.next_idx >= full_target
        return min_ema_mask & stagnated_mask & full_mask

    return stagnated
