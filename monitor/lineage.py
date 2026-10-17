"""The family history of a run's population, for the leaderboard and lineage views.

A run logs every member of every generation (monitor.client.Members), one line each
in <run dir>/members.jsonl:

    {"step": 12, "ids": [...], "parents": [[a, b], ...], "species": [...],
     "fitness": [...]}

(Runs from before champions were dropped also have a "champion" id, which is
ignored.)

Lineage reads that file as it grows and keeps, for every genome id ever seen, when
it was born and last seen, its parents, species and fitness, in arrays indexed by
id. Genome ids are handed out densely, in birth order, so a genome's parents always
have smaller ids than it does.
"""

import bisect
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from monitor.aliases import alias, species_alias

# how many parents off the line of descent the pedigree reaches: each step brings at
# most twice as many ancestors per member of the line
PEDIGREE_REACH = 3


def _num(x: Any) -> Optional[float]:
    """A JSON-safe float: None for NaN and inf, which JSON can't carry."""
    x = float(x)
    return x if np.isfinite(x) else None


@dataclass
class Generation:
    step: int
    ids: np.ndarray
    species: np.ndarray
    fitness: np.ndarray
    # 1 for the fittest; equal fitness, equal rank; NaN fitness ranks last
    rank: np.ndarray
    # ids first seen in this generation
    newborn: np.ndarray


class Lineage:
    def __init__(self, path: Path):
        self.path = path
        # held while reading the file or answering from the arrays
        self.lock = threading.Lock()
        self.offset = 0
        self.generations: List[Generation] = []
        self.gen_index: Dict[int, int] = {}  # step -> index into generations
        self.max_species = -1
        # per genome id; -1 (or NaN for fitness) for ids not seen yet
        self.born = np.full(0, -1, dtype=np.int32)
        self.last = np.full(0, -1, dtype=np.int32)
        self.parents = np.full((0, 2), -1, dtype=np.int32)
        self.birth_species = np.full(0, -1, dtype=np.int32)
        self.species = np.full(0, -1, dtype=np.int32)
        self.fitness = np.full(0, np.nan, dtype=np.float64)
        self.best = np.full(0, np.nan, dtype=np.float64)
        self.offspring = np.zeros(0, dtype=np.int32)

    # ------------------------------------------------------------ indexing

    def update(self) -> None:
        """Index the generations logged since the last update."""
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        # a line still being written has no newline yet; leave it for next time
        end = data.rfind(b"\n") + 1
        self.offset += end
        for line in data[:end].splitlines():
            if line.strip():
                self._add(json.loads(line))

    def _grow(self, size: int) -> None:
        old = len(self.born)
        if size <= old:
            return
        new = max(size, 2 * old)

        def pad(a: np.ndarray, fill: Any) -> np.ndarray:
            out = np.full((new,) + a.shape[1:], fill, dtype=a.dtype)
            out[:old] = a
            return out

        self.born = pad(self.born, -1)
        self.last = pad(self.last, -1)
        self.parents = pad(self.parents, -1)
        self.birth_species = pad(self.birth_species, -1)
        self.species = pad(self.species, -1)
        self.fitness = pad(self.fitness, np.nan)
        self.best = pad(self.best, np.nan)
        self.offspring = pad(self.offspring, 0)

    def _add(self, row: Dict[str, Any]) -> None:
        step = int(row["step"])
        ids = np.asarray(row["ids"], dtype=np.int32)
        parents = np.asarray(row["parents"], dtype=np.int32).reshape(-1, 2)
        species = np.asarray(row["species"], dtype=np.int32)
        fitness = np.array(
            [np.nan if f is None else f for f in row["fitness"]], dtype=np.float64
        )
        if not len(ids):
            return
        self._grow(int(max(ids.max(), parents.max())) + 1)

        is_new = self.born[ids] < 0
        newborn = ids[is_new]
        self.born[newborn] = step
        self.parents[newborn] = parents[is_new]
        self.birth_species[newborn] = species[is_new]
        # a mate that is also the fitter parent is one parent, not two
        primary, mate = parents[is_new, 0], parents[is_new, 1]
        np.add.at(self.offspring, primary[primary >= 0], 1)
        np.add.at(self.offspring, mate[(mate >= 0) & (mate != primary)], 1)

        self.last[ids] = step
        self.species[ids] = species
        self.fitness[ids] = fitness
        self.best[ids] = np.fmax(self.best[ids], fitness)
        self.max_species = max(self.max_species, int(species.max()))

        # 1 + how many did better
        worse = np.nan_to_num(-fitness, nan=np.inf)
        rank = np.searchsorted(np.sort(worse), worse, side="left") + 1

        self.gen_index[step] = len(self.generations)
        self.generations.append(Generation(step, ids, species, fitness, rank, newborn))

    # ------------------------------------------------------------ queries

    @property
    def latest(self) -> Optional[Generation]:
        return self.generations[-1] if self.generations else None

    def summary(self, gid: int) -> Dict[str, Any]:
        """Who a genome is: its name (after the species it was born into) and record."""
        latest = self.latest
        return {
            "id": gid,
            "name": alias(gid, int(self.birth_species[gid])),
            "species": int(self.species[gid]),
            "species_name": species_alias(int(self.species[gid])),
            "birth_species": int(self.birth_species[gid]),
            "born": int(self.born[gid]),
            "last": int(self.last[gid]),
            "alive": latest is not None and int(self.last[gid]) == latest.step,
            "age": int(self.last[gid] - self.born[gid] + 1),
            "fitness": _num(self.fitness[gid]),
            "best": _num(self.best[gid]),
            "offspring": int(self.offspring[gid]),
            "parents": [int(p) for p in self.parents[gid]],
        }

    def history(self, gid: int) -> List[List[Any]]:
        """[step, fitness, rank] for every generation the genome was in."""
        lo = self.gen_index[int(self.born[gid])]
        hi = self.gen_index[int(self.last[gid])]
        points = []
        for gen in self.generations[lo : hi + 1]:
            where = np.flatnonzero(gen.ids == gid)
            if where.size:
                i = where[0]
                points.append([gen.step, _num(gen.fitness[i]), int(gen.rank[i])])
        return points

    def known(self, gid: int) -> bool:
        return 0 <= gid < len(self.born) and self.born[gid] >= 0

    def species_names(self) -> Dict[int, str]:
        return {s: species_alias(s) for s in range(self.max_species + 1)}

    def generation_at(self, step: Optional[int]) -> Generation:
        """The generation logged at epoch `step`, or the last one before it (the
        latest if None)."""
        if step is None or not self.generations:
            return self.generations[-1]
        steps = [g.step for g in self.generations]
        return self.generations[max(0, bisect.bisect_right(steps, step) - 1)]

    def as_of(self, gen: Generation) -> Tuple[np.ndarray, ...]:
        """Every genome's fitness, species and last epoch as of generation `gen`, as
        last seen up to then, and how many offspring it had by then."""
        if gen is self.latest:
            return self.fitness, self.species, self.last, self.offspring
        n = len(self.born)
        fitness = np.full(n, np.nan)
        species = np.full(n, -1, dtype=np.int32)
        last = np.full(n, -1, dtype=np.int32)
        for g in self.generations[: self.gen_index[gen.step] + 1]:
            fitness[g.ids] = g.fitness
            species[g.ids] = g.species
            last[g.ids] = g.step
        offspring = np.zeros(n, dtype=np.int32)
        born = (self.born >= 0) & (self.born <= gen.step)
        primary, mate = self.parents[born, 0], self.parents[born, 1]
        np.add.at(offspring, primary[primary >= 0], 1)
        np.add.at(offspring, mate[(mate >= 0) & (mate != primary)], 1)
        return fitness, species, last, offspring

    def leaderboard(
        self,
        scope: str,
        limit: int,
        species: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """The `limit` fittest genomes among a generation ("alive") or everyone the
        run had seen by then ("all"), of one species if given; and the species there
        are to choose from, biggest first. The generation is the latest, or the one
        logged at epoch `step` (or the last before it), with everyone's record as it
        stood then."""
        latest = self.latest
        if latest is None:
            return {"step": None, "rows": [], "species": [], "species_names": {}}
        gen = self.generation_at(step)
        if scope == "alive":
            ids = np.unique(gen.ids)
        elif scope == "all":
            ids = np.flatnonzero((self.born >= 0) & (self.born <= gen.step))
        else:
            raise ValueError("scope must be alive or all")
        fitness, species_of, last, offspring = self.as_of(gen)
        # a genome's species is the one it was in when last seen
        present, counts = np.unique(species_of[ids], return_counts=True)
        if species is not None:
            ids = ids[species_of[ids] == species]
        # fittest first, ties to the oldest id
        order = np.lexsort((ids, -np.nan_to_num(fitness[ids], nan=-np.inf)))[:limit]

        def row(gid: int) -> Dict[str, Any]:
            born, seen = int(self.born[gid]), int(last[gid])
            return {
                "id": gid,
                "name": alias(gid, int(self.birth_species[gid])),
                "species_name": species_alias(int(species_of[gid])),
                "birth_species": int(self.birth_species[gid]),
                "born": born,
                "last": seen,
                "alive": seen == gen.step,
                "age": seen - born + 1,
                "fitness": _num(fitness[gid]),
                "offspring": int(offspring[gid]),
            }

        return {
            "step": gen.step,
            "latest": latest.step,
            "population": int(len(gen.ids)),
            "total": int(((self.born >= 0) & (self.born <= gen.step)).sum()),
            "rows": [row(gid) for gid in ids[order].tolist()],
            "species": [
                {"id": int(s), "name": species_alias(int(s)), "members": int(n)}
                for s, n in sorted(zip(present, counts), key=lambda sn: -sn[1])
            ],
            "species_names": self.species_names(),
        }

    def living_descendants(self, gid: int) -> int:
        """How many of the current generation descend from the genome (not counting
        itself)."""
        latest = self.generations[-1]
        desc = np.zeros(len(self.born), dtype=bool)
        desc[gid] = True
        # parents are born before their offspring, so one pass forward in time
        # reaches every descendant
        for gen in self.generations[self.gen_index[int(self.born[gid])] + 1 :]:
            if not gen.newborn.size:
                continue
            parents = self.parents[gen.newborn]
            from_parent = desc[np.maximum(parents, 0)] & (parents >= 0)
            desc[gen.newborn] = from_parent.any(axis=1)
        desc[gid] = False
        return int(desc[latest.ids].sum())

    def individual(self, gid: int) -> Dict[str, Any]:
        """Everything the lineage view shows about one genome: its record, its own
        fitness history, its offspring, and its pedigree: its line of descent back
        to a founder, and the ancestors up to PEDIGREE_REACH parents off it."""
        if not self.known(gid):
            raise KeyError(gid)
        latest = self.generations[-1]

        # the line of descent follows the fitter parent, whose structure the
        # offspring inherits; ids only go down, so this always ends at a founder
        line = [gid]
        while self.parents[line[-1], 0] >= 0:
            line.append(int(self.parents[line[-1], 0]))
        line.reverse()

        # the pedigree: the whole line, then each step through either parent takes
        # an ancestor one further off it. Counting fitter parents too keeps it to
        # 2 + 4 + ... + 2^reach ancestors per member of the line, not every mate's
        # own line back to a founder
        reach_of = {aid: 0 for aid in line}
        frontier = line
        for r in range(1, PEDIGREE_REACH + 1):
            nxt = []
            for x in frontier:
                for p in self.parents[x].tolist():
                    if p >= 0 and p not in reach_of:
                        reach_of[p] = r
                        nxt.append(p)
            frontier = nxt
        pedigree = []
        for aid, r in reach_of.items():
            node = self.summary(aid)
            node["reach"] = r
            node["on_line"] = r == 0
            pedigree.append(node)

        offspring_ids = np.flatnonzero(
            (self.parents[:, 0] == gid) | (self.parents[:, 1] == gid)
        )
        offspring = sorted(
            (self.summary(int(c)) for c in offspring_ids),
            key=lambda c: -np.inf if c["best"] is None else c["best"],
            reverse=True,
        )

        return {
            **self.summary(gid),
            "step": latest.step,
            "population": int(len(latest.ids)),
            "history": self.history(gid),
            "descendants": self.living_descendants(gid),
            "pedigree": pedigree,
            "offspring_list": offspring[:60],
            "species_names": self.species_names(),
        }
