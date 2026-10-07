"""The family history of a run's population, for the leaderboard and lineage views.

A run logs every member of every generation (monitor.client.Members), one line each
in <run dir>/members.jsonl:

    {"step": 12, "champion": 1031, "ids": [...], "parents": [[a, b], ...],
     "species": [...], "fitness": [...]}

Lineage reads that file as it grows and keeps, for every genome id ever seen, when
it was born and last seen, its parents, species and fitness, in arrays indexed by
id. Genome ids are handed out densely, in birth order, so a genome's parents always
have smaller ids than it does.
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from monitor.aliases import alias, species_alias

# what the leaderboard can rank by, highest first
SORTS = ("fitness", "best", "age", "children")
# points of rank history sent with each leaderboard row, for its sparkline
SPARK_POINTS = 40
# the pedigree stops growing past this many ancestors, however deep it was asked to go
MAX_PEDIGREE = 160


def _num(x: Any) -> Optional[float]:
    """A JSON-safe float: None for NaN and inf, which JSON can't carry."""
    x = float(x)
    return x if np.isfinite(x) else None


@dataclass
class Generation:
    step: int
    champion: int
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
        self.children = np.zeros(0, dtype=np.int32)
        self.champion_gens = np.zeros(0, dtype=np.int32)

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
        self.children = pad(self.children, 0)
        self.champion_gens = pad(self.champion_gens, 0)

    def _add(self, row: Dict[str, Any]) -> None:
        step = int(row["step"])
        ids = np.asarray(row["ids"], dtype=np.int32)
        parents = np.asarray(row["parents"], dtype=np.int32).reshape(-1, 2)
        species = np.asarray(row["species"], dtype=np.int32)
        fitness = np.array(
            [np.nan if f is None else f for f in row["fitness"]], dtype=np.float64
        )
        champion = -1 if row.get("champion") is None else int(row["champion"])
        if not len(ids):
            return
        self._grow(int(max(ids.max(), parents.max(), champion)) + 1)

        is_new = self.born[ids] < 0
        newborn = ids[is_new]
        self.born[newborn] = step
        self.parents[newborn] = parents[is_new]
        self.birth_species[newborn] = species[is_new]
        # a mate that is also the fitter parent is one parent, not two
        primary, mate = parents[is_new, 0], parents[is_new, 1]
        np.add.at(self.children, primary[primary >= 0], 1)
        np.add.at(self.children, mate[(mate >= 0) & (mate != primary)], 1)

        self.last[ids] = step
        self.species[ids] = species
        self.fitness[ids] = fitness
        self.best[ids] = np.fmax(self.best[ids], fitness)
        if champion >= 0:
            self.champion_gens[champion] += 1
        self.max_species = max(self.max_species, int(species.max()))

        # 1 + how many did better
        worse = np.nan_to_num(-fitness, nan=np.inf)
        rank = np.searchsorted(np.sort(worse), worse, side="left") + 1

        self.gen_index[step] = len(self.generations)
        self.generations.append(
            Generation(step, champion, ids, species, fitness, rank, newborn)
        )

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
            "champion": latest is not None and latest.champion == gid,
            "age": int(self.last[gid] - self.born[gid] + 1),
            "fitness": _num(self.fitness[gid]),
            "best": _num(self.best[gid]),
            "children": int(self.children[gid]),
            "champion_gens": int(self.champion_gens[gid]),
            "parents": [int(p) for p in self.parents[gid]],
        }

    def history(self, gid: int, since: Optional[int] = None) -> List[List[Any]]:
        """[step, fitness, rank] for every generation the genome was in, from `since`
        on."""
        lo = self.gen_index[int(self.born[gid])]
        hi = self.gen_index[int(self.last[gid])]
        if since is not None:
            lo = max(lo, self.gen_index.get(since, lo))
        points = []
        for gen in self.generations[lo : hi + 1]:
            where = np.flatnonzero(gen.ids == gid)
            if where.size:
                i = where[0]
                points.append([gen.step, _num(gen.fitness[i]), int(gen.rank[i])])
        return points

    def known(self, gid: int) -> bool:
        return 0 <= gid < len(self.born) and self.born[gid] >= 0

    def champions(self) -> List[Dict[str, Any]]:
        """Each generation's champion, as it stood after that generation's test."""
        return [
            {
                "step": gen.step,
                "id": gen.champion,
                "name": alias(gen.champion, int(self.birth_species[gen.champion])),
                "birth_species": int(self.birth_species[gen.champion]),
            }
            for gen in self.generations
            if self.known(gen.champion)
        ]

    def species_names(self) -> Dict[int, str]:
        return {s: species_alias(s) for s in range(self.max_species + 1)}

    def leaderboard(self, scope: str, sort: str, limit: int) -> Dict[str, Any]:
        """The top `limit` genomes by `sort`, among the current generation ("alive")
        or everyone the run has seen ("all")."""
        latest = self.latest
        if latest is None:
            return {"step": None, "rows": [], "species_names": {}}
        if sort not in SORTS:
            raise ValueError(f"sort must be one of {', '.join(SORTS)}")
        if scope == "alive":
            ids = np.unique(latest.ids)
        elif scope == "all":
            ids = np.flatnonzero(self.born >= 0)
        else:
            raise ValueError("scope must be alive or all")
        keys = {
            "fitness": self.fitness[ids],
            "best": self.best[ids],
            "age": (self.last[ids] - self.born[ids]).astype(np.float64),
            "children": self.children[ids].astype(np.float64),
        }
        # highest first, ties broken by best fitness, then by the oldest id
        primary = np.nan_to_num(keys[sort], nan=-np.inf)
        tiebreak = np.nan_to_num(keys["best"], nan=-np.inf)
        order = np.lexsort((ids, -tiebreak, -primary))[:limit]
        rows = []
        for gid in ids[order].tolist():
            row = self.summary(gid)
            spark_from = max(
                int(self.born[gid]), int(self.last[gid]) - SPARK_POINTS + 1
            )
            row["history"] = self.history(gid, since=spark_from)
            rows.append(row)
        return {
            "step": latest.step,
            "champion": latest.champion,
            "population": int(len(latest.ids)),
            "total": int((self.born >= 0).sum()),
            "rows": rows,
            "species_names": self.species_names(),
        }

    def living_descendants(self, gid: int) -> int:
        """How many of the current generation descend from the genome (not counting
        itself)."""
        latest = self.generations[-1]
        desc = np.zeros(len(self.born), dtype=bool)
        desc[gid] = True
        # parents are born before their children, so one pass forward in time
        # reaches every descendant
        for gen in self.generations[self.gen_index[int(self.born[gid])] + 1 :]:
            if not gen.newborn.size:
                continue
            parents = self.parents[gen.newborn]
            from_parent = desc[np.maximum(parents, 0)] & (parents >= 0)
            desc[gen.newborn] = from_parent.any(axis=1)
        desc[gid] = False
        return int(desc[latest.ids].sum())

    def individual(self, gid: int, depth: int) -> Dict[str, Any]:
        """Everything the lineage view shows about one genome: its record, its own
        fitness history, its line of descent back to a founder, its children, and
        its pedigree `depth` generations of parents back."""
        if not self.known(gid):
            raise KeyError(gid)
        latest = self.generations[-1]

        # the line of descent follows the fitter parent, whose structure the
        # offspring inherits; ids only go down, so this always ends at a founder
        line = [gid]
        while self.parents[line[-1], 0] >= 0:
            line.append(int(self.parents[line[-1], 0]))
        line.reverse()
        # fitness along the line: each ancestor's own, until its heir is born
        line_fitness: List[List[Any]] = []
        for i, aid in enumerate(line):
            heir_born = self.born[line[i + 1]] if i + 1 < len(line) else None
            line_fitness += [
                p[:2]
                for p in self.history(aid)
                if heir_born is None or p[0] < heir_born
            ]
        line_rows = []
        for aid in line:
            row = self.summary(aid)
            primary, mate = self.parents[aid]
            if mate >= 0 and mate != primary:
                row["mate"] = {
                    "id": int(mate),
                    "name": alias(int(mate), int(self.birth_species[mate])),
                }
            line_rows.append(row)

        # the pedigree: ancestors through both parents, nearest first
        depth_of = {gid: 0}
        frontier = [gid]
        for d in range(1, depth + 1):
            nxt = []
            for x in frontier:
                for p in self.parents[x].tolist():
                    if p >= 0 and p not in depth_of:
                        depth_of[p] = d
                        nxt.append(p)
            frontier = nxt
            if len(depth_of) >= MAX_PEDIGREE:
                break
        on_line = set(line)
        pedigree = []
        for aid, d in depth_of.items():
            node = self.summary(aid)
            node["depth"] = d
            node["on_line"] = aid in on_line
            pedigree.append(node)
        deeper = any((self.parents[x] >= 0).any() for x in frontier)

        child_ids = np.flatnonzero(
            (self.parents[:, 0] == gid) | (self.parents[:, 1] == gid)
        )
        children = sorted(
            (self.summary(int(c)) for c in child_ids),
            key=lambda c: -np.inf if c["best"] is None else c["best"],
            reverse=True,
        )

        return {
            **self.summary(gid),
            "step": latest.step,
            "population": int(len(latest.ids)),
            "history": self.history(gid),
            "descendants": self.living_descendants(gid),
            "line": line_rows,
            "line_fitness": line_fitness,
            "pedigree": {"nodes": pedigree, "depth": depth, "deeper": deeper},
            "child_list": children[:60],
            "species_names": self.species_names(),
        }
