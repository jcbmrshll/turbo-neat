"""A small HTTP server that training runs push metrics and media to, and that serves
the dashboard for browsing them.

    uv run turbo-neat-monitor --port 8008 --dir runs

Everything lives on disk under the run directory, one folder per run:

    <dir>/<run id>/meta.json      project, name, config, timestamps, status
    <dir>/<run id>/metrics.jsonl  one {"step": ..., "time": ..., <metrics>} per line
    <dir>/<run id>/media.jsonl    one {"key": ..., "step": ..., "file": ...} per line
    <dir>/<run id>/members.jsonl  every member of each generation, one line per
                                  generation (see monitor.lineage)
    <dir>/<run id>/media/         the media files themselves
    <dir>/<run id>/episodes/      the tests' episodes, as raw arrays (.npz)
    <dir>/<run id>/networks/      every member's network, one .npz per generation
                                  (see monitor.networks)
    <dir>/<run id>/returns/       every member's reward in each episode it played
                                  for its fitness, one .npz per generation
    <dir>/<run id>/member_episodes/  the episodes members played for their fitness,
                                  <step>-<env>.npz per generation

Runs send episodes as raw arrays, and the dashboard gets them as they are (see
pack_arrays) and draws them itself (static/renderers.js), so it can play, pause
and scrub them, and show what a network was doing at each frame.
"""

import argparse
import gzip
import io
import json
import mimetypes
import os
import re
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from monitor.aliases import alias
from monitor.lineage import Lineage
from monitor.networks import describe_network, unpack_network

STATIC_DIR = Path(__file__).parent / "static"

# content types the server accepts as media, and the extension each is stored under
MEDIA_TYPES = {
    "image/png": "png",
    "image/gif": "gif",
    "image/jpeg": "jpg",
    "image/svg+xml": "svg",
    "image/webp": "webp",
    "video/mp4": "mp4",
    "application/json": "json",
}


# runs get a memorable name unless the training script gives one
ADJECTIVES = """amber brisk calm deft eager fallow gilded hardy idle jolly keen lucid
mellow nimble olive patient quiet rustic sly tawny umber vivid wary young zesty""".split()
NOUNS = """badger crane dune ember finch grove heron ibis juniper kestrel lark maple
newt otter pine quail reed sparrow thistle vole wren yarrow""".split()


def _safe(name: str) -> str:
    """Make a metric/media key safe to use as a file name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "media"


class RunStore:
    """Runs on disk. One lock guards every write; reads go straight to the files."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        # run id -> its lineage, indexed on first use and kept up to date after
        self.lineages: Dict[str, Lineage] = {}
        # run id -> its latest mean fitness, for runs whose meta doesn't keep it
        self.mean_fitness: Dict[str, Optional[float]] = {}

    def _dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise KeyError(run_id)
        path = self.root / run_id
        if not path.is_dir():
            raise KeyError(run_id)
        return path

    def _read_meta(self, run_dir: Path) -> Dict[str, Any]:
        return json.loads((run_dir / "meta.json").read_text())

    def _write_meta(self, run_dir: Path, meta: Dict[str, Any]) -> None:
        tmp = run_dir / "meta.json.tmp"
        tmp.write_text(json.dumps(meta))
        tmp.replace(run_dir / "meta.json")

    def create(
        self, project: str, name: Optional[str], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        now = time.time()
        run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        run_id += "-" + secrets.token_hex(2)
        if name is None:
            num = sum(1 for m in self.list() if m["project"] == project) + 1
            name = f"{secrets.choice(ADJECTIVES)}-{secrets.choice(NOUNS)}-{num}"
        meta = {
            "id": run_id,
            "project": project,
            "name": name,
            "config": config,
            "created": now,
            "updated": now,
            "status": "running",
            "step": None,
            "seed": config.get("run", {}).get("seed"),
            "num_generations": config.get("run", {}).get("num_generations"),
        }
        with self.lock:
            run_dir = self.root / run_id
            (run_dir / "media").mkdir(parents=True)
            (run_dir / "episodes").mkdir()
            self._write_meta(run_dir, meta)
        return meta

    def log(self, run_id: str, step: int, metrics: Dict[str, float]) -> None:
        run_dir = self._dir(run_id)
        now = time.time()
        with self.lock:
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "time": now, **metrics}) + "\n")
            meta = self._read_meta(run_dir)
            meta["updated"] = now
            meta["step"] = step if meta["step"] is None else max(meta["step"], step)
            # the run list shows it, without reading every run's metrics
            if "mean_fitness" in metrics:
                meta["mean_fitness"] = metrics["mean_fitness"]
            self._write_meta(run_dir, meta)

    def add_members(self, run_id: str, step: int, body: Dict[str, Any]) -> None:
        run_dir = self._dir(run_id)
        if len({len(body[k]) for k in ("ids", "parents", "species", "fitness")}) > 1:
            raise ValueError("ids, parents, species and fitness differ in length")
        if any(len(p) != 2 for p in body["parents"]):
            raise ValueError("parents must be pairs of genome ids")
        row = {
            "step": step,
            "time": time.time(),
            "ids": body["ids"],
            "parents": body["parents"],
            "species": body["species"],
            "fitness": body["fitness"],
        }
        with self.lock:
            with open(run_dir / "members.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            meta = self._read_meta(run_dir)
            meta["updated"] = row["time"]
            self._write_meta(run_dir, meta)

    def lineage(self, run_id: str) -> Lineage:
        """The run's lineage, caught up with everything logged so far. Hold its lock
        while querying it."""
        run_dir = self._dir(run_id)
        with self.lock:
            if run_id not in self.lineages:
                self.lineages[run_id] = Lineage(run_dir / "members.jsonl")
            lineage = self.lineages[run_id]
        with lineage.lock:
            lineage.update()
        return lineage

    def leaderboard(
        self,
        run_id: str,
        scope: str,
        limit: int,
        species: Optional[int],
        step: Optional[int],
    ) -> Dict[str, Any]:
        lineage = self.lineage(run_id)
        with lineage.lock:
            return lineage.leaderboard(scope, limit, species, step)

    def players(self, run_id: str, genome_ids: Any) -> List[Dict[str, Any]]:
        """Genomes by id, name and the species they were born into (None for any
        not logged as members yet)."""
        lineage = self.lineage(run_id)
        with lineage.lock:
            players = []
            for g in map(int, genome_ids):
                known = lineage.known(g)
                species = int(lineage.birth_species[g]) if known else None
                players.append(
                    {
                        "id": g,
                        "name": alias(g, species) if species is not None else None,
                        "birth_species": species,
                    }
                )
            return players

    def individual(self, run_id: str, genome_id: int) -> Dict[str, Any]:
        lineage = self.lineage(run_id)
        with lineage.lock:
            individual = lineage.individual(genome_id)
        individual["returns"] = self.returns(
            run_id, genome_id, individual["born"], individual["last"]
        )
        return individual

    def _save(self, run_id: str, kind: str, step: int, data: bytes) -> None:
        path = self._dir(run_id) / kind / f"{step}.npz"
        path.parent.mkdir(exist_ok=True)
        # written aside and moved into place, so a reader never sees half a file
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def save_networks(self, run_id: str, step: int, data: bytes) -> None:
        self._save(run_id, "networks", step, data)

    def save_returns(self, run_id: str, step: int, data: bytes) -> None:
        self._save(run_id, "returns", step, data)

    def returns(
        self, run_id: str, genome_id: int, born: int, last: int
    ) -> Optional[Dict[str, Any]]:
        """A summary of the member's reward in every episode it played, over every
        epoch it was alive, or None if the run didn't log them."""
        run_dir = self._dir(run_id)
        rewards, epochs = [], 0
        for step in range(born, last + 1):
            path = run_dir / "returns" / f"{step}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as npz:
                row = np.flatnonzero(npz["ids"] == genome_id)
                if row.size:
                    rewards.append(npz["returns"][row[0]])
                    epochs += 1
        if not rewards:
            return None
        r = np.concatenate(rewards)
        r = r[np.isfinite(r)]
        if not r.size:
            return None
        return {
            "episodes": int(r.size),
            "epochs": epochs,
            "mean": float(r.mean()),
            "median": float(np.median(r)),
            "max": float(r.max()),
            "min": float(r.min()),
            "stddev": float(r.std()),
        }

    def network(
        self, run_id: str, genome_id: int, step: Optional[int]
    ) -> Dict[str, Any]:
        """A member's network in one of its generations (its latest by default)."""
        run_dir = self._dir(run_id)
        lineage = self.lineage(run_id)
        with lineage.lock:
            if not lineage.known(genome_id):
                raise KeyError(genome_id)
            born, last = int(lineage.born[genome_id]), int(lineage.last[genome_id])
        steps = [
            s
            for s in range(born, last + 1)
            if (run_dir / "networks" / f"{s}.npz").exists()
        ]
        if not steps:
            raise KeyError(genome_id)
        step = steps[-1] if step is None else step
        if step not in steps:
            raise KeyError(step)
        with np.load(run_dir / "networks" / f"{step}.npz", allow_pickle=False) as npz:
            arrays = unpack_network(npz, genome_id)
        if arrays is None:
            raise KeyError(genome_id)
        genome_config = self._read_meta(run_dir)["config"].get("genome_config", {})
        network = describe_network(
            **arrays,
            input_size=genome_config["input_size"],
            output_size=genome_config["output_size"],
            # the config was logged with each activation function by name; runs from
            # before input and output activations were given as functions have no
            # activations, and their ids index activation_fns
            activation_names=dict(
                enumerate(
                    genome_config.get("activations")
                    or genome_config.get("activation_fns", [])
                )
            ),
            input_labels=genome_config.get("input_labels"),
            output_labels=genome_config.get("output_labels"),
        )
        return {**network, "step": step}

    def save_member_episodes(
        self, run_id: str, step: int, env: str, data: bytes
    ) -> None:
        path = self._dir(run_id) / "member_episodes" / f"{step}-{env}.npz"
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def _member_episode_files(self, run_dir: Path) -> Dict[int, Path]:
        """step -> that generation's members' episodes"""
        files = (run_dir / "member_episodes").glob("*.npz")
        return {int(f.stem.split("-", 1)[0]): f for f in files}

    def member_episodes(
        self, run_id: str, genome_id: int, step: Optional[int]
    ) -> Dict[str, Any]:
        """The episodes a member played in one of its generations (its latest by
        default), and who played them with it."""
        run_dir = self._dir(run_id)
        lineage = self.lineage(run_id)
        with lineage.lock:
            if not lineage.known(genome_id):
                raise KeyError(genome_id)
            born, last = int(lineage.born[genome_id]), int(lineage.last[genome_id])
        files = self._member_episode_files(run_dir)
        steps = sorted(s for s in files if born <= s <= last)
        if not steps:
            raise KeyError(genome_id)
        step = steps[-1] if step is None else step
        if step not in steps:
            raise KeyError(step)
        with np.load(files[step], allow_pickle=False) as npz:
            players = npz["players"]
            # each seat's reward over the episode, if the run recorded it
            rewards = npz["rewards"][:, -1] if "rewards" in npz.files else None
        with lineage.lock:

            def player(p: int) -> Dict[str, Any]:
                if not lineage.known(p):
                    return {"id": p, "name": None}
                return {
                    "id": p,
                    "name": alias(p, int(lineage.birth_species[p])),
                }

            episodes = [
                {
                    "index": int(i),
                    "players": [
                        _with_reward(player(int(p)), rewards, i, seat)
                        for seat, p in enumerate(players[i])
                    ],
                    "seat": int(np.flatnonzero(players[i] == genome_id)[0]),
                }
                for i in np.flatnonzero((players == genome_id).any(axis=1))
            ]
        return {
            "step": step,
            "env": files[step].stem.split("-", 1)[1],
            "episodes": episodes,
        }

    def member_episode(self, run_id: str, step: int, index: int) -> bytes:
        """One of a generation's member episodes, packed for the dashboard."""
        source = self._member_episode_files(self._dir(run_id))[step]
        with np.load(source, allow_pickle=False) as npz:
            if not 0 <= index < len(npz["players"]):
                raise KeyError(index)
            return pack_arrays({k: npz[k][index] for k in npz.files if k != "players"})

    def episode(self, run_id: str, file: str) -> bytes:
        """A test's episode, packed for the dashboard."""
        path = self._dir(run_id) / "episodes" / _safe(file)
        if not path.is_file():
            raise KeyError(file)
        with np.load(path, allow_pickle=False) as npz:
            return pack_arrays({k: npz[k] for k in npz.files if k != "players"})

    def _add_entry(self, run_dir: Path, entry: Dict[str, Any]) -> None:
        """Append to the run's media, holding the lock."""
        with open(run_dir / "media.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        meta = self._read_meta(run_dir)
        meta["updated"] = entry["time"]
        self._write_meta(run_dir, meta)

    def add_media(
        self,
        run_id: str,
        key: str,
        step: int,
        content_type: str,
        data: bytes,
        players: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Store media; `players` names the genomes it shows."""
        run_dir = self._dir(run_id)
        ext = MEDIA_TYPES[content_type]
        file = f"{_safe(key)}-{step}.{ext}"
        entry = {"key": key, "step": step, "file": file, "time": time.time()}
        if players is not None:
            entry["players"] = players
        with self.lock:
            (run_dir / "media" / file).write_bytes(data)
            self._add_entry(run_dir, entry)
        return entry

    def add_episode(
        self, run_id: str, key: str, step: int, env: str, data: bytes
    ) -> Dict[str, Any]:
        """Store a test's episode, raw, for the dashboard to draw: an entry in the
        run's media naming its `env`, and the genomes playing it, one per seat."""
        run_dir = self._dir(run_id)
        file = f"{_safe(key)}-{step}.npz"
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            arrays = {name: npz[name] for name in npz.files}
        entry = {"key": key, "step": step, "env": env, "episode": file}
        # who played each seat, if the run said, for the dashboard to name, with
        # each one's reward over the episode if the run recorded it; a lone policy
        # among bots says which seat is its
        ids = arrays.get("players")
        if ids is not None:
            players = self.players(run_id, ids)
            final = arrays["rewards"][-1] if "rewards" in arrays else None
            seats = [int(arrays["policy"])] if "policy" in arrays else range(len(ids))
            entry["players"] = [
                _with_reward(p, final, s) for s, p in zip(seats, players)
            ]
        entry["time"] = time.time()
        with self.lock:
            # runs from before episodes were kept raw have no episodes/ folder
            (run_dir / "episodes").mkdir(exist_ok=True)
            (run_dir / "episodes" / file).write_bytes(data)
            self._add_entry(run_dir, entry)
        return entry

    def finish(self, run_id: str, status: str) -> None:
        run_dir = self._dir(run_id)
        with self.lock:
            meta = self._read_meta(run_dir)
            meta["status"] = status
            meta["updated"] = time.time()
            self._write_meta(run_dir, meta)

    def list(self) -> List[Dict[str, Any]]:
        runs = []
        for run_dir in self.root.iterdir():
            try:
                meta = self._read_meta(run_dir)
            except (OSError, ValueError):
                continue
            meta.pop("config", None)
            if "mean_fitness" not in meta:
                meta["mean_fitness"] = self._last_mean_fitness(run_dir)
            runs.append(meta)
        return sorted(runs, key=lambda m: m["created"], reverse=True)

    def _last_mean_fitness(self, run_dir: Path) -> Optional[float]:
        """The latest mean fitness in a run's metrics, for runs from before the meta
        kept it; read once."""
        if run_dir.name not in self.mean_fitness:
            value = None
            try:
                with open(run_dir / "metrics.jsonl") as f:
                    for line in f:
                        value = json.loads(line).get("mean_fitness", value)
            except (OSError, ValueError):
                pass
            self.mean_fitness[run_dir.name] = value
        return self.mean_fitness[run_dir.name]

    def get(self, run_id: str) -> Dict[str, Any]:
        return self._read_meta(self._dir(run_id))

    def _read_jsonl(self, path: Path, since: int) -> Dict[str, Any]:
        """Lines from `since` on, plus the offset to ask for next time."""
        if not path.exists():
            return {"rows": [], "next": since}
        with open(path) as f:
            lines = f.readlines()
        # a line still being written has no newline yet; leave it for the next poll
        if lines and not lines[-1].endswith("\n"):
            lines = lines[:-1]
        return {
            "rows": [json.loads(line) for line in lines[since:]],
            "next": len(lines),
        }

    def metrics(self, run_id: str, since: int = 0) -> Dict[str, Any]:
        return self._read_jsonl(self._dir(run_id) / "metrics.jsonl", since)

    def media(self, run_id: str, since: int = 0) -> Dict[str, Any]:
        return self._read_jsonl(self._dir(run_id) / "media.jsonl", since)

    def media_path(self, run_id: str, file: str) -> Path:
        path = self._dir(run_id) / "media" / _safe(file)
        if not path.is_file():
            raise KeyError(file)
        return path


def _with_reward(player: Dict[str, Any], rewards: Any, *index: int) -> Dict[str, Any]:
    """player, with its reward from rewards[index] if there are rewards."""
    if rewards is None:
        return player
    return {**player, "reward": float(rewards[index])}


def _env(query: Dict[str, List[str]]) -> str:
    """The env an episode was logged for, which names the dashboard's renderer for
    it, and the generation's file of members' episodes."""
    env = query["env"][0]
    if not re.fullmatch(r"[A-Za-z0-9_]+", env):
        raise ValueError(f"env names are letters, digits and underscores, not {env!r}")
    return env


def pack_arrays(arrays: Dict[str, np.ndarray]) -> bytes:
    """Arrays as the dashboard reads them (static/episodes.js loadArrays): the
    length of a JSON header as a little-endian uint32, the header, {name: {"dtype",
    "shape", "offset"}}, then each array's bytes from that offset past the header,
    4-byte aligned. Floats go as float32, other numbers as int32, bools as uint8."""
    header, chunks, offset = {}, [], 0
    for name, a in arrays.items():
        a = np.asarray(a)
        if a.dtype == np.bool_:
            a = a.astype(np.uint8)
        elif np.issubdtype(a.dtype, np.floating):
            a = a.astype("<f4")
        else:
            a = a.astype("<i4")
        data = a.tobytes()
        header[name] = {"dtype": a.dtype.name, "shape": a.shape, "offset": offset}
        chunks.append(data + bytes(-len(data) % 4))
        offset += len(chunks[-1])
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 4)
    return len(head).to_bytes(4, "little") + head + b"".join(chunks)


class Handler(BaseHTTPRequestHandler):
    store: RunStore

    def log_message(self, format, *args):
        # one line per request is noise next to a training run
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def _arrays(self, body: bytes) -> None:
        # raw floats compress several times over
        if "gzip" not in self.headers.get("Accept-Encoding", ""):
            self._send(HTTPStatus.OK, body, "application/octet-stream")
            return
        body = gzip.compress(body, compresslevel=5)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        since = int(query.get("since", ["0"])[0])
        parts = [p for p in url.path.split("/") if p]
        try:
            match parts:
                case ["api", "runs"]:
                    self._json(self.store.list())
                case ["api", "runs", run_id]:
                    self._json(self.store.get(run_id))
                case ["api", "runs", run_id, "metrics"]:
                    self._json(self.store.metrics(run_id, since))
                case ["api", "runs", run_id, "media"]:
                    self._json(self.store.media(run_id, since))
                case ["api", "runs", run_id, "leaderboard"]:
                    board = self.store.leaderboard(
                        run_id,
                        scope=query.get("scope", ["alive"])[0],
                        limit=min(int(query.get("limit", ["50"])[0]), 500),
                        species=(
                            int(query["species"][0]) if "species" in query else None
                        ),
                        step=int(query["epoch"][0]) if "epoch" in query else None,
                    )
                    self._json(board)
                case ["api", "runs", run_id, "individuals", genome_id]:
                    try:
                        individual = self.store.individual(run_id, int(genome_id))
                    except KeyError:
                        self._error(HTTPStatus.NOT_FOUND, "no such individual")
                        return
                    self._json(individual)
                case ["api", "runs", run_id, "individuals", genome_id, "network"]:
                    step = query.get("step", [None])[0]
                    try:
                        network = self.store.network(
                            run_id, int(genome_id), None if step is None else int(step)
                        )
                    except KeyError:
                        self._error(HTTPStatus.NOT_FOUND, "no network logged for it")
                        return
                    self._json(network)
                case ["api", "runs", run_id, "individuals", genome_id, "episodes"]:
                    step = query.get("step", [None])[0]
                    try:
                        episodes = self.store.member_episodes(
                            run_id, int(genome_id), None if step is None else int(step)
                        )
                    except KeyError:
                        self._error(HTTPStatus.NOT_FOUND, "no episodes logged for it")
                        return
                    self._json(episodes)
                case ["api", "runs", run_id, "member_episodes", step, index]:
                    try:
                        body = self.store.member_episode(run_id, int(step), int(index))
                    except KeyError:
                        self._error(HTTPStatus.NOT_FOUND, "no such episode")
                        return
                    self._arrays(body)
                case ["api", "runs", run_id, "episodes", file]:
                    try:
                        body = self.store.episode(run_id, file)
                    except KeyError:
                        self._error(HTTPStatus.NOT_FOUND, "no such episode")
                        return
                    self._arrays(body)
                case ["media", run_id, file]:
                    self._file(self.store.media_path(run_id, file))
                case [] | ["run", *_]:
                    # the dashboard routes client-side, so every page is index.html
                    self._file(STATIC_DIR / "index.html")
                case ["static", file] if (STATIC_DIR / _safe(file)).is_file():
                    self._file(STATIC_DIR / _safe(file))
                case _:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "no such run")
        except ValueError as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))

    def do_POST(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        parts = [p for p in url.path.split("/") if p]
        try:
            match parts:
                case ["api", "runs"]:
                    body = json.loads(self._body())
                    meta = self.store.create(
                        project=body.get("project") or "default",
                        name=body.get("name"),
                        config=body.get("config") or {},
                    )
                    self._json(meta, HTTPStatus.CREATED)
                case ["api", "runs", run_id, "log"]:
                    body = json.loads(self._body())
                    self.store.log(run_id, int(body["step"]), body["metrics"])
                    self._json({"ok": True})
                case ["api", "runs", run_id, "members"]:
                    body = json.loads(self._body())
                    self.store.add_members(run_id, int(body["step"]), body)
                    self._json({"ok": True})
                case ["api", "runs", run_id, "member_episodes"]:
                    env = _env(query)
                    self.store.save_member_episodes(
                        run_id, int(query["step"][0]), env, self._body()
                    )
                    self._json({"ok": True})
                case ["api", "runs", run_id, "networks"]:
                    self.store.save_networks(
                        run_id, int(query["step"][0]), self._body()
                    )
                    self._json({"ok": True})
                case ["api", "runs", run_id, "returns"]:
                    self.store.save_returns(run_id, int(query["step"][0]), self._body())
                    self._json({"ok": True})
                case ["api", "runs", run_id, "media"]:
                    content_type = self.headers.get("Content-Type", "")
                    if content_type not in MEDIA_TYPES:
                        self._error(
                            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                            f"unsupported media type {content_type!r}",
                        )
                        return
                    body = self._body()
                    # structured media (a network, say) may name the genomes it
                    # shows, for the dashboard to name too
                    players = None
                    if content_type == "application/json":
                        media = json.loads(body)
                        ids = media.get("players") if isinstance(media, dict) else None
                        if ids is not None:
                            players = self.store.players(run_id, ids)
                    entry = self.store.add_media(
                        run_id,
                        key=query["key"][0],
                        step=int(query["step"][0]),
                        content_type=content_type,
                        data=body,
                        players=players,
                    )
                    self._json(entry, HTTPStatus.CREATED)
                case ["api", "runs", run_id, "episodes"]:
                    env = _env(query)
                    entry = self.store.add_episode(
                        run_id,
                        key=query["key"][0],
                        step=int(query["step"][0]),
                        env=env,
                        data=self._body(),
                    )
                    self._json(entry, HTTPStatus.CREATED)
                case ["api", "runs", run_id, "finish"]:
                    body = json.loads(self._body() or b"{}")
                    self.store.finish(run_id, body.get("status", "finished"))
                    self._json({"ok": True})
                case _:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "no such run")
        except (ValueError, TypeError) as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))


def serve(host: str, port: int, run_dir: str) -> None:
    Handler.store = RunStore(Path(run_dir))
    server = ThreadingHTTPServer((host, port), Handler)
    print(
        f"turbo-neat monitor on http://{host}:{port}  (runs in {os.path.abspath(run_dir)})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Serve the turbo-neat run monitor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--dir", default="runs", help="where runs are stored")
    args = parser.parse_args()
    serve(args.host, args.port, args.dir)


if __name__ == "__main__":
    main()
