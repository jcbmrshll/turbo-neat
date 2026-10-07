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
    <dir>/<run id>/episodes/      raw episode data (.npz), rendered into media/
    <dir>/<run id>/networks/      every member's network, one .npz per generation
                                  (see monitor.networks)
    <dir>/<run id>/member_episodes/  the episodes members played for their fitness,
                                  <step>-<env>.npz per generation, rendered into
                                  rendered/ the first time one is asked for

Runs send episodes as raw arrays; the server renders them (monitor.renderers)
on a background thread, so neither the training run nor the dashboard draws them.
Members' episodes are far too many to render them all, so they're drawn on demand.
"""

import argparse
import json
import mimetypes
import os
import re
import secrets
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from monitor.aliases import alias
from monitor.lineage import Lineage
from monitor.media import encode_media
from monitor.networks import describe_network, unpack_network
from monitor.renderers import RENDERERS

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
        # members' episodes render one at a time, like the champion's
        self.render_lock = threading.Lock()

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
            "champion": body.get("champion"),
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
        self, run_id: str, scope: str, sort: str, limit: int
    ) -> Dict[str, Any]:
        lineage = self.lineage(run_id)
        with lineage.lock:
            return lineage.leaderboard(scope, sort, limit)

    def champions(self, run_id: str) -> List[Dict[str, Any]]:
        lineage = self.lineage(run_id)
        with lineage.lock:
            return lineage.champions()

    def individual(self, run_id: str, genome_id: int, depth: int) -> Dict[str, Any]:
        lineage = self.lineage(run_id)
        with lineage.lock:
            return lineage.individual(genome_id, depth)

    def save_networks(self, run_id: str, step: int, data: bytes) -> None:
        path = self._dir(run_id) / "networks" / f"{step}.npz"
        path.parent.mkdir(exist_ok=True)
        # written aside and moved into place, so a reader never sees half a file
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def network(
        self, run_id: str, genome_id: int, step: Optional[int]
    ) -> Dict[str, Any]:
        """A member's network in one of its generations (its latest by default), and
        the generations it has one for."""
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
            # the config was logged with each activation function by name
            activation_names=dict(enumerate(genome_config.get("activation_fns", []))),
            input_labels=genome_config.get("input_labels"),
            output_labels=genome_config.get("output_labels"),
        )
        return {**network, "step": step, "steps": steps}

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
        default), who played them with it, and the generations it has any for."""
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
                    "players": [player(int(p)) for p in players[i]],
                    "seat": int(np.flatnonzero(players[i] == genome_id)[0]),
                }
                for i in np.flatnonzero((players == genome_id).any(axis=1))
            ]
        return {
            "step": step,
            "steps": steps,
            "env": files[step].stem.split("-", 1)[1],
            "episodes": episodes,
        }

    def member_episode_media(self, run_id: str, step: int, index: int) -> Path:
        """One of a generation's member episodes, rendered (the first time it's
        asked for) like the champion's."""
        run_dir = self._dir(run_id)
        source = self._member_episode_files(run_dir)[step]
        rendered = run_dir / "member_episodes" / "rendered"
        cached = list(rendered.glob(f"{step}-{index}.*"))
        if cached:
            return cached[0]
        with self.render_lock:
            cached = list(rendered.glob(f"{step}-{index}.*"))
            if cached:
                return cached[0]
            with np.load(source, allow_pickle=False) as npz:
                if not 0 <= index < len(npz["players"]):
                    raise KeyError(index)
                data = {
                    k: npz[k][index].astype(np.float32)
                    if np.issubdtype(npz[k].dtype, np.floating)
                    else npz[k][index]
                    for k in npz.files
                    if k != "players"
                }
            env = source.stem.split("-", 1)[1]
            media = encode_media(RENDERERS[env](data))
            if media is None:
                raise TypeError(f"the {env!r} renderer returned nothing displayable")
            content_type, body = media
            rendered.mkdir(exist_ok=True)
            path = rendered / f"{step}-{index}.{MEDIA_TYPES[content_type]}"
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(body)
            tmp.replace(path)
            return path

    def add_media(
        self,
        run_id: str,
        key: str,
        step: int,
        content_type: str,
        data: bytes,
        env: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store media; `env` names the environment of a rendered episode."""
        run_dir = self._dir(run_id)
        ext = MEDIA_TYPES[content_type]
        file = f"{_safe(key)}-{step}.{ext}"
        entry = {"key": key, "step": step, "file": file, "time": time.time()}
        if env is not None:
            entry["env"] = env
        with self.lock:
            (run_dir / "media" / file).write_bytes(data)
            with open(run_dir / "media.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            meta = self._read_meta(run_dir)
            meta["updated"] = entry["time"]
            self._write_meta(run_dir, meta)
        return entry

    def add_media_error(self, run_id: str, key: str, step: int, error: str) -> None:
        """Record media that failed to render, so the dashboard can say why."""
        run_dir = self._dir(run_id)
        entry = {"key": key, "step": step, "error": error, "time": time.time()}
        with self.lock:
            with open(run_dir / "media.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")

    def save_episode(self, run_id: str, key: str, step: int, data: bytes) -> Path:
        path = self._dir(run_id) / "episodes" / f"{_safe(key)}-{step}.npz"
        # runs from before episodes were rendered here have no episodes/ folder
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        return path

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
            runs.append(meta)
        return sorted(runs, key=lambda m: m["created"], reverse=True)

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


def render_episode(
    store: RunStore, run_id: str, key: str, step: int, env: str, path: Path
) -> None:
    """Render a saved episode into media; on failure, record the error instead."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            data = {name: npz[name] for name in npz.files}
        media = encode_media(RENDERERS[env](data))
        if media is None:
            raise TypeError(f"the {env!r} renderer returned nothing displayable")
        store.add_media(run_id, key, step, *media, env=env)
    except Exception as e:
        traceback.print_exc()
        store.add_media_error(run_id, key, step, f"{type(e).__name__}: {e}")


class Handler(BaseHTTPRequestHandler):
    store: RunStore
    # one thread, so renders queue up rather than competing with request handling
    renderer = ThreadPoolExecutor(max_workers=1)

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
                case ["api", "runs", run_id, "champions"]:
                    self._json(self.store.champions(run_id))
                case ["api", "runs", run_id, "leaderboard"]:
                    board = self.store.leaderboard(
                        run_id,
                        scope=query.get("scope", ["alive"])[0],
                        sort=query.get("sort", ["fitness"])[0],
                        limit=min(int(query.get("limit", ["50"])[0]), 500),
                    )
                    self._json(board)
                case ["api", "runs", run_id, "individuals", genome_id]:
                    try:
                        individual = self.store.individual(
                            run_id,
                            int(genome_id),
                            depth=min(int(query.get("depth", ["6"])[0]), 50),
                        )
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
                    # render its own game now (the first it's seated first in, which
                    # is the one the dashboard shows), so it's ready sooner
                    own = [e for e in episodes["episodes"] if e["seat"] == 0]
                    for episode in own[:1]:
                        self.renderer.submit(
                            self.store.member_episode_media,
                            run_id,
                            episodes["step"],
                            episode["index"],
                        )
                    self._json(episodes)
                case ["api", "runs", run_id, "member_episodes", step, index]:
                    try:
                        path = self.store.member_episode_media(
                            run_id, int(step), int(index)
                        )
                    except KeyError:
                        self._error(HTTPStatus.NOT_FOUND, "no such episode")
                        return
                    except Exception as e:
                        traceback.print_exc()
                        self._error(
                            HTTPStatus.INTERNAL_SERVER_ERROR,
                            f"couldn't render: {type(e).__name__}: {e}",
                        )
                        return
                    self._file(path)
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
                    env = query["env"][0]
                    if env not in RENDERERS:
                        known = ", ".join(sorted(RENDERERS))
                        self._error(
                            HTTPStatus.BAD_REQUEST,
                            f"no renderer for env {env!r} (known: {known})",
                        )
                        return
                    self.store.save_member_episodes(
                        run_id, int(query["step"][0]), env, self._body()
                    )
                    self._json({"ok": True})
                case ["api", "runs", run_id, "networks"]:
                    self.store.save_networks(
                        run_id, int(query["step"][0]), self._body()
                    )
                    self._json({"ok": True})
                case ["api", "runs", run_id, "media"]:
                    content_type = self.headers.get("Content-Type", "")
                    if content_type not in MEDIA_TYPES:
                        self._error(
                            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                            f"unsupported media type {content_type!r}",
                        )
                        return
                    entry = self.store.add_media(
                        run_id,
                        key=query["key"][0],
                        step=int(query["step"][0]),
                        content_type=content_type,
                        data=self._body(),
                    )
                    self._json(entry, HTTPStatus.CREATED)
                case ["api", "runs", run_id, "episodes"]:
                    env = query["env"][0]
                    if env not in RENDERERS:
                        known = ", ".join(sorted(RENDERERS))
                        self._error(
                            HTTPStatus.BAD_REQUEST,
                            f"no renderer for env {env!r} (known: {known})",
                        )
                        return
                    key, step = query["key"][0], int(query["step"][0])
                    path = self.store.save_episode(run_id, key, step, self._body())
                    self.renderer.submit(
                        render_episode, self.store, run_id, key, step, env, path
                    )
                    self._json({"queued": True}, HTTPStatus.ACCEPTED)
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
