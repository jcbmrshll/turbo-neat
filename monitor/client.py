"""Push metrics and media from a training run to the monitor server.

    monitor = Monitor("http://localhost:8008", project="slimevolley")
    monitor.start(config=config.to_dict())
    monitor.log(step, {"max_fitness": 3.2, "episode": Episode("boids", boids=states)})
    monitor.log(step, {"members": Members(ids, parents, species, fitness, champion)})
    monitor.log(step, {"networks": Networks(ids, from_nodes, to_nodes, weights, ...)})
    monitor.log(step, {"episodes": Episodes(Episode("boids", boids=...), players)})
    monitor.finish()

Logging never raises: if the server is down the run keeps training and the
failure is reported once.
"""

import io
import json
import sys
import urllib.error
import urllib.request
from numbers import Number
from typing import Any, Dict, Optional
from urllib.parse import quote

import numpy as np

from monitor.media import encode_media

DEFAULT_URL = "http://localhost:8008"


class Episode:
    """The raw data of one episode, e.g. Episode("boids", boids=states). The server
    renders it with its renderer for `env` (see monitor.renderers), so the
    training run never draws anything."""

    def __init__(self, env: str, **data: Any):
        self.env = env
        self.data: Dict[str, Any] = {k: np.asarray(v) for k, v in data.items()}

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, **self.data)
        return buf.getvalue()


class Episodes:
    """The episodes a generation's members played for their fitness: an Episode
    whose arrays have a leading episode axis, and the genome ids playing each
    episode, shape (episodes, players). Floats are kept as float16, which is plenty
    to draw them and halves a generation's worth."""

    def __init__(self, episode: Episode, players: Any):
        if "players" in episode.data:
            raise ValueError("an episode logged with Episodes can't have 'players'")
        self.episode = episode
        self.players = players

    def to_bytes(self) -> bytes:
        data = {
            k: v.astype(np.float16) if np.issubdtype(v.dtype, np.floating) else v
            for k, v in self.episode.data.items()
        }
        buf = io.BytesIO()
        np.savez_compressed(
            buf, players=np.asarray(self.players, dtype=np.int32), **data
        )
        return buf.getvalue()


class Members:
    """Every member of one generation, for the leaderboard and lineage views: their
    genome ids, their parents' ids (shape (n, 2), the fitter parent first, -1 for
    none), species ids and fitness, plus the champion's genome id.

    Arrays may still be on the device; they're copied off it when logged, so a
    background logging thread does the copy."""

    def __init__(
        self,
        ids: Any,
        parents: Any,
        species: Any,
        fitness: Any,
        champion: Optional[Any] = None,
    ):
        self.ids = ids
        self.parents = parents
        self.species = species
        self.fitness = fitness
        self.champion = champion

    def to_dict(self) -> Dict[str, Any]:
        fitness = np.asarray(self.fitness, dtype=np.float32)
        return {
            "ids": np.asarray(self.ids).astype(int).tolist(),
            "parents": np.asarray(self.parents).astype(int).tolist(),
            "species": np.asarray(self.species).astype(int).tolist(),
            # 6 significant digits keeps float32 noise out of the file; JSON has no
            # NaN or inf, so a diverged fitness is null
            "fitness": [float(f"{f:.6g}") if np.isfinite(f) else None for f in fitness],
            "champion": None if self.champion is None else int(self.champion),
        }


class Networks:
    """Every member's network in one generation, for the lineage view: genome ids,
    and the batched arrays of their condensed graphs, of which the first
    num_connections connections and num_nodes nodes are each genome's. Packed as
    described in monitor.networks.

    Like Members, the arrays are copied off the device when logged."""

    def __init__(
        self,
        ids: Any,
        from_nodes: Any,
        to_nodes: Any,
        weights: Any,
        num_connections: Any,
        activation_ids: Any,
        biases: Any,
        num_nodes: Any,
    ):
        self.ids = ids
        self.from_nodes = from_nodes
        self.to_nodes = to_nodes
        self.weights = weights
        self.num_connections = num_connections
        self.activation_ids = activation_ids
        self.biases = biases
        self.num_nodes = num_nodes

    def to_bytes(self) -> bytes:
        num_connections = np.asarray(self.num_connections)
        num_nodes = np.asarray(self.num_nodes)
        capacity = np.asarray(self.from_nodes).shape[1]
        # row-major masking keeps each genome's entries together, in order
        conns = np.arange(capacity) < num_connections[:, None]
        nodes = np.arange(capacity) < num_nodes[:, None]
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            ids=np.asarray(self.ids, dtype=np.int32),
            num_connections=num_connections.astype(np.int32),
            from_nodes=np.asarray(self.from_nodes)[conns].astype(np.int32),
            to_nodes=np.asarray(self.to_nodes)[conns].astype(np.int32),
            weights=np.asarray(self.weights)[conns].astype(np.float32),
            num_nodes=num_nodes.astype(np.int32),
            activation_ids=np.asarray(self.activation_ids)[nodes].astype(np.int16),
            biases=np.asarray(self.biases)[nodes].astype(np.float32),
        )
        return buf.getvalue()


def to_number(value: Any) -> Optional[float]:
    """A plain float for scalars (python, numpy or jax), None for anything else."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Number):
        return float(value)  # type: ignore[arg-type]
    if getattr(value, "shape", None) == () and hasattr(value, "item"):
        return float(value.item())
    return None


class Monitor:
    def __init__(
        self,
        url: str = DEFAULT_URL,
        project: str = "default",
        name: Optional[str] = None,
    ):
        self.url = url.rstrip("/")
        self.project = project
        self.name = name
        self.run_id: Optional[str] = None
        self._failing = False
        self._rejected: set = set()

    def _request(
        self,
        path: str,
        data: bytes,
        content_type: str = "application/json",
        timeout: float = 10,
    ) -> Optional[Dict[str, Any]]:
        req = urllib.request.Request(
            self.url + path,
            data=data,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            # the server is up but refused this request; say why, once per reason
            reason = e.read().decode(errors="replace")
            if reason not in self._rejected:
                self._rejected.add(reason)
                print(
                    f"monitor: {path.split('?')[0]} rejected: {reason}", file=sys.stderr
                )
            return None
        except (urllib.error.URLError, OSError, ValueError) as e:
            if not self._failing:
                print(f"monitor: can't reach {self.url} ({e})", file=sys.stderr)
            self._failing = True
            return None
        if self._failing:
            print(f"monitor: reconnected to {self.url}", file=sys.stderr)
            self._failing = False
        return result

    def _post_json(self, path: str, obj: Any) -> Optional[Dict[str, Any]]:
        # config holds functions (activations, stagnation); name them rather than repr
        body = json.dumps(obj, default=lambda o: getattr(o, "__name__", None) or str(o))
        return self._request(path, body.encode())

    def start(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Create the run on the server."""
        meta = self._post_json(
            "/api/runs",
            {"project": self.project, "name": self.name, "config": config or {}},
        )
        if meta is None:
            print("monitor: logging is off for this run", file=sys.stderr)
            return
        self.run_id = meta["id"]
        print(f"monitor: {self.url}/run/{self.run_id}")

    def log(self, step: int, data: Dict[str, Any]) -> None:
        """Log scalars and media for one step. Values that are neither are dropped."""
        if self.run_id is None:
            return
        metrics = {}
        for key, value in data.items():
            number = to_number(value)
            if number is not None:
                metrics[key] = number
                continue
            if isinstance(value, Members):
                self._post_json(
                    f"/api/runs/{self.run_id}/members",
                    {"step": step, **value.to_dict()},
                )
                continue
            if isinstance(value, Networks):
                self._request(
                    f"/api/runs/{self.run_id}/networks?step={step}",
                    value.to_bytes(),
                    content_type="application/x-npz",
                    timeout=60,
                )
                continue
            if isinstance(value, Episodes):
                self._request(
                    f"/api/runs/{self.run_id}/member_episodes"
                    f"?step={step}&env={quote(value.episode.env)}",
                    value.to_bytes(),
                    content_type="application/x-npz",
                    timeout=60,
                )
                continue
            if isinstance(value, Episode):
                self._request(
                    f"/api/runs/{self.run_id}/episodes"
                    f"?key={quote(key)}&step={step}&env={quote(value.env)}",
                    value.to_bytes(),
                    content_type="application/x-npz",
                    timeout=60,
                )
                continue
            media = encode_media(value)
            if media is not None:
                content_type, body = media
                self._request(
                    f"/api/runs/{self.run_id}/media?key={quote(key)}&step={step}",
                    body,
                    content_type=content_type,
                    timeout=60,
                )
        if metrics:
            self._post_json(
                f"/api/runs/{self.run_id}/log", {"step": step, "metrics": metrics}
            )

    def finish(self, status: str = "finished") -> None:
        if self.run_id is None:
            return
        self._post_json(f"/api/runs/{self.run_id}/finish", {"status": status})
