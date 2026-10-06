"""Push metrics and media from a training run to the monitor server.

    monitor = Monitor("http://localhost:8008", project="slimevolley")
    monitor.start(config=config.to_dict())
    monitor.log(step, {"max_fitness": 3.2, "episode": Episode("boids", boids=states)})
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
