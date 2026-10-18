"""Export finished runs as static files, for the dashboard to show them without a
server (monitor/static/sources.js staticSource), e.g. on a static site.

The run monitor's server answers the dashboard from a run's folder on demand. A
static site can't, so this precomputes what the dashboard asks for, using the
monitor's own code, and writes it under <out>/<key>/ for each KEY=RUN_ID given:

    run.json         meta (with config), every epoch's metrics, the species map
                     at every epoch, the test episodes there are, the epochs
                     there are leaderboards for, and the featured members
    boards.json      the leaderboard ("alive" and "all") at each checkpoint epoch
    episodes/<step>.bin            the test's episode at that epoch, packed
    networks/<id>-<step>.json      a member's network in an epoch
    members/<id>.json              a featured member: its record, lineage, network
                                   and the episodes it played in its last epoch
    members/<id>-descent.json      its line of descent, each ancestor's network
    member-episodes/<step>-<i>.bin one of an epoch's members' episodes, packed

and <out>/index.json, a summary of each run, in the order given. The dashboard
reads <out>/<key> with staticSource(base).

    python -m monitor.export --runs runs --out export \
        cartpole=20261017-003025-7bad f1tenth=20261017-002705-b595 ...
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from monitor.server import RunStore, pack_arrays

# at most this many test episodes (and leaderboard checkpoints) per run, evenly
# spread, always with the last
MAX_CHECKPOINTS = 12
BOARD_LIMIT = 50
# the leaderboard filters by species in the browser, so each checkpoint also keeps
# this many of every species
BOARD_PER_SPECIES = 10
# members with a page of their own: the fittest at the end, and at each checkpoint
# (and everyone who played in an exported test episode)
FEATURED_FINAL = 10
FEATURED_PER_CHECKPOINT = 2
# a pedigree's line of descent is cut to its newest this many generations
PEDIGREE_GENERATIONS = 40
OFFSPRING = 30
# what each agent saw, for the dashboard to light its network up, is most of an
# episode's size (an 11-car race's is 3.6 MB, 1.6 MB gzipped). Test episodes keep
# every car's, so any of them can be picked; a member's episode keeps just the
# member's own.


def rounded(x: Any, digits: int = 5) -> Any:
    """Floats to `digits` significant digits, recursively, so the JSON stays small."""
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        # big numbers (timestamps) keep their whole part
        if abs(x) >= 10**digits:
            return round(x, 1)
        return float(f"{x:.{digits}g}")
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v, digits) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rounded(obj), separators=(",", ":")))


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def spread(steps: List[int], count: int) -> List[int]:
    """`count` of the steps, evenly spread, keeping the first and the last."""
    if len(steps) <= count:
        return steps
    picks = {steps[round(i * (len(steps) - 1) / (count - 1))] for i in range(count)}
    return sorted(picks)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def member_episode(run_dir: Path, step: int, env: str, ep: Dict[str, Any]) -> bytes:
    """One of a generation's members' episodes, packed (the monitor's
    member_episode), with what only the member's own seat saw when several played."""
    with np.load(run_dir / "member_episodes" / f"{step}-{env}.npz", allow_pickle=False) as npz:
        arrays = {k: npz[k][ep["index"]] for k in npz.files if k != "players"}
    obs = arrays.get("obs")
    if obs is not None and obs.ndim == 3 and len(ep["players"]) > 1:
        # (frames, rows, inputs), a row a seat unless obs_seats says which
        seats = arrays.get("obs_seats", np.arange(obs.shape[1]))
        rows = np.flatnonzero(seats == ep["seat"])
        arrays["obs"] = obs[:, rows]
        arrays["obs_seats"] = seats[rows]
    return pack_arrays(arrays)


def export_run(store: RunStore, key: str, run_id: str, out: Path) -> Dict[str, Any]:
    run_dir = store.root / run_id
    meta = store.get(run_id)
    lineage = store.lineage(run_id)
    latest = lineage.latest.step
    metrics = store.metrics(run_id)["rows"]
    media = read_jsonl(run_dir / "media.jsonl")
    dest = out / key
    if dest.exists():
        shutil.rmtree(dest)
    print(f"{key}: {meta['name']} ({run_id}), {latest + 1} epochs", file=sys.stderr)

    # ------------------------------------------------ test episodes

    tests = {m["step"]: m for m in media if m["key"] == "episode" and m.get("env")}
    test_steps = [s for s in sorted(tests) if (run_dir / "episodes" / f"episode-{s}.npz").exists()]
    checkpoints = spread(test_steps, MAX_CHECKPOINTS)
    if latest not in checkpoints:
        checkpoints.append(latest)
    episodes = []
    # each test player's network in its test epoch, for the network beside the
    # episode
    networks_wanted = set()
    # the members with a page of their own
    featured = set()
    for step in checkpoints:
        entry = tests.get(step)
        path = run_dir / "episodes" / f"episode-{step}.npz"
        if entry is None or not path.exists():
            continue
        with np.load(path, allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
        write_bytes(dest / "episodes" / f"{step}.bin", pack_arrays({k: v for k, v in arrays.items() if k != "players"}))
        # who played each seat, as the monitor's add_episode names them
        players = None
        if "players" in arrays:
            named = store.players(run_id, arrays["players"])
            final = arrays["rewards"][-1] if "rewards" in arrays else None
            seats = [int(arrays["policy"])] if "policy" in arrays else range(len(named))
            players = [
                {**p, "reward": float(final[s])} if final is not None else p
                for s, p in zip(seats, named)
            ]
            for p in named:
                networks_wanted.add((p["id"], step))
                # whoever played a test the page shows can be opened, from its name
                # in the episode
                featured.add(p["id"])
        episodes.append({"key": "episode", "step": step, "env": entry["env"], "episode": f"{step}.bin", "players": players})

    # ------------------------------------------------ species map

    species_pca = []
    for m in media:
        if m["key"] != "species_pca":
            continue
        path = run_dir / "media" / m["file"]
        if path.exists():
            species_pca.append({"step": m["step"], "data": json.loads(path.read_text())})

    # ------------------------------------------------ leaderboards

    boards = {}
    for step in checkpoints:
        at = {}
        for scope in ("alive", "all"):
            board = lineage.leaderboard(scope, BOARD_LIMIT, None, step)
            rows = {r["id"]: r for r in board["rows"]}
            for s in board["species"]:
                for r in lineage.leaderboard(scope, BOARD_PER_SPECIES, s["id"], step)["rows"]:
                    rows.setdefault(r["id"], r)
            # the browser filters by the species each is in, which the monitor's
            # rows only name; give its id too
            _, species_of, _, _ = lineage.as_of(lineage.generation_at(step))
            board_rows = sorted(rows.values(), key=lambda r: (-(r["fitness"] if r["fitness"] is not None else -np.inf), r["id"]))
            for r in board_rows:
                r["species"] = int(species_of[r["id"]])
            board["rows"] = board_rows
            board.pop("species_names", None)
            at[scope] = board
        boards[step] = at
        top = at["alive"]["rows"][: FEATURED_FINAL if step == latest else FEATURED_PER_CHECKPOINT]
        featured.update(r["id"] for r in top)

    # ------------------------------------------------ featured members

    for gid in sorted(featured):
        ind = lineage.individual(gid)
        ind["returns"] = store.returns(run_id, gid, ind["born"], ind["last"])
        # a long line of descent is cut to its newest generations, with the
        # ancestors off it that are left
        line = [n for n in ind["pedigree"] if n["on_line"]]
        if len(line) > PEDIGREE_GENERATIONS:
            cut = sorted(n["born"] for n in line)[-PEDIGREE_GENERATIONS]
            ind["pedigree"] = [n for n in ind["pedigree"] if n["born"] >= cut]
            ind["pedigree_cut"] = cut
        ind["offspring_list"] = ind["offspring_list"][:OFFSPRING]
        ind.pop("species_names", None)
        ind.pop("history", None)
        step = ind["last"]
        try:
            network = store.network(run_id, gid, step)
            ind["network_step"] = network["step"]
            networks_wanted.add((gid, network["step"]))
        except KeyError:
            ind["network_step"] = None
        # its own game in its last epoch, the one it's seated first in
        ind["episodes"] = None
        try:
            eps = store.member_episodes(run_id, gid, None)
        except KeyError:
            eps = None
        if eps and eps["episodes"]:
            ep = next((e for e in eps["episodes"] if e["seat"] == 0), eps["episodes"][0])
            file = f"{eps['step']}-{ep['index']}-{ep['seat']}.bin"
            write_bytes(dest / "member-episodes" / file, member_episode(run_dir, eps["step"], eps["env"], ep))
            ind["episodes"] = {"step": eps["step"], "env": eps["env"], "episodes": [ep], "file": file}
        # its line of descent, each ancestor's network as it was born (the
        # monitor's lineage_networks), for the lineage evolution card
        try:
            line = store.line_networks(run_id, gid)
        except KeyError:
            line = None
        ind["descent"] = bool(line and line["networks"])
        if ind["descent"]:
            write_json(dest / "members" / f"{gid}-descent.json", line)
        write_json(dest / "members" / f"{gid}.json", ind)

    for gid, step in sorted(networks_wanted):
        try:
            network = store.network(run_id, gid, step)
        except KeyError:
            continue
        write_json(dest / "networks" / f"{gid}-{step}.json", network)

    # ------------------------------------------------ run

    write_json(dest / "boards.json", boards)
    run = {
        "key": key,
        "meta": meta,
        "metrics": metrics,
        "episodes": episodes,
        "species_pca": species_pca,
        "checkpoints": sorted(boards),
        "featured": sorted(featured),
        "networks": sorted(f"{g}-{s}" for g, s in networks_wanted),
        "species_names": lineage.species_names(),
    }
    write_json(dest / "run.json", run)
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"  {len(episodes)} episodes, {len(boards)} boards, {len(featured)} featured, {size / 1e6:.1f} MB", file=sys.stderr)
    return {"key": key, "id": run_id, "project": meta["project"], "name": meta["name"], "epochs": latest + 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_to_export", nargs="+", metavar="KEY=RUN_ID", help="a run to export, under <out>/<key>")
    parser.add_argument("--runs", required=True, help="the monitor's run folder")
    parser.add_argument("--out", required=True, help="where to write the export")
    parser.add_argument("--only", action="append", metavar="KEY", help="export just this key (again for more), keeping the others' entries in index.json")
    args = parser.parse_args()
    showcase = [tuple(pair.split("=", 1)) for pair in args.runs_to_export]

    store = RunStore(Path(args.runs).expanduser())
    out = Path(args.out).expanduser()
    index_path = out / "index.json"
    # with --only, the other runs' entries stay as they were
    index = {e["key"]: e for e in json.loads(index_path.read_text())} if args.only and index_path.exists() else {}
    for key, run_id in showcase:
        if args.only and key not in args.only:
            continue
        index[key] = export_run(store, key, run_id, out)
    order = [k for k, _ in showcase]
    write_json(index_path, sorted(index.values(), key=lambda e: order.index(e["key"]) if e["key"] in order else 99))


if __name__ == "__main__":
    main()
