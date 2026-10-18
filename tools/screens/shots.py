"""Screenshots of the NEAT dashboard, to check that a change to it leaves the run
monitor and the /neat showcase looking exactly as they did.

The same views are shot on each target, with the clock, the episodes and the
random car a race opens on frozen (freeze.js), so two captures of an unchanged
dashboard match pixel for pixel:

    for each environment's showcase run (SHOWCASE):
      run      the run's tab at the last epoch, scrolled to show all of it
      narrow   the same in a narrower window, where the layout changes
      epoch    the run's tab at an epoch partway through (EPOCHS)
      member   the tab of the member at the top of the population
    every episode on the page is paused at the same frame (FRAME of the way in)

Targets:

    monitor  the run monitor from this checkout, serving the baseline runs
    site     the /neat page of a running build of the site, at --url
    harness  site/index.html: the dashboard in a page laid out like /neat, served
             from this checkout with the site's export of the runs (--data)

Capture a target, then compare two captures:

    uv run --no-project --with playwright==1.64.0 --with numpy --with pillow \\
        python tools/screens/shots.py capture monitor --out /tmp/shots/monitor
    ... shots.py compare ~/turbo-neat-dash-baseline/shots/monitor /tmp/shots/monitor

See README.md for the baseline and how the site is built for it.
"""

import argparse
import http.server
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# the run each environment's showcase on the site uses (README.md: the baseline's
# neat-data), from ~/turbo-neat-baselines/v2/runs
SHOWCASE = [
    ("f1tenth", "20261017-002705-b595"),
    ("slimevolley", "20261017-121215-59bf"),
    ("cartpole", "20261017-003025-7bad"),
    ("boids", "20261017-141452-a065"),
    ("spiral", "20261017-132140-b4e4"),
    ("xor", "20261017-002703-340b"),
]
# the epoch the "epoch" shots pin, one the export kept a leaderboard and episode for
EPOCHS = {"f1tenth": 540, "slimevolley": 50, "cartpole": 50, "boids": 50, "spiral": 160, "xor": 50}
# how far into each episode it's paused
FRAME = 0.6
WIDE, NARROW = 1680, 1180

FREEZE = (HERE / "freeze.js").read_text()


# ---------------------------------------------------------------- servers


class Files(http.server.SimpleHTTPRequestHandler):
    """Serves a few directories under URL prefixes, with no caching."""

    mounts = {}

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        for prefix, root in self.mounts.items():
            if path.startswith(prefix):
                rest = path[len(prefix) :].lstrip("/")
                return str(Path(root) / rest)
        return str(Path("/nonexistent") / path.lstrip("/"))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def serve_files(mounts):
    handler = type("Mounted", (Files,), {"mounts": mounts})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def serve_monitor(runs):
    """The monitor from this checkout, on a folder of just the showcase runs."""
    folder = Path(tempfile.mkdtemp(prefix="shots-runs-"))
    for _, run_id in SHOWCASE:
        (folder / run_id).symlink_to(Path(runs).expanduser().resolve() / run_id)
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "monitor.server", "--dir", str(folder), "--port", str(port)],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    import urllib.request

    for _ in range(100):
        try:
            urllib.request.urlopen(f"{url}/api/runs", timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    return proc, url


def free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- the page


def settle(page, quiet=800, timeout=90):
    """Wait until nothing's been fetched or changed for `quiet` ms."""
    page.evaluate("document.fonts.ready")
    end = time.time() + timeout
    while time.time() < end:
        if page.evaluate(f"window.__quiet({quiet})"):
            return
        time.sleep(0.1)
    raise TimeoutError("the page never settled")


def fit_height(page, target):
    """Make the window tall enough that nothing scrolls, so a shot shows it all."""
    for _ in range(6):
        if target == "monitor":
            # what scrolls: the dashboard's main column, or before it filled the
            # window with the leaderboard beside it, the window's main area
            over = page.evaluate("(() => { const m = document.querySelector('.nd-main') || document.getElementById('main'); return m.scrollHeight - m.clientHeight; })()")
        else:
            over = page.evaluate("document.documentElement.scrollHeight - innerHeight")
        if over <= 0:
            return
        size = page.viewport_size
        page.set_viewport_size({"width": size["width"], "height": size["height"] + over})
        settle(page)


def pause_episodes(page):
    """Every episode on the page to FRAME of the way in, paused."""
    page.evaluate(
        """(at) => {
          for (const input of document.querySelectorAll('.player-controls input[type=range]')) {
            input.value = Math.round(Number(input.max) * at);
            input.dispatchEvent(new Event('input', { bubbles: true }));
          }
        }""",
        FRAME,
    )
    settle(page)


def pin_epoch(page, epoch):
    page.evaluate(
        """(epoch) => {
          const input = document.querySelector('.epoch-bar input[type=range]');
          input.value = epoch;
          input.dispatchEvent(new Event('input', { bubbles: true }));
        }""",
        epoch,
    )
    settle(page)


def open_top_member(page):
    page.locator(".board tbody tr.row").first.click()
    settle(page)


def shoot(page, target, path):
    """The dashboard: the whole window for the monitor, its own element for the site.
    Shot twice, until two shots in a row agree."""
    take = (lambda: page.screenshot()) if target == "monitor" else (lambda: page.locator(".neat-dash").first.screenshot())
    # painted afresh: what was painted while the page came together can leave a
    # shade's difference in the edges of a slider's knob
    page.evaluate("""async () => {
      const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      document.documentElement.style.visibility = "hidden";
      await frame();
      document.documentElement.style.visibility = "";
      await frame();
    }""")
    last = take()
    for _ in range(10):
        time.sleep(0.3)
        shot = take()
        if shot == last:
            path.write_bytes(shot)
            return
        last = shot
    raise RuntimeError(f"{path.name} never held still")


# ---------------------------------------------------------------- capture


def capture(args):
    from playwright.sync_api import sync_playwright

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stop = []
    if args.target == "monitor":
        proc, base = serve_monitor(args.runs)
        stop.append(proc.terminate)
        url = lambda env, run_id: f"{base}/run/{run_id}"
    elif args.target == "site":
        url = lambda env, run_id: f"{args.url}#{env}"
    else:
        mounts = {"/static/": REPO / "monitor" / "static", "/site/": HERE / "site", "/neat-data/": Path(args.data).expanduser()}
        if args.fork:
            mounts["/fork/"] = Path(args.fork).expanduser()
        server, base = serve_files(mounts)
        stop.append(server.shutdown)
        impl = "fork" if args.fork else "shared"
        url = lambda env, run_id: f"{base}/site/index.html?env={env}&impl={impl}"

    envs = [(e, r) for e, r in SHOWCASE if not args.only or e in args.only]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for env, run_id in envs:
                for shot in ["run", "narrow", "epoch", "member"]:
                    width = NARROW if shot == "narrow" else WIDE
                    context = browser.new_context(viewport={"width": width, "height": 1000}, device_scale_factor=1, timezone_id="UTC", locale="en-US")
                    context.add_init_script(FREEZE)
                    page = context.new_page()
                    page.goto(url(env, run_id))
                    page.wait_for_selector(".board tbody tr.row", timeout=120_000)
                    settle(page)
                    if shot == "epoch":
                        pin_epoch(page, EPOCHS[env])
                    if shot == "member":
                        open_top_member(page)
                    fit_height(page, args.target)
                    pause_episodes(page)
                    path = out / f"{env}-{shot}.png"
                    shoot(page, args.target, path)
                    print(f"{path.name}", file=sys.stderr)
                    context.close()
            browser.close()
    finally:
        for f in stop:
            f()


# ---------------------------------------------------------------- compare


def compare(args):
    import numpy as np
    from PIL import Image

    a, b = Path(args.a).expanduser(), Path(args.b).expanduser()
    diff_dir = Path(args.diff).expanduser() if args.diff else None
    names = sorted({p.name for p in a.glob("*.png")} | {p.name for p in b.glob("*.png")})
    bad = 0
    for name in names:
        if not (a / name).exists() or not (b / name).exists():
            print(f"{name:28} missing from {'first' if not (a / name).exists() else 'second'}")
            bad += 1
            continue
        x = np.asarray(Image.open(a / name).convert("RGB")).astype(int)
        y = np.asarray(Image.open(b / name).convert("RGB")).astype(int)
        if x.shape != y.shape:
            print(f"{name:28} size {x.shape[1]}x{x.shape[0]} vs {y.shape[1]}x{y.shape[0]}")
            bad += 1
            continue
        off = (x != y).any(axis=2)
        if not off.any():
            print(f"{name:28} same")
            continue
        bad += 1
        rows, cols = np.nonzero(off)
        print(f"{name:28} {off.sum()} pixels differ, in x {cols.min()}-{cols.max()}, y {rows.min()}-{rows.max()}")
        if diff_dir:
            diff_dir.mkdir(parents=True, exist_ok=True)
            # the second, faded, with what differs in red
            img = (255 - (255 - y) * 0.3).astype(np.uint8)
            img[off] = [220, 0, 0]
            Image.fromarray(img).save(diff_dir / name)
    print(f"{len(names) - bad} of {len(names)} the same")
    sys.exit(1 if bad else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture", help="shoot a target's views into --out")
    cap.add_argument("target", choices=["monitor", "site", "harness"])
    cap.add_argument("--out", required=True)
    cap.add_argument("--runs", default="~/turbo-neat-baselines/v2/runs", help="monitor: the baseline runs")
    cap.add_argument("--url", default="http://127.0.0.1:3100/neat", help="site: the /neat page")
    cap.add_argument("--data", default="~/turbo-neat-dash-baseline/neat-data", help="harness: the site's export of the runs")
    cap.add_argument("--fork", help="harness: the site's own dashboard (its app/neat/dashboard), to check the harness against the site")
    cap.add_argument("--only", nargs="*", help="just these environments")
    cmp = sub.add_parser("compare", help="compare two captures, pixel for pixel")
    cmp.add_argument("a")
    cmp.add_argument("b")
    cmp.add_argument("--diff", help="write what differs here, in red")
    args = parser.parse_args()
    capture(args) if args.cmd == "capture" else compare(args)


if __name__ == "__main__":
    main()
