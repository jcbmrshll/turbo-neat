"""Screenshots of the run monitor's dashboard, to check that a change to it leaves
it looking exactly as it did.

The same views are shot of each run, with the clock, the episodes and Math.random
frozen (freeze.js), so two captures of an unchanged dashboard match pixel for pixel:

    run      the run's tab at the last epoch, scrolled to show all of it
    narrow   the same in a narrower window, where the layout changes
    epoch    the run's tab at the epoch halfway through
    member   the tab of the member at the top of the population

Every episode on the page is paused at the same frame (FRAME of the way in). Capture
before and after a change, then compare the two:

    tools/screens/shots capture --runs runs --out /tmp/shots/before RUN_ID ...
    tools/screens/shots capture --runs runs --out /tmp/shots/after RUN_ID ...
    tools/screens/shots compare /tmp/shots/before /tmp/shots/after --diff /tmp/shots/diff

compare prints each shot as `same` or where it differs, writes what differs in red
with --diff, and exits non-zero unless all are the same. Run one capture at a time.
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# how far into each episode it's paused
FRAME = 0.6
WIDE, NARROW = 1680, 1180

FREEZE = (HERE / "freeze.js").read_text()


# ---------------------------------------------------------------- servers


def serve_monitor(runs, run_ids):
    """The monitor from this checkout, on a folder of just these runs."""
    folder = Path(tempfile.mkdtemp(prefix="shots-runs-"))
    for run_id in run_ids:
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


def fit_height(page):
    """Make the window tall enough that nothing scrolls, so a shot shows it all."""
    for _ in range(6):
        # what scrolls: the dashboard's main column
        over = page.evaluate("(() => { const m = document.querySelector('.nd-main'); return m.scrollHeight - m.clientHeight; })()")
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


def pin_halfway(page):
    page.evaluate(
        """() => {
          const input = document.querySelector('.epoch-bar input[type=range]');
          input.value = Math.round((Number(input.min) + Number(input.max)) / 2);
          input.dispatchEvent(new Event('input', { bubbles: true }));
        }"""
    )
    settle(page)


def open_top_member(page):
    page.locator(".board tbody tr.row").first.click()
    settle(page)


def shoot(page, path):
    """The whole window, shot until two shots in a row agree."""
    take = page.screenshot
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
    proc, base = serve_monitor(args.runs, args.run_ids)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for run_id in args.run_ids:
                for shot in ["run", "narrow", "epoch", "member"]:
                    width = NARROW if shot == "narrow" else WIDE
                    context = browser.new_context(viewport={"width": width, "height": 1000}, device_scale_factor=1, timezone_id="UTC", locale="en-US")
                    context.add_init_script(FREEZE)
                    page = context.new_page()
                    page.goto(f"{base}/run/{run_id}")
                    page.wait_for_selector(".board tbody tr.row", timeout=120_000)
                    settle(page)
                    if shot == "epoch":
                        pin_halfway(page)
                    if shot == "member":
                        open_top_member(page)
                    fit_height(page)
                    pause_episodes(page)
                    path = out / f"{run_id}-{shot}.png"
                    shoot(page, path)
                    print(f"{path.name}", file=sys.stderr)
                    context.close()
            browser.close()
    finally:
        proc.terminate()


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
    cap = sub.add_parser("capture", help="shoot each run's views into --out")
    cap.add_argument("run_ids", nargs="+", metavar="RUN_ID")
    cap.add_argument("--runs", required=True, help="the folder the runs are in")
    cap.add_argument("--out", required=True)
    cmp = sub.add_parser("compare", help="compare two captures, pixel for pixel")
    cmp.add_argument("a")
    cmp.add_argument("b")
    cmp.add_argument("--diff", help="write what differs here, in red")
    args = parser.parse_args()
    capture(args) if args.cmd == "capture" else compare(args)


if __name__ == "__main__":
    main()
