# Dashboard screenshots

The NEAT dashboard is drawn in two places: the run monitor (`monitor/static`) and
the /neat showcase on lowrollr.github.io. `shots.py` shoots the same views of both,
so a change to the dashboard can be checked against a baseline pixel for pixel.

For each environment's showcase run it shoots the run's tab at the last epoch
(`run`), the same in a narrower window (`narrow`), the run's tab at an epoch partway
through (`epoch`), and the tab of the member at the top of the population
(`member`): 24 shots a target. `freeze.js` stops the clock, the episodes and
`Math.random` first, and every episode is paused at the same frame, so an unchanged
dashboard shoots the same every time; the page is painted afresh before each shot,
as what was painted while it came together can otherwise leave a shade's difference
in the edge of a slider's knob. Shoot one target at a time.

`./shots` runs `shots.py` with what it needs (Playwright, its browser pinned).

## Targets

- `monitor`: the monitor from this checkout, serving the showcase runs from
  `~/turbo-neat-baselines/v2/runs` (`--runs`). The shot is the whole window.
- `site`: the /neat page of a running build of the site (`--url`). The shot is
  the dashboard's element, `.neat-dash`.
- `harness`: `site/index.html`, the dashboard in a page laid out like /neat (the
  site's `globals.css`, its fonts, its intro and environment tabs), reading the
  site's export of the runs (`--data`). It mounts this checkout's dashboard with
  the site's options (`site/preset.js`) and stylesheets: `site/neat.css`, the
  page around the dashboard as the site is to keep it, then `dashboard.css`.
  With `--fork <site>/app/neat/dashboard` it mounts the site's own dashboard
  with its own stylesheet (`site/neat-388a7e5.css`, verbatim) instead, which
  shoots exactly as the site does: that's what makes the harness a stand-in for
  the site.

## The baseline

In `~/turbo-neat-dash-baseline` (outside the repo, the shots are ~19 MB):

- `neat-data/`: the showcase runs exported for the site, byte for byte what it
  serves from GCS. Made by its `scripts/export_neat.py` at 388a7e5; now
  `python -m monitor.export` makes the same files:

      uv run --no-project --with numpy python -m monitor.export \
          --runs ~/turbo-neat-baselines/v2/runs --out ~/turbo-neat-dash-baseline/neat-data \
          cartpole=20261017-003025-7bad f1tenth=20261017-002705-b595 \
          spiral=20261017-132140-b4e4 xor=20261017-002703-340b \
          boids=20261017-141452-a065 slimevolley=20261017-121215-59bf
- `shots/monitor/`: `./shots capture monitor` at turbo-neat f7837d6.
- `shots/site/`: `./shots capture site` of lowrollr.github.io 388a7e5, built
  with its data read from `neat-data`:

      git -C ~/repos/lowrollr.github.io worktree add --detach /tmp/site 388a7e5
      cd /tmp/site && npm ci
      ln -s ~/turbo-neat-dash-baseline/neat-data public/neat-data
      NEXT_PUBLIC_NEAT_DATA_URL=/neat-data npx next build
      python3 -m http.server 3100 --bind 127.0.0.1 -d out
      tools/screens/shots capture site --url http://127.0.0.1:3100/neat.html --out ...

  `./shots capture harness --fork /tmp/site/app/neat/dashboard` matches it, 24 of 24.

## Checking a change

    tools/screens/shots capture monitor --out /tmp/shots/monitor
    tools/screens/shots compare ~/turbo-neat-dash-baseline/shots/monitor /tmp/shots/monitor --diff /tmp/shots/monitor-diff

and the same with `capture harness` against `shots/site`. `compare` prints each
shot as `same` or where it differs, writes what differs in red with `--diff`, and
exits non-zero unless all are the same. `--only f1tenth cartpole` shoots just those
environments.
