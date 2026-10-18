// Where the dashboard (dashboard.js) gets a run from. Two sources answer the same
// questions: liveSource, the run monitor's server, which works everything out from
// the run's folder as it's asked and keeps up with a run as it logs; and
// staticSource, a finished run exported as static files (python -m monitor.export)
// for a site with no server, where what the server would work out on demand was
// worked out by the export.
//
// A source:
//   run()                     the run: {id, name, project, status, created, updated,
//                             seed, step, num_generations, config}, and
//                             species_names if it knows them already
//   metrics(since)            {rows, next}: the metrics' rows from the since-th on
//   media(since)              {rows, next}: the media logged, likewise; each
//                             {key, step} and one of file (mediaUrl), episode
//                             (episodeUrl, with env and players), data or error
//   mediaUrl(entry)           a media file's URL
//   episodeUrl(entry)         a test episode's packed arrays (episodes.js loadArrays)
//   board({scope, limit, species, epoch})
//                             the leaderboard: {step, population, species, rows,
//                             species_names}, rows marked featured if they have a
//                             page of their own
//   member(id)                a member's record, or null if there's no such member
//   network(id, step)         its network in an epoch, or null
//   memberEpisodes(id, step)  the episodes it played in an epoch, or null:
//                             {step, env, episodes: [{index, seat, players}]}
//   memberEpisodeUrl(eps, ep) one of those episodes' packed arrays
//   descent(id)               its line of descent with each ancestor's network, or null
//   hasNetwork(id, step)      (optional) whether network(id, step) has one, if it
//                             can tell without asking
//   known(id)                 (optional) a member's record from anywhere it came by
//   lastOnly                  (optional) true if it keeps members' networks and
//                             episodes from their last epoch only

async function json(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json();
}

// the run monitor's server (monitor/server.py), for the run runId
export function liveSource(runId) {
  const api = (path) => json(`/api/runs/${runId}${path}`);
  // not logged (or not yet)
  const none = () => null;
  return {
    run: () => api(""),
    metrics: (since) => api(`/metrics?since=${since}`),
    media: (since) => api(`/media?since=${since}`),
    mediaUrl: (entry) => `/media/${runId}/${encodeURIComponent(entry.file)}`,
    episodeUrl: (entry) => `/api/runs/${runId}/episodes/${encodeURIComponent(entry.episode)}`,
    board({ scope, limit, species, epoch }) {
      let q = `scope=${scope}&limit=${limit}`;
      if (species != null) q += `&species=${species}`;
      if (epoch != null) q += `&epoch=${epoch}`;
      return api(`/leaderboard?${q}`);
    },
    member: (id) =>
      api(`/individuals/${id}`).catch((e) => {
        if (String(e.message).startsWith("404")) return null;
        throw e;
      }),
    network: (id, step) => api(`/individuals/${id}/network?step=${step}`).catch(none),
    memberEpisodes: (id, step) => api(`/individuals/${id}/episodes?step=${step}`).catch(none),
    memberEpisodeUrl: (eps, ep) => `/api/runs/${runId}/member_episodes/${eps.step}/${ep.index}`,
    descent: (id) => api(`/individuals/${id}/lineage_networks`).catch(none),
  };
}

// A run exported under `base` (monitor/export.py writes the layout):
//
//   run.json        meta, every epoch's metrics, the test episodes and species maps
//                   kept, the epochs with leaderboards (checkpoints), the members
//                   with pages (featured) and the networks kept (<id>-<step>)
//   boards.json     the leaderboard, alive and all time, at each checkpoint
//   episodes/<step>.bin, networks/<id>-<step>.json, members/<id>.json,
//   members/<id>-descent.json, member-episodes/<file>
//
// The leaderboard at an epoch is the one at the checkpoint at or before it. A member
// without a page is told from what's known of it (from a leaderboard, a pedigree, a
// list of offspring), marked partial, with the newest network kept for it.
export function staticSource(base) {
  const file = (path) => json(`${base}/${path}`);
  let loading = null;
  let loaded = null;
  const load = () =>
    (loading ??= Promise.all([file("run.json"), file("boards.json")]).then(([run, boards]) => {
      loaded = { run, boards, featured: new Set(run.featured), networks: new Set(run.networks) };
      return loaded;
    }));
  // every member whose record has come by
  const known = new Map();
  const remember = (n) => known.has(n.id) || known.set(n.id, n);
  // the featured members' pages, once fetched
  const pages = new Map();
  const page = async (id) => {
    if (!pages.has(id)) {
      try {
        pages.set(id, await file(`members/${id}.json`));
      } catch {
        return null;
      }
    }
    return pages.get(id);
  };
  const media = async () => {
    const { run } = await load();
    return [...run.episodes, ...run.species_pca.map((m) => ({ key: "species_pca", ...m }))];
  };

  return {
    async run() {
      const { run } = await load();
      return { ...run.meta, species_names: run.species_names };
    },
    async metrics(since) {
      const { run } = await load();
      return { rows: run.metrics.slice(since), next: run.metrics.length };
    },
    async media(since) {
      const rows = await media();
      return { rows: rows.slice(since), next: rows.length };
    },
    mediaUrl: () => null,
    episodeUrl: (entry) => `${base}/episodes/${entry.episode}`,
    async board({ scope, limit, species, epoch }) {
      const { run, boards, featured } = await load();
      const steps = run.checkpoints;
      const at = epoch ?? steps.at(-1);
      let step = steps[0];
      for (const s of steps) if (s <= at) step = s;
      const kept = boards[step][scope];
      kept.rows.forEach(remember);
      const rows = species == null ? kept.rows : kept.rows.filter((r) => r.species === species);
      return {
        ...kept,
        rows: rows.slice(0, limit).map((r) => ({ ...r, featured: featured.has(r.id) })),
        species_names: run.species_names,
      };
    },
    async member(id) {
      const { featured, networks } = await load();
      const data = featured.has(id) ? await page(id) : null;
      if (data) {
        for (const n of [...(data.pedigree || []), ...(data.offspring_list || [])]) remember(n);
        return data;
      }
      const record = known.get(id);
      if (!record) return null;
      const kept = [...networks].filter((k) => k.startsWith(`${id}-`)).map((k) => Number(k.split("-")[1]));
      return { ...record, partial: true, network_step: kept.length ? Math.max(...kept) : null };
    },
    async network(id, step) {
      const { networks } = await load();
      const key = `${id}-${step}`;
      if (!networks.has(key)) return null;
      try {
        return await file(`networks/${key}.json`);
      } catch {
        return null;
      }
    },
    // the episodes it played in its last epoch, whatever epoch's are asked for:
    // the only ones kept
    memberEpisodes: async (id) => (await page(id))?.episodes ?? null,
    memberEpisodeUrl: (eps) => `${base}/member-episodes/${eps.file}`,
    async descent(id) {
      if (!(await page(id))?.descent) return null;
      try {
        return await file(`members/${id}-descent.json`);
      } catch {
        return null;
      }
    },
    hasNetwork: (id, step) => Boolean(loaded?.networks.has(`${id}-${step}`)),
    lastOnly: true,
    known: (id) => known.get(id) ?? null,
  };
}
