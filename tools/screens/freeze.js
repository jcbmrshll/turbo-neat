// Run before any of the page's own scripts (shots.py adds it as an init script), so
// the dashboard draws the same picture every time: the clock stopped at one moment
// ("3m ago", a run's wall time), episodes that never advance on their own (every
// frame is drawn at the same instant), and Math.random fixed (the car a race opens
// on). It also keeps what shots.py needs to tell when the page has settled: the
// fetches in flight and when the page last changed, on the real clock.

(() => {
  const FROZEN = Date.parse("2026-10-18T12:00:00Z");
  const realNow = Date.now.bind(Date);
  const RealDate = Date;

  class FrozenDate extends RealDate {
    constructor(...args) {
      super(...(args.length ? args : [FROZEN]));
    }
    static now() {
      return FROZEN;
    }
  }
  window.Date = FrozenDate;

  // the instant every animation frame is drawn at
  const T = 1000;
  performance.now = () => T;
  const raf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => raf(() => cb(T));

  Math.random = () => 0.5;

  let inFlight = 0;
  let changed = realNow();
  const fetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    inFlight++;
    changed = realNow();
    try {
      return await fetch(...args);
    } finally {
      inFlight--;
      changed = realNow();
    }
  };
  new MutationObserver(() => (changed = realNow())).observe(document, {
    subtree: true,
    childList: true,
    attributes: true,
    characterData: true,
  });
  // whether nothing has been fetched or changed for `ms`
  window.__quiet = (ms) => inFlight === 0 && realNow() - changed >= ms;
})();
