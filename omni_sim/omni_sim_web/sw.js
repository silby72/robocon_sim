// Offline cache for the browser pages.
//
// Read this before relying on it: **a service worker cannot register from
// file://**. The API requires a secure origin (https, or localhost), and the
// file scheme is not one -- navigator.serviceWorker is simply absent there.
//
// That collides head-on with the no-server rule. The two cannot both hold:
// opening index.html directly is what makes the page serverless, and it is
// also what makes offline caching unavailable. So this worker is written for
// the hosted case and skipped entirely on file://, where the browser's own
// HTTP cache is the only thing keeping the 6 MB of Pyodide and numpy from
// being fetched twice. See js/register-sw.js for the guard.
//
// Bump CACHE when anything under omni_sim_web/ changes, or a visitor keeps
// running yesterday's bundle against today's golden values.

const CACHE = "omni-sim-web-v2";

// The page's own files. Listed rather than globbed: there is no build step to
// generate a manifest, and a wrong glob would cache a stale file silently.
const LOCAL = [
  "./",
  "./index.html",
  "./css/style.css",
  "./js/vendor/roslib.min.js",
  "./js/core-bundle.js",
  "./js/golden.js",
  "./js/web-api.js",
  "./js/pyodide-bridge.js",
  "./js/register-sw.js",
  "./js/panel-sim.js",
  "./js/panel-chassis.js",
  "./js/panel-verify.js",
  "./js/shell.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(LOCAL)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  const isPyodide = url.hostname === "cdn.jsdelivr.net";

  // Local files: network first, so a re-synced bundle is picked up without
  // anyone having to remember to hard-reload. Cache is the offline fallback.
  //
  // Pyodide and numpy: cache first. They are ~6 MB, immutable at a pinned
  // version, and re-fetching them on every visit is the single biggest thing
  // between opening the page and seeing a number.
  if (isPyodide) {
    event.respondWith(
      caches.match(event.request).then((hit) => hit || fetch(event.request)
        .then((res) => {
          if (res.ok || res.type === "opaque") {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put(event.request, copy));
          }
          return res;
        }))
    );
    return;
  }

  if (url.origin !== self.location.origin) return;
  event.respondWith(
    fetch(event.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(event.request, copy));
        return res;
      })
      .catch(() => caches.match(event.request))
  );
});
