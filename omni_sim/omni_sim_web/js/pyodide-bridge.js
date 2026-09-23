// Pyodide bridge: get omni_sim_core running in the browser, unmodified.
//
// The whole design is driven by one constraint -- no server, no build step, so
// the page is opened with file://. Chrome treats that origin as opaque, which
// blocks fetch() of a sibling file *and* `<script type="module">`. So:
//
//   * every script here is a classic script attaching to one global
//   * Python sources     -> core-bundle.js  (a <script>, which is allowed)
//   * motor presets      -> core-bundle.js  (YAML converted to JSON at sync
//                                            time, so no YAML parser ships)
//   * golden values      -> golden.js
//   * Pyodide and numpy  -> the CDN over https, which a file:// page may do
//
// Nothing is read from disk at run time. The modules are written into
// Pyodide's in-memory filesystem from the bundle and imported from there, so
// what runs in the browser is the same bytes pytest runs.

window.OmniBridge = (function () {
  "use strict";

  const PYODIDE_VERSION = "0.26.4";
  const PYODIDE_URL =
    "https://cdn.jsdelivr.net/pyodide/v" + PYODIDE_VERSION + "/full/";

  // One shared instance, one shared promise: callers arriving during startup
  // wait on the same load instead of racing to start a second interpreter.
  let ready = null;
  const timings = {};

  function mark(name, t0) {
    timings[name] = performance.now() - t0;
    return timings[name];
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = function () {
        reject(new Error("could not load " + src + " (offline?)"));
      };
      document.head.appendChild(s);
    });
  }

  /** Write the bundled sources into Pyodide's FS and put them on sys.path. */
  function installCore(pyodide) {
    const root = "/omni";
    const FS = pyodide.FS;
    const modules = window.OMNI_SIM_CORE.modules;
    Object.keys(modules).forEach(function (rel) {
      const path = root + "/" + rel;
      FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
      FS.writeFile(path, modules[rel], { encoding: "utf8" });
    });
    // sys.path rather than a copy into site-packages: if a stale omni_sim_core
    // were ever present, this makes which one wins explicit, not accidental.
    pyodide.runPython("import sys; sys.path.insert(0, " +
                      JSON.stringify(root) + ")");
  }

  /**
   * Boot Pyodide, install the copied core, and resolve to the interpreter.
   * @param {(stage: string, detail: string) => void} [onProgress]
   */
  function init(onProgress) {
    const report = onProgress || function () {};
    if (ready) return ready;
    ready = (async function () {
      const tStart = performance.now();

      if (!window.OMNI_SIM_CORE) {
        throw new Error(
          "core-bundle.js did not load -- run scripts/sync_web_core.sh");
      }
      if (!window.OMNI_WEB_API_PY) {
        throw new Error("web-api.js did not load");
      }

      report("pyodide", "fetching the interpreter");
      let t = performance.now();
      await loadScript(PYODIDE_URL + "pyodide.js");
      const pyodide = await window.loadPyodide({ indexURL: PYODIDE_URL });
      mark("pyodide_ms", t);

      report("numpy", "fetching numpy");
      t = performance.now();
      await pyodide.loadPackage("numpy", { messageCallback: function () {} });
      mark("numpy_ms", t);

      report("core", "installing omni_sim_core");
      t = performance.now();
      installCore(pyodide);
      mark("install_ms", t);

      report("import", "importing mechanism/derive");
      t = performance.now();
      await pyodide.runPythonAsync(window.OMNI_WEB_API_PY);
      mark("import_ms", t);

      mark("total_ms", tStart);
      report("ready", (timings.total_ms / 1000).toFixed(1) + " s");
      return pyodide;
    })();
    return ready;
  }

  /** A named motor preset, straight out of the bundle (already JSON). */
  function preset(name) {
    const p = window.OMNI_SIM_CORE.presets[name];
    if (!p) throw new Error("no such motor preset: " + name);
    return p;
  }

  function presetNames() {
    return Object.keys(window.OMNI_SIM_CORE.presets).sort();
  }

  /**
   * Evaluate one chassis. JSON crosses the boundary both ways, so no PyProxy
   * escapes into JS and there is nothing left to destroy but the function.
   */
  async function evaluate(opts) {
    const pyodide = await init();
    // presetDoc wins when given: that is a motor being typed into the page,
    // which does not exist in the bundle and may never become a file.
    const payload = JSON.stringify({
      preset: opts.presetDoc || preset(opts.presetName),
      chassis: window.OMNI_SIM_CORE.chassis,
      gear_ratio: opts.gearRatio,
      wheel_radius_m: opts.wheelRadiusM,
      total_mass_kg: opts.totalMassKg,
      datasheet_overrides: opts.overrides || null,
    });
    const fn = pyodide.globals.get("evaluate_json");
    try {
      const answer = JSON.parse(fn(payload));
      if (!answer.ok) throw new Error(answer.error);
      return answer.result;
    } finally {
      fn.destroy();
    }
  }

  /** Step 3 of the brief: one numpy array across the bridge, nothing else. */
  async function smoke() {
    const pyodide = await init();
    const fn = pyodide.globals.get("smoke");
    try {
      return JSON.parse(fn());
    } finally {
      fn.destroy();
    }
  }

  return {
    init: init,
    evaluate: evaluate,
    smoke: smoke,
    preset: preset,
    presetNames: presetNames,
    chassis: function () { return window.OMNI_SIM_CORE.chassis; },
    timings: timings,
    version: PYODIDE_VERSION,
  };
})();
