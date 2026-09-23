// 検証パネル -- ブラウザの出力 = リポジトリの pytest が通している値か。
//
// 同じ入力を両側に与えて、出力が一致することを確かめる。比較するのは出力の
// **全フィールド**で、選んだ数個ではない。13 個のうち 3 個しか見ないテストは、
// 残り 10 個が間違っているページを通してしまう -- それがまさに、この仕組み
// 全体が捕まえるために存在する失敗の形。

window.OmniVerifyPanel = (function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  let started = false;

  function log(s) { $("log").textContent += s + "\n"; }

  /** Relative difference, falling back to absolute near zero. */
  function diff(a, b) {
    const scale = Math.max(Math.abs(a), Math.abs(b));
    return scale < 1e-12 ? Math.abs(a - b) : Math.abs(a - b) / scale;
  }

  /** Walk both trees together; every leaf must match, and the shapes too. */
  function compare(expected, actual, path, out, tol) {
    if (Array.isArray(expected)) {
      if (!Array.isArray(actual) || actual.length !== expected.length) {
        out.push({ path: path, why: "shape: expected length " + expected.length });
        return;
      }
      expected.forEach((v, i) => compare(v, actual[i], path + "[" + i + "]", out, tol));
    } else if (expected !== null && typeof expected === "object") {
      Object.keys(expected).forEach((k) => {
        if (!(k in actual)) { out.push({ path: path + "." + k, why: "missing" }); return; }
        compare(expected[k], actual[k], path + "." + k, out, tol);
      });
    } else if (typeof expected === "number") {
      const d = diff(expected, actual);
      if (!(d <= tol)) {
        out.push({ path: path,
                   why: expected + " vs " + actual + "  (rel " + d.toExponential(2) + ")" });
      }
    } else if (expected !== actual) {
      out.push({ path: path,
                 why: JSON.stringify(expected) + " vs " + JSON.stringify(actual) });
    }
  }

  function countLeaves(o) {
    if (Array.isArray(o)) return o.reduce((n, v) => n + countLeaves(v), 0);
    if (o !== null && typeof o === "object") {
      return Object.values(o).reduce((n, v) => n + countLeaves(v), 0);
    }
    return 1;
  }

  function line(kind, name, detail) {
    const div = document.createElement("div");
    div.className = "result-line";
    const led = document.createElement("span");
    led.className = "led " + (kind === "pass" ? "on" : "bad");
    const n = document.createElement("span");
    n.className = "name";
    n.textContent = name;
    const d = document.createElement("span");
    d.className = "detail";
    d.textContent = detail;
    div.append(led, n, d);
    $("results").append(div);
  }

  function timing(k, v, unit) {
    const f = document.createElement("div");
    f.className = "figure";
    const kd = document.createElement("div");
    kd.className = "k";
    kd.textContent = k;
    const vd = document.createElement("div");
    vd.className = "v";
    vd.textContent = v;
    const u = document.createElement("span");
    u.className = "u";
    u.textContent = unit;
    vd.append(u);
    f.append(kd, vd);
    $("timing").append(f);
  }

  async function run(onStatus) {
    const GOLD = window.OMNI_SIM_GOLDEN;
    const TOL = GOLD.tolerance;
    $("results").innerHTML = "";
    $("verdict").className = "verdict";
    $("verdict").textContent = "実行中…";

    log("golden source: " + GOLD.source);
    log("tolerance: " + TOL.toExponential(0) + " (relative)");
    log("bundle hashes:");
    Object.entries(window.OMNI_SIM_CORE.hashes)
      .forEach(([k, v]) => log("  " + v + "  " + k));

    onStatus("busy", "Pyodide を起動して照合中…");
    const py = await OmniBridge.init((stage, d) => log("[" + stage + "] " + d));
    log("python: " + py.runPython("import sys; sys.version").split("\n")[0]);
    log("pyodide: " + OmniBridge.version);

    let failures = 0, checked = 0;
    for (const c of GOLD.cases) {
      const inp = c.input;
      let actual;
      try {
        actual = await OmniBridge.evaluate({
          presetName: inp.preset,
          gearRatio: inp.gear_ratio,
          wheelRadiusM: inp.wheel_radius_m,
          totalMassKg: inp.total_mass_kg,
          overrides: Object.keys(inp.datasheet_overrides || {}).length
            ? inp.datasheet_overrides : null,
        });
      } catch (err) {
        failures++;
        line("fail", c.name, "例外: " + (err.message || err));
        continue;
      }
      const bad = [];
      compare(c.expected, actual, "", bad, TOL);
      checked += countLeaves(c.expected);
      if (bad.length) {
        failures++;
        line("fail", c.name, bad.length + " 件不一致");
        bad.slice(0, 8).forEach((b) => log("  " + c.name + b.path + ": " + b.why));
      } else {
        line("pass", c.name, countLeaves(c.expected) + " 値 — " + c.why);
      }
    }

    const t = OmniBridge.timings;
    $("timing").innerHTML = "";
    timing("合計", (t.total_ms / 1000).toFixed(2), "s");
    timing("Pyodide", (t.pyodide_ms / 1000).toFixed(2), "s");
    timing("numpy", (t.numpy_ms / 1000).toFixed(2), "s");
    timing("core 展開", t.install_ms.toFixed(1), "ms");
    timing("import", t.import_ms.toFixed(0), "ms");

    const v = $("verdict");
    if (failures) {
      v.className = "verdict fail";
      v.textContent = "FAIL — " + failures + " / " + GOLD.cases.length + " ケース不一致";
      onStatus("error", "検証 FAIL: " + failures + " ケース不一致");
      document.title = "FAIL — omni_sim console";
    } else {
      v.className = "verdict pass";
      v.textContent = "PASS — " + GOLD.cases.length + " ケース / " + checked
        + " 個の値が許容誤差 " + TOL.toExponential(0) + " 以内で一致";
      onStatus("ok", "検証 PASS — " + checked + " 値一致");
      document.title = "DONE — omni_sim console";
    }
    log("done: " + (failures ? failures + " failures" : "all passed"));
    return failures === 0;
  }

  return {
    /** Called by the shell the first time this panel is opened. */
    start: function (onStatus) {
      if (started) return;
      started = true;
      run(onStatus || function () {}).catch(function (e) {
        $("verdict").className = "verdict fail";
        $("verdict").textContent = "FAIL — " + (e.message || e);
        document.title = "FAIL — omni_sim console";
        log("ERROR " + (e.stack || e));
      });
    },
    rerun: function (onStatus) {
      started = true;
      return run(onStatus || function () {});
    },
  };
})();
