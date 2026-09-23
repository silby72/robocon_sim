// 機体設定パネル -- スライダ in、omni_sim_core out。
//
// このファイルは物理を持たない。表示している数値はすべて Python から返って
// きたもので、pytest が走らせるのと同じ mechanism/derive と
// mechanism/jacobian の出力。ここの表示がおかしいときバグは表示側か core 側
// のどちらかで、どちらかは検証パネルが答える。
//
// 単位: Python へは SI、読み手へは mm / 度。既存 GUI に合わせてある。保存は
// しない -- リロードすれば既定値に戻るのがこのパネルの意図。

window.OmniChassisPanel = (function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  let started = false;
  let onStatus = function () {};

  const els = {};
  function bind() {
    ["preset", "gear", "gear-v", "radius", "radius-v", "mass", "mass-v",
     "kv", "kv-v", "v-diag", "v-axis", "climb", "climb-note", "push",
     "derived", "jac", "plan", "plan-caption", "provenance",
     "yaw-panel", "yaw-warning", "preset-hint",
     "nm-box", "nm-name", "nm-v", "nm-kv", "nm-istall", "nm-inl", "nm-j",
     "nm-cpr", "nm-r", "nm-tau", "nm-rpm", "nm-tau-led", "nm-tau-msg",
     "nm-rpm-led", "nm-rpm-msg", "nm-yaml-btn", "nm-out", "nm-yaml",
     "nm-path", "nm-copy", "nm-dl"].forEach(function (id) { els[id] = $(id); });
  }

  const fmt = (x, d) => Number(x).toFixed(d);

  /** A figure tile: big number, small unit. */
  function figure(el, value, unit) {
    el.textContent = value;
    const u = document.createElement("span");
    u.className = "u";
    u.textContent = unit;
    el.append(u);
  }

  /** One RawSlotTable-style row: name, value, unit, optional badge. */
  function row(tbody, label, value, unit, badge) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = label;
    if (badge) {
      const b = document.createElement("span");
      b.className = "badge est";
      b.textContent = badge;
      th.append(b);
    }
    const td = document.createElement("td");
    td.textContent = value;
    if (unit) {
      const u = document.createElement("span");
      u.className = "unit";
      u.textContent = unit;
      td.append(u);
    }
    tr.append(th, td);
    tbody.append(tr);
  }

  // ------------------------------------------------------------- plan view
  /**
   * Top view, drawn from the geometry Python returned rather than from the
   * slider values, so the picture cannot drift from the numbers beside it.
   */
  function drawPlan(geom) {
    const cv = els.plan;
    cv.dataset.geom = JSON.stringify(geom);   // so a resize can redraw it
    const dpr = window.devicePixelRatio || 1;
    const css = cv.clientWidth || 320;
    cv.width = Math.round(css * dpr);
    cv.height = Math.round(css * dpr);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const S = css;
    ctx.clearRect(0, 0, S, S);

    const span = geom.footprint_m * 1.45;          // a little air around it
    const px = (m) => (m / span) * S;
    // world +x right, +y up; canvas y is down, so flip
    const X = (x) => S / 2 + px(x);
    const Y = (y) => S / 2 - px(y);

    ctx.strokeStyle = "#3a3a3a";                   // 100 mm grid
    ctx.lineWidth = 1;
    for (let m = -1.0; m <= 1.0 + 1e-9; m += 0.1) {
      if (Math.abs(m) > span / 2) continue;
      ctx.beginPath(); ctx.moveTo(X(m), 0); ctx.lineTo(X(m), S); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, Y(m)); ctx.lineTo(S, Y(m)); ctx.stroke();
    }

    const h = geom.footprint_m / 2;
    ctx.strokeStyle = "#6a6a6a";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(X(-h), Y(h), px(geom.footprint_m), px(geom.footprint_m));

    ctx.strokeStyle = "#4a4a4a";                   // body axes, +x forward
    ctx.beginPath(); ctx.moveTo(X(-h), Y(0)); ctx.lineTo(X(h), Y(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(0), Y(-h)); ctx.lineTo(X(0), Y(h)); ctx.stroke();

    ctx.font = "11px 'Ubuntu Mono', monospace";
    geom.wheels.forEach(function (w) {
      const cx = X(w.x), cy = Y(w.y);
      const c = Math.cos(w.axis_rad), s = Math.sin(w.axis_rad);
      const rr = px(w.radius_m);

      ctx.save();                                  // the wheel, across its axis
      ctx.translate(cx, cy);
      ctx.rotate(-w.axis_rad);                     // canvas y is flipped
      ctx.strokeStyle = "#cbd2d9";
      ctx.lineWidth = 2;
      ctx.strokeRect(-rr, -rr * 0.42, rr * 2, rr * 0.84);
      ctx.restore();

      // drive-axis arrow: where this wheel pushes at positive torque
      const L = rr * 2.3;
      const ax = cx + c * L, ay = cy - s * L;
      ctx.strokeStyle = "#308280";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ax, ay); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - (c * 0.87 - s * 0.5) * 7, ay + (s * 0.87 + c * 0.5) * 7);
      ctx.lineTo(ax - (c * 0.87 + s * 0.5) * 7, ay + (s * 0.87 - c * 0.5) * 7);
      ctx.closePath();
      ctx.fillStyle = "#308280";
      ctx.fill();

      ctx.fillStyle = "#969696";
      ctx.textAlign = "center";
      ctx.fillText(w.id, cx, cy - rr - 7);
    });

    ctx.strokeStyle = "#969696";                   // 100 mm scale bar
    ctx.lineWidth = 1;
    const y0 = S - 16, x0 = 14;
    ctx.beginPath();
    ctx.moveTo(x0, y0); ctx.lineTo(x0 + px(0.1), y0);
    ctx.moveTo(x0, y0 - 3); ctx.lineTo(x0, y0 + 3);
    ctx.moveTo(x0 + px(0.1), y0 - 3); ctx.lineTo(x0 + px(0.1), y0 + 3);
    ctx.stroke();
    ctx.fillStyle = "#969696";
    ctx.textAlign = "left";
    ctx.fillText("100 mm", x0 + px(0.1) + 6, y0 + 4);
  }

  // ---------------------------------------------------------------- render
  const WHEEL_LABEL = { fl: "前左 fl", fr: "前右 fr", rl: "後左 rl", rr: "後右 rr" };

  function render(res, input) {
    const p = res.page, d = res.derived;

    figure(els["v-axis"], fmt(p.v_noload_axis_ms, 2), "m/s");
    figure(els["v-diag"], fmt(p.v_noload_diag_ms, 2), "m/s");
    figure(els.climb, fmt(p.climb_deg_at_stall, 1), "°");
    figure(els.push, fmt(p.push_stall_n, 0), "N");

    // 45 deg is where the planar model saturates, not a real limit -- say so
    // rather than letting the page imply the robot can climb a 45 deg ramp.
    els["climb-note"].textContent = p.climb_deg_at_stall >= 44.99
      ? "平面モデルの上限に飽和（実際は転倒・トラクションが先）"
      : "ランプ 9.9° に対して余裕 " + fmt(p.climb_deg_at_stall - 9.9, 1) + "°";

    const est = new Set(d.estimated);
    const t = els.derived;
    t.innerHTML = "";
    row(t, "トルク定数 Kt", fmt(d.torque_constant_nm_a, 6), "N·m/A",
        est.has("torque_constant_nm_a") ? "推定" : null);
    row(t, "逆起電力定数 Ke", fmt(d.back_emf_v_s, 6), "V·s/rad",
        est.has("back_emf_v_s") ? "推定" : null);
    row(t, "巻線抵抗 R", fmt(d.resistance_ohm, 4), "Ω",
        est.has("resistance_ohm") ? "推定" : null);
    row(t, "ストールトルク τ", fmt(d.stall_torque_nm, 4), "N·m");
    row(t, "無負荷回転数 ω", fmt(d.no_load_speed_rad_s, 3), "rad/s");
    row(t, "　〃", fmt(d.no_load_speed_rad_s * 60 / (2 * Math.PI), 0), "rpm");
    row(t, "ロータ慣性 J", d.rotor_inertia_kgm2.toExponential(4), "kg·m²",
        est.has("rotor_inertia_kgm2") ? "推定" : null);
    row(t, "反映慣性", d.reflected_inertia_kgm2.toExponential(4), "kg·m²");
    row(t, "うち負荷分", d.load_contribution_kgm2.toExponential(4), "kg·m²");
    row(t, "粘性摩擦 B", d.damping_nms.toExponential(4), "N·m·s",
        est.has("damping_nms") ? "推定" : null);

    const j = els.jac;
    j.innerHTML = "";
    res.drive_matrix.forEach(function (r, i) {
      const id = res.geometry.wheels[i].id;
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.textContent = WHEEL_LABEL[id] || id;
      tr.append(th);
      r.forEach(function (v) {
        const td = document.createElement("td");
        td.textContent = fmt(v, 3);
        tr.append(td);
      });
      const share = document.createElement("td");
      const s = p.yaw_authority[i];
      share.textContent = fmt(s, 3);
      if (s < 0.05) share.className = "bad";
      tr.append(share);
      j.append(tr);
    });

    // A well-formed X layout gives all four wheels the same yaw share. If
    // some are near zero the layout is inconsistent, and saying so here is
    // the only reason anyone would notice -- the jacobian stays invertible,
    // so the SingularConfigError guard never fires.
    const dead = p.yaw_authority
      .map(function (s, i) { return { s: s, id: res.geometry.wheels[i].id }; })
      .filter(function (w) { return w.s < 0.05; });
    els["yaw-panel"].hidden = dead.length === 0;
    if (dead.length) {
      els["yaw-warning"].textContent =
        dead.map(function (w) { return w.id; }).join(", ")
        + " の旋回寄与がほぼ 0。config/robot/chassis.yaml は駆動軸に対して"
        + "ホイール位置が 90° ずれており、旋回は残り 2 輪の逆回しだけで"
        + "作られている（コメントは fl を +x,+y 隅と書いているが、値は"
        + " [0.34, -0.34]）。ヤコビアンは可逆なままなので特異配置ガードには"
        + "掛からない。本パネルの範囲外なので設定は変更していない。";
    }

    drawPlan(res.geometry);
    els["plan-caption"].textContent =
      "上面図。外形 " + fmt(res.geometry.footprint_m * 1000, 0) + " mm 角、"
      + "ホイール半径 " + fmt(input.wheelRadiusM * 1000, 1) + " mm。"
      + "矢印は各輪の駆動軸。X 配置ゆえ軸方向が対角方向の √2 倍速い"
      + "（輪は自分の軸に沿う動きを全速で受けるので、対角方向のほうが"
      + "先に輪速上限に当たる）。";

    const h = window.OMNI_SIM_CORE.hashes;
    els.provenance.textContent =
      input.presetName + " プリセット + derive.py ("
      + h["omni_sim_core/mechanism/derive.py"] + ") / jacobian.py ("
      + h["omni_sim_core/mechanism/jacobian.py"] + ")。"
      + "抗力とマージンは含まない無負荷値で、mechanism/robot_config の "
      + "achievable_speed とは別物。";
  }


  // ==================================================================== //
  // 新規モータ
  //
  // A motor typed in here never touches the repo -- it is evaluated through
  // exactly the same Pyodide path as a shipped preset, and the only way it
  // becomes permanent is the YAML you save into config/presets/motors/.
  // That split is on purpose: trying a motor should cost nothing, and
  // committing one should be a deliberate file.
  // ==================================================================== //
  const CUSTOM = "__new__";

  /** Empty means "not given" -- not zero. The schema treats null as unknown
   *  and derive.py estimates it, which is a different thing from 0. */
  function optNum(el) {
    const s = String(el.value).trim();
    if (s === "") return null;
    const v = Number(s);
    return Number.isFinite(v) ? v : null;
  }
  function num(el, fallback) {
    const v = Number(el.value);
    return Number.isFinite(v) ? v : fallback;
  }

  function customDoc() {
    const ds = {
      rated_voltage_v: num(els["nm-v"], 24),
      kv_rpm_per_v: num(els["nm-kv"], 20),
      stall_torque_nm: num(els["nm-tau"], 1),
      stall_current_a: num(els["nm-istall"], 10),
      no_load_speed_rpm: num(els["nm-rpm"], 480),
      no_load_current_a: num(els["nm-inl"], 0.5),
      rotor_inertia_kgm2: optNum(els["nm-j"]),
      encoder_cpr: Math.max(1, Math.round(num(els["nm-cpr"], 8192))),
    };
    const doc = { datasheet: ds };
    const r = optNum(els["nm-r"]);
    if (r !== null) doc.physical = { resistance_ohm: r };
    return doc;
  }

  function isCustom() { return els.preset.value === CUSTOM; }

  /** The two declared-but-unread fields, against what the model computed.
   *  Green means the datasheet is self-consistent; red means one of three
   *  numbers is a transcription error and the simulator will use the other
   *  two regardless. */
  function showCrossChecks(declared) {
    const pairs = [
      ["nm-tau", declared.torque_ratio, "N·m", declared.stall_torque_nm,
       declared.stall_torque_nm * declared.torque_ratio, 0.05],
      ["nm-rpm", declared.speed_ratio, "rpm", declared.no_load_speed_rpm,
       declared.no_load_speed_rpm * declared.speed_ratio, 0.02],
    ];
    pairs.forEach(function (p) {
      const [id, ratio, unit, said, used, tol] = p;
      const ok = Math.abs(ratio - 1.0) <= tol;
      els[id + "-led"].className = "led " + (ok ? "on" : "bad");
      els[id + "-msg"].textContent = ok
        ? "一致（モデルは " + used.toFixed(unit === "rpm" ? 0 : 3) + " " + unit + " を使用）"
        : "不一致: 記載 " + said.toFixed(unit === "rpm" ? 0 : 3) + " に対しモデルは "
          + used.toFixed(unit === "rpm" ? 0 : 3) + " " + unit
          + "（" + ratio.toFixed(2) + "×）を使う";
      els[id + "-msg"].style.color = ok ? "#888" : "var(--danger)";
    });
  }

  function toYaml() {
    const name = (els["nm-name"].value || "my_motor")
      .trim().replace(/[^A-Za-z0-9_]/g, "_");
    const d = customDoc();
    const ds = d.datasheet;
    const j = ds.rotor_inertia_kgm2;
    const lines = [
      "# " + name + " -- FILL IN: manufacturer, part number, and which shaft",
      "# these numbers are referred to (rotor or gearbox output). A gearmotor",
      "# entry that mixes the two is the failure mode this comment exists to",
      "# prevent; see config/presets/motors/ak40_10_v3.yaml.",
      "# Entered in the omni_sim console chassis panel on "
        + new Date().toISOString().slice(0, 10) + ".",
      "datasheet:",
      "  rated_voltage_v: " + ds.rated_voltage_v,
      "  kv_rpm_per_v: " + ds.kv_rpm_per_v,
      "  stall_torque_nm: " + ds.stall_torque_nm
        + "       # cross-check only; derive.py recomputes Kt*(I_stall-I_nl)",
      "  stall_current_a: " + ds.stall_current_a,
      "  no_load_speed_rpm: " + ds.no_load_speed_rpm
        + "     # cross-check only; derive.py recomputes kv*V",
      "  no_load_current_a: " + ds.no_load_current_a,
      "  rotor_inertia_kgm2: " + (j === null
        ? "null    # unknown -> estimated (WARNING + ESTIMATED tag)" : j),
      "  encoder_cpr: " + ds.encoder_cpr,
    ];
    if (d.physical) {
      lines.push("", "physical:",
                 "  # A measured winding resistance overrides V/I_stall.",
                 "  resistance_ohm: " + d.physical.resistance_ohm);
    }
    return { name: name, text: lines.join("\n") + "\n" };
  }

  function wireNewMotor() {
    const o = document.createElement("option");
    o.value = CUSTOM;
    o.textContent = "＋ 新規モータ…";
    els.preset.append(o);

    els.preset.addEventListener("change", function () {
      els["nm-box"].hidden = !isCustom();
      els["preset-hint"].textContent = isCustom()
        ? "このページの中だけ。YAML を保存するまでリポジトリには入りません"
        : "config/presets/motors/ から";
    });

    ["nm-v", "nm-kv", "nm-istall", "nm-inl", "nm-j", "nm-cpr", "nm-r",
     "nm-tau", "nm-rpm"].forEach(function (id) {
      els[id].addEventListener("input", recompute);
    });

    els["nm-yaml-btn"].addEventListener("click", function () {
      const y = toYaml();
      els["nm-yaml"].textContent = y.text;
      els["nm-path"].textContent =
        "config/presets/motors/" + y.name + ".yaml として保存:";
      els["nm-out"].hidden = false;
    });

    els["nm-copy"].addEventListener("click", function () {
      navigator.clipboard.writeText(els["nm-yaml"].textContent).then(
        function () { onStatus("ok", "YAML をクリップボードにコピーしました"); },
        function () { onStatus("error", "コピーできません。手動で選択してください"); });
    });

    els["nm-dl"].addEventListener("click", function () {
      const y = toYaml();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([y.text], { type: "text/yaml" }));
      a.download = y.name + ".yaml";
      a.click();
      URL.revokeObjectURL(a.href);
      onStatus("ok", y.name + ".yaml をダウンロードしました"
        + " — config/presets/motors/ に置いて sync_web_core.sh");
    });
  }

  // ---------------------------------------------------------------- inputs
  function readInputs() {
    const kv = Number(els.kv.value);
    return {
      presetName: isCustom() ? "(新規)" : els.preset.value,
      presetDoc: isCustom() ? customDoc() : null,
      gearRatio: Number(els.gear.value),
      wheelRadiusM: Number(els.radius.value),
      totalMassKg: Number(els.mass.value),
      overrides: kv > 0 ? { kv_rpm_per_v: kv } : null,
    };
  }

  function showInputs() {
    els["gear-v"].textContent = fmt(els.gear.value, 2);
    els["radius-v"].textContent = fmt(els.radius.value * 1000, 1);
    els["mass-v"].textContent = fmt(els.mass.value, 1);
    const kv = Number(els.kv.value);
    els["kv-v"].textContent = kv > 0 ? fmt(kv, 1) : "—";
  }

  // One evaluation in flight at a time. A slider drag fires dozens of input
  // events and each call is a round trip into the interpreter; without this
  // the queue outlives the drag and the panel shows a stale answer last.
  let running = false, queued = false;

  async function recompute() {
    if (running) { queued = true; return; }
    running = true;
    const input = readInputs();
    try {
      const res = await OmniBridge.evaluate(input);
      render(res, input);
      if (res.declared) showCrossChecks(res.declared);
      onStatus("ok", "omni_sim_core 実行中 — 入力を変えると再計算");
    } catch (err) {
      onStatus("error", String(err.message || err));
    } finally {
      running = false;
      if (queued) { queued = false; recompute(); }
    }
  }

  function wire() {
    ["input", "change"].forEach(function (ev) {
      [els.gear, els.radius, els.mass, els.kv].forEach(function (el) {
        el.addEventListener(ev, function () { showInputs(); recompute(); });
      });
      els.preset.addEventListener(ev, recompute);
    });
    window.addEventListener("resize", function () {
      if (els.plan.dataset.geom) drawPlan(JSON.parse(els.plan.dataset.geom));
    });
    OmniBridge.presetNames().forEach(function (name) {
      const o = document.createElement("option");
      o.value = name;
      o.textContent = name;
      els.preset.append(o);
    });
    els.preset.value = "m3508_c620";
    wireNewMotor();
    showInputs();
  }

  return {
    /** Called by the shell the first time this panel is opened. */
    start: function (statusCb) {
      onStatus = statusCb || function () {};
      if (started) { onStatus("ok", "omni_sim_core 実行中"); return; }
      started = true;
      bind();
      wire();
      OmniBridge.init(function (stage, detail) {
        const msg = {
          pyodide: "Pyodide を CDN から取得中…",
          numpy: "numpy を取得中…",
          core: "omni_sim_core を展開中…",
          import: "mechanism/derive を import 中…",
        }[stage];
        if (msg) onStatus("busy", msg);
        else if (stage === "ready") onStatus("busy", "起動 " + detail + " — 計算中…");
      })
        .then(recompute)
        .catch(function (err) {
          onStatus("error", "起動に失敗: " + (err.message || err)
            + "（CDN に到達できない場合はオフラインです）");
        });
    },
    redraw: function () {
      if (started && els.plan && els.plan.dataset.geom) {
        drawPlan(JSON.parse(els.plan.dataset.geom));
      }
    },
  };
})();
