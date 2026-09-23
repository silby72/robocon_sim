// 自動走行パネル -- rosbridge 経由のライブ sim。
//
// omni_sim_ros/web/dashboard.html から**そのまま**移設したもの。描画コード
// (drawWalls / lethalOverlay / drawRobot / drawParticles ...) は壁へのめり込み
// と多層フィールドの表示で何度も直した箇所なので、見た目の統合にあたっても
// ロジックには触れていない。変わったのは CSS と、どの DOM に載るかだけ。
//
// DOM の id は移設前と同じままにしてある。そうすれば差分が「移動」だけになり、
// レビューで挙動の変更と見分けがつく。
//
// このファイルは #panel-sim が存在するときだけ動く。シェル (shell.js) が
// パネルを初めて開いたときに OmniSimPanel.start() を呼ぶ。

window.OmniSimPanel = (function () {
  "use strict";
  let started = false;

  const $ = (id) => document.getElementById(id);
  let ros = null;
  let allTopics = [];          // [{name, type}]
  const subscribed = new Map(); // name -> {listener, card, times: []}

  function setStatus(state, text) {
    const el = $("status");
    el.className = state;
    $("statusText").textContent = text;
  }

  function connect() {
    const url = $("url").value.trim();
    setStatus("connecting", "接続中…");
    $("connectBtn").disabled = true;

    ros = new ROSLIB.Ros({ url });

    ros.on("connection", () => {
      setStatus("connected", url);
      $("connectBtn").disabled = false;
      $("connectBtn").textContent = "切断";
      refreshTopics();
      initMap();
    });
    ros.on("error", (e) => {
      setStatus("", "error: " + e);
      $("connectBtn").disabled = false;
    });
    ros.on("close", () => {
      setStatus("", "未接続");
      $("connectBtn").disabled = false;
      $("connectBtn").textContent = "接続";
      $("topicList").innerHTML = '<div class="empty">未接続</div>';
      allTopics = [];
    });
  }

  function disconnect() {
    if (ros) ros.close();
    ros = null;
  }

  $("connectBtn").onclick = () => {
    if (ros && ros.isConnected) disconnect();
    else connect();
  };

  function refreshTopics() {
    ros.getTopics(
      (result) => {
        const types = result.types;
        allTopics = result.topics
          .map((name, i) => ({ name, type: types[i] }))
          .sort((a, b) => a.name.localeCompare(b.name));
        renderTopicList();
      },
      (err) => setStatus("", "getTopics failed: " + err)
    );
  }

  function renderTopicList() {
    const filter = $("filter").value.trim().toLowerCase();
    const list = $("topicList");
    list.innerHTML = "";
    const shown = allTopics.filter((t) => t.name.toLowerCase().includes(filter));
    if (!shown.length) {
      list.innerHTML = '<div class="empty">該当なし</div>';
      return;
    }
    for (const t of shown) {
      const row = document.createElement("div");
      row.className = "topic-row" + (subscribed.has(t.name) ? " active" : "");
      row.innerHTML = `<span class="name">${t.name}</span><span class="type">${t.type.split("/").pop()}</span>`;
      row.onclick = () => toggleSubscribe(t.name, t.type);
      list.appendChild(row);
    }
  }

  $("filter").oninput = renderTopicList;

  function toggleSubscribe(name, type) {
    if (subscribed.has(name)) {
      unsubscribe(name);
    } else {
      subscribeTopic(name, type);
    }
    renderTopicList();
  }

  function subscribeTopic(name, type) {
    const emptyMsg = $("cards").querySelector(".empty");
    if (emptyMsg) emptyMsg.remove();

    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="head">
        <span class="name">${name}</span>
        <span class="hz" data-hz></span>
        <span class="close" title="unsubscribe">&times;</span>
      </div>
      <pre data-body>waiting for a message...</pre>`;
    card.querySelector(".close").onclick = () => { unsubscribe(name); renderTopicList(); };
    $("cards").appendChild(card);

    const listener = new ROSLIB.Topic({ ros, name, messageType: type });
    const state = { listener, card, times: [] };
    listener.subscribe((msg) => {
      const now = performance.now();
      state.times.push(now);
      if (state.times.length > 20) state.times.shift();
      card.querySelector("[data-body]").textContent = JSON.stringify(msg, null, 2);
      if (state.times.length > 1) {
        const dt = (state.times[state.times.length - 1] - state.times[0]) / 1000;
        const hz = (state.times.length - 1) / dt;
        card.querySelector("[data-hz]").textContent = hz.toFixed(1) + " Hz";
      }
    });
    subscribed.set(name, state);
  }

  function unsubscribe(name) {
    const state = subscribed.get(name);
    if (!state) return;
    state.listener.unsubscribe();
    state.card.remove();
    subscribed.delete(name);
    if (!subscribed.size) {
      $("cards").innerHTML = '<div class="empty">接続してから左のトピックをクリックすると購読します。</div>';
    }
  }

  // periodically refresh the topic list while connected (nodes may come and go)
  setInterval(() => { if (ros && ros.isConnected) refreshTopics(); }, 4000);

  // ---------------------------------------------------------------------
  // Map & Planning.
  //
  // Nothing about the field or the robot is hard-coded here any more. The node
  // publishes the whole scene on /sim/scene (omni_sim_core/ui/web_scene.py):
  // the painted zones come from field_layout_2027._zones(), the walls are the
  // real FieldSpec primitives, and the robot is the chassis from
  // config/robot/chassis.yaml -- the same file the GUI's chassis page edits and
  // the same geometry the node's collision check judges against.
  //
  // That last point is the reason for the rewrite. The robot used to be a 9 px
  // dot, which is the one shape that can never show why a 0.9 m body does not
  // fit through a 0.23 m gap; and a hand-copied field drifts from the spec
  // silently. Now both sides draw the same geometry, so "the screen says clear,
  // the sim says blocked" cannot happen without one of them being visibly wrong.
  // ---------------------------------------------------------------------
  let FIELD_MM = 11000;
  const rgba = (r, g, b, a) => `rgba(${r},${g},${b},${a})`;

  let mapReady = false;
  let robotPose = null;   // {x, y, theta} metres/rad -- ground truth
  let odomPose = null;    // what wheel odometry believes
  let mclPose = null;     // what the particle filter believes
  let particles = [];     // [[x, y, yaw], ...]
  let planPoints = [];    // [[x,y], ...] metres -- the A* polyline
  let trajPoints = [];    // [[x,y], ...] metres -- the C2 curve actually followed
  let lastGoal = null;    // {x, y} as clicked
  let planEnd = null;     // {x, y} where the plan actually ends (may differ)
  let goalPub = null;     // ROSLIB.Topic, set once connected
  let scene = null;       // /sim/scene, parsed
  let simState = null;    // /sim/state, parsed
  const lethalCache = {}; // level -> offscreen canvas of "everything fatal here"

  function worldToCanvas(xM, yM, canvas) {
    const s = canvas.width / (FIELD_MM / 1000);
    return [xM * s, canvas.height - yM * s];
  }
  function canvasToWorld(px, py, canvas) {
    const s = canvas.width / (FIELD_MM / 1000);
    return [px / s, (canvas.height - py) / s];
  }

  // --- static field ----------------------------------------------------
  function drawZones(ctx, canvas) {
    const toC = (xmm, ymm) => worldToCanvas(xmm / 1000, ymm / 1000, canvas);
    const zones = [...scene.zones].sort((a, b) => a.z - b.z);
    for (const zn of zones) {
      if (zn.kind === "rect") {
        const [x0, y0, x1, y1] = zn.geom;
        const [cx0, cy0] = toC(x0, y1), [cx1, cy1] = toC(x1, y0);
        ctx.fillStyle = zn.color;
        ctx.fillRect(cx0, cy0, cx1 - cx0, cy1 - cy0);
        ctx.strokeStyle = "rgba(0,0,0,0.18)";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(cx0, cy0, cx1 - cx0, cy1 - cy0);
        if (zn.label) drawLabel(ctx, (cx0 + cx1) / 2, (cy0 + cy1) / 2, zn.label);
      } else if (zn.kind === "circle") {
        const [xmm, ymm, dmm] = zn.geom;
        const [cx, cy] = toC(xmm, ymm);
        ctx.fillStyle = zn.color;
        ctx.beginPath(); ctx.arc(cx, cy, mmToPx(dmm / 2, canvas), 0, 2 * Math.PI); ctx.fill();
      } else if (zn.kind === "line") {
        const [x0, y0, x1, y1] = zn.geom;
        const [ax, ay] = toC(x0, y0), [bx, by] = toC(x1, y1);
        ctx.strokeStyle = zn.color; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      } else if (zn.kind === "outline") {
        const [x0, y0, x1, y1] = zn.geom;
        const [cx0, cy0] = toC(x0, y1), [cx1, cy1] = toC(x1, y0);
        ctx.strokeStyle = zn.color; ctx.lineWidth = 2.5;
        ctx.strokeRect(cx0, cy0, cx1 - cx0, cy1 - cy0);
      } else if (zn.kind === "text" && zn.label) {
        const [cx, cy] = toC(zn.geom[0], zn.geom[1]);
        drawLabel(ctx, cx, cy, zn.label);
      }
    }
  }

  function drawLabel(ctx, cx, cy, label) {
    ctx.fillStyle = "rgba(30,30,30,0.85)";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    label.split("\n").forEach((line, i, all) => {
      ctx.fillText(line, cx, cy + (i - (all.length - 1) / 2) * 12);
    });
  }

  function mmToPx(mm, canvas) { return (mm / 1000) * (canvas.width / (FIELD_MM / 1000)); }

  /** Hatching, not a solid fill: the L1 slab is a wall seen from the ground and
   *  covers a third of the field, so filling it would black out the very zone
   *  art that tells you where the robot is trying to go. Hatch says "wall" and
   *  still lets the floor through. */
  let hatchPattern = null;
  function wallHatch(ctx) {
    if (hatchPattern) return hatchPattern;
    const tile = document.createElement("canvas");
    tile.width = tile.height = 8;
    const t = tile.getContext("2d");
    t.strokeStyle = "rgba(60,40,40,0.55)";
    t.lineWidth = 2;
    t.beginPath(); t.moveTo(-2, 6); t.lineTo(6, -2);
    t.moveTo(2, 10); t.lineTo(10, 2);
    t.stroke();
    hatchPattern = ctx.createPattern(tile, "repeat");
    return hatchPattern;
  }

  /** Walls, drawn according to whether they are a wall *for the level the robot
   *  is on*. A primitive only obstructs a layer when it pokes into that layer's
   *  robot band, so the L1 barrier is solid from L1 and nothing at all from the
   *  ground -- exactly what walls[].blocks records (see slicer.slice_nav). */
  function drawWalls(ctx, canvas) {
    if (!scene.walls) return;
    const level = simState ? simState.level : "ground";
    for (const w of scene.walls) {
      const active = w.blocks.includes(level);
      ctx.beginPath();
      if (w.kind === "circle") {
        const [cx, cy] = worldToCanvas(w.center[0] / 1000, w.center[1] / 1000, canvas);
        ctx.arc(cx, cy, mmToPx(w.r, canvas), 0, 2 * Math.PI);
      } else {
        w.points.forEach(([x, y], i) => {
          const [cx, cy] = worldToCanvas(x / 1000, y / 1000, canvas);
          if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
        });
        ctx.closePath();
      }
      ctx.fillStyle = active ? wallHatch(ctx) : rgba(120, 120, 120, 0.10);
      ctx.fill();
      ctx.strokeStyle = active ? "rgba(120,30,30,0.9)" : rgba(140, 140, 140, 0.3);
      ctx.lineWidth = active ? 1.8 : 0.8;
      ctx.stroke();
    }
  }

  /** Everything that is fatal on ``level`` but is not a wall: off the slab is a
   *  fall, and a hole in the slab is a fall too. The ramps are then erased back
   *  out, because a connector is carved FREE on both layers it links -- that
   *  erase is the only reason a ground->L1 plan exists at all. */
  function lethalOverlay(level, canvas) {
    if (lethalCache[level]) return lethalCache[level];
    const layer = scene.layers && scene.layers[level];
    const off = document.createElement("canvas");
    off.width = canvas.width; off.height = canvas.height;
    if (!layer || !layer.extent) { lethalCache[level] = off; return off; }
    const c = off.getContext("2d");
    const toC = (xmm, ymm) => worldToCanvas(xmm / 1000, ymm / 1000, off);

    c.fillStyle = rgba(255, 95, 95, 0.16);
    const [ex0, ey0, ex1, ey1] = layer.extent;
    const [ax, ay] = toC(ex0, ey1), [bx, by] = toC(ex1, ey0);
    c.beginPath();
    c.rect(0, 0, off.width, off.height);
    c.rect(ax, ay, bx - ax, by - ay);
    c.fill("evenodd");                       // outside the slab = a fall
    for (const [hx0, hy0, hx1, hy1] of (layer.holes || [])) {
      const [hax, hay] = toC(hx0, hy1), [hbx, hby] = toC(hx1, hy0);
      c.fillRect(hax, hay, hbx - hax, hby - hay);
    }
    c.globalCompositeOperation = "destination-out";
    for (const conn of (scene.connectors || [])) {
      if (!conn.links.includes(level)) continue;
      const [cx0, cy0, cx1, cy1] = conn.rect;
      const [rax, ray] = toC(cx0, cy1), [rbx, rby] = toC(cx1, cy0);
      c.fillRect(rax, ray, rbx - rax, rby - ray);
    }
    lethalCache[level] = off;
    return off;
  }

  /** The edge of the floor the robot is standing on, plus the ramps that break
   *  through it. A 16 %-alpha wash over nine tenths of the field is easy to
   *  miss; the line where "floor" becomes "fall" is the thing worth being able
   *  to point at, and the ramp gaps in it are where a level change is possible
   *  at all. */
  function drawLevelEdge(ctx, canvas, level) {
    const toC = (xmm, ymm) => worldToCanvas(xmm / 1000, ymm / 1000, canvas);
    const layer = scene.layers && scene.layers[level];
    if (layer && layer.extent) {          // the ground has no edge to fall off
      const [x0, y0, x1, y1] = layer.extent;
      const [ax, ay] = toC(x0, y1), [bx, by] = toC(x1, y0);
      ctx.strokeStyle = "#ffaf00";
      ctx.lineWidth = 2;
      ctx.setLineDash([7, 5]);
      ctx.strokeRect(ax, ay, bx - ax, by - ay);
      ctx.setLineDash([]);
    }
    // ...but the ramps still matter there: from the ground they are the only
    // way up, so outline them on every layer they link, not just the raised
    // ones, and shade them along the climb so which way is uphill is visible
    // rather than something you infer from the rest of the field.
    for (const conn of (scene.connectors || [])) {
      if (!conn.links.includes(level)) continue;
      const [cx0, cy0, cx1, cy1] = conn.rect;
      const [rax, ray] = toC(cx0, cy1), [rbx, rby] = toC(cx1, cy0);
      if (conn.uphill) {
        const [ux, uy] = conn.uphill;
        const g = ctx.createLinearGradient(
          ux > 0 ? rax : rbx, uy > 0 ? rby : ray,
          ux > 0 ? rbx : rax, uy > 0 ? ray : rby);
        g.addColorStop(0, "rgba(0,215,95,0.05)");
        g.addColorStop(1, "rgba(0,215,95,0.35)");   // bright end is the top
        ctx.fillStyle = g;
        ctx.fillRect(rax, ray, rbx - rax, rby - ray);
      }
      ctx.strokeStyle = "#00d75f";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(rax, ray, rbx - rax, rby - ray);
    }
  }

  // --- the robot -------------------------------------------------------
  /** The configured chassis, at the right size, rotated to the right heading.
   *  Footprint and wheels come from chassis.yaml via /sim/scene, so this is the
   *  same rectangle the node's collision check rotates -- not an icon. */
  function drawRobot(ctx, canvas) {
    if (!robotPose) return;
    const [cx, cy] = worldToCanvas(robotPose.x, robotPose.y, canvas);
    const blocked = simState && simState.blocked;
    const body = scene && scene.robot;

    if (!body) {   // no chassis file -> say so rather than drawing a fake one
      ctx.fillStyle = "#ffaf00";
      ctx.beginPath(); ctx.arc(cx, cy, 6, 0, 2 * Math.PI); ctx.fill();
      return;
    }

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-robotPose.theta);            // canvas y is flipped, so is the sense

    // the circle A* inflates by: the gap between it and the body is the whole
    // spare clearance budget ("plan with the circle, judge with the rectangle")
    ctx.strokeStyle = rgba(95, 215, 255, 0.35);
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(0, 0, mmToPx(body.r_circ, canvas), 0, 2 * Math.PI); ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    body.footprint.forEach(([x, y], i) => {
      const px = mmToPx(x, canvas), py = -mmToPx(y, canvas);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fillStyle = blocked ? rgba(255, 95, 95, 0.55) : rgba(0, 215, 95, 0.45);
    ctx.fill();
    ctx.strokeStyle = blocked ? "#ff5f5f" : "#00d75f";
    ctx.lineWidth = 2;
    ctx.stroke();

    // drive wheels, each drawn along its own drive axis so the X-configuration
    // is readable rather than four identical blobs
    for (const w of body.wheels) {
      ctx.save();
      ctx.translate(mmToPx(w.pos[0], canvas), -mmToPx(w.pos[1], canvas));
      ctx.rotate(-w.axis);
      const len = mmToPx(w.radius * 2, canvas), wid = Math.max(mmToPx(w.radius * 0.8, canvas), 2);
      ctx.fillStyle = "#1c1c1c";
      ctx.strokeStyle = "#00afd7";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.rect(-len / 2, -wid / 2, len, wid);
      ctx.fill(); ctx.stroke();
      ctx.restore();
    }

    // centre of mass + heading
    ctx.fillStyle = "#ff5f5f";
    ctx.beginPath();
    ctx.arc(mmToPx(body.com[0], canvas), -mmToPx(body.com[1], canvas), 3, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = ctx.fillStyle = blocked ? "#ff5f5f" : "#00d75f";
    ctx.lineWidth = 2.5;
    const nose = mmToPx(body.size * 0.75, canvas);
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(nose, 0); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(nose, 0); ctx.lineTo(nose - 7, -5); ctx.lineTo(nose - 7, 5);
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  function drawPolyline(ctx, canvas, pts, color, width, dash) {
    if (pts.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    pts.forEach(([x, y], i) => {
      const [cx, cy] = worldToCanvas(x, y, canvas);
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawMap() {
    const canvas = $("mapCanvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!scene) return;

    const level = simState ? simState.level : "ground";
    drawZones(ctx, canvas);
    drawWalls(ctx, canvas);
    ctx.drawImage(lethalOverlay(level, canvas), 0, 0);
    drawLevelEdge(ctx, canvas, level);

    drawPolyline(ctx, canvas, planPoints, rgba(95, 215, 255, 0.45), 2, [6, 4]);
    drawPolyline(ctx, canvas, trajPoints, "#5fd7ff", 3);

    if (lastGoal) {
      const [cx, cy] = worldToCanvas(lastGoal.x, lastGoal.y, canvas);
      ctx.strokeStyle = "#ffaf00";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8); ctx.stroke();
      // where the robot will actually stop, when that is not where you clicked
      if (planEnd && Math.hypot(planEnd.x - lastGoal.x, planEnd.y - lastGoal.y) > 0.1) {
        const [ex, ey] = worldToCanvas(planEnd.x, planEnd.y, canvas);
        ctx.beginPath(); ctx.arc(ex, ey, 7, 0, 2 * Math.PI); ctx.stroke();
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    drawParticles(ctx, canvas);
    drawOdomGhost(ctx, canvas);
    drawMclGhost(ctx, canvas);
    drawRobot(ctx, canvas);
  }

  /** The particle cloud. Drawn first and faintly: it is the *shape* of the
   *  filter's belief, and a tight cloud that has converged on the wrong place
   *  is the failure this makes visible. */
  function drawParticles(ctx, canvas) {
    if (!particles.length) return;
    ctx.fillStyle = "rgba(95,215,255,0.5)";
    for (const [x, y] of particles) {
      const [cx, cy] = worldToCanvas(x, y, canvas);
      ctx.fillRect(cx - 1, cy - 1, 2, 2);
    }
  }

  /** The filter's estimate, as a solid outline. Cyan against the odometry
   *  ghost's orange, so which of the two is tracking the truth is readable at
   *  a glance rather than by reading numbers. */
  function drawMclGhost(ctx, canvas) {
    if (!mclPose || !scene || !scene.robot) return;
    const [cx, cy] = worldToCanvas(mclPose.x, mclPose.y, canvas);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-mclPose.theta);
    ctx.strokeStyle = "rgba(95,215,255,0.95)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    scene.robot.footprint.forEach(([x, y], i) => {
      const px = mmToPx(x, canvas), py = -mmToPx(y, canvas);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }

  /** The believed pose, dashed and hollow. Deliberately drawn *under* the real
   *  robot: when odometry is good the two coincide and the ghost is invisible,
   *  which is exactly the right amount of attention for a well-behaved
   *  estimator. */
  function drawOdomGhost(ctx, canvas) {
    if (!odomPose || !scene || !scene.robot) return;
    const [cx, cy] = worldToCanvas(odomPose.x, odomPose.y, canvas);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-odomPose.theta);
    ctx.strokeStyle = "rgba(255,175,0,0.85)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    scene.robot.footprint.forEach(([x, y], i) => {
      const px = mmToPx(x, canvas), py = -mmToPx(y, canvas);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  function renderRobotStatus() {
    if (!scene) return;
    const body = scene.robot;
    const parts = [];
    if (body) {
      parts.push(`<span class="key">chassis</span> ${body.size.toFixed(0)} mm ` +
                 `${body.shape}, ${body.wheels.length} wheels, ` +
                 `r_circ ${(body.r_circ / 1000).toFixed(3)} m`);
    } else {
      parts.push('<span class="bad">no chassis</span> -- pass chassis_yaml to sim_node ' +
                 '(collision judging is off too)');
    }
    if (simState) {
      const z = (simState.z !== undefined)
        ? ` &middot; <span class="key">z</span> ${(simState.z * 1000).toFixed(0)} mm`
          + (simState.on_slope
              ? ` <span class="bad">${simState.pitch.toFixed(1)}° on ${simState.on_slope}</span>`
              : "")
        : "";
      if (robotPose && odomPose) {
        const d = Math.hypot(robotPose.x - odomPose.x, robotPose.y - odomPose.y);
        const dth = Math.abs(((robotPose.theta - odomPose.theta) * 180 / Math.PI));
        let line = `<span class="key">odom</span> ${(d * 1000).toFixed(0)} mm` +
                   ` &middot; ${dth.toFixed(2)}\u00b0 <span class="key">(orange dashes)</span>`;
        if (mclPose) {
          const md = Math.hypot(robotPose.x - mclPose.x, robotPose.y - mclPose.y);
          const cls = md < d ? "ok" : "bad";
          line += `<br><span class="key">MCL</span> <span class="${cls}">` +
                  `${(md * 1000).toFixed(0)} mm</span> &middot; spread ` +
                  `${(mclPose.spread * 1000).toFixed(0)} mm ` +
                  `<span class="key">(cyan outline + cloud)</span>`;
        }
        parts.push(line);
      }
      parts.push(`<span class="key">level</span> <span class="lvl">${simState.level}</span>` + z +
                 (simState.blocked
                   ? ' <span class="bad">BLOCKED</span>'
                   : ' <span class="ok">clear</span>') +
                 ` &middot; ${simState.collision_steps} blocked steps` +
                 (simState.following ? " &middot; following" : ""));
    }
    $("robotStatus").innerHTML = parts.join("<br>");
    renderStates();
    renderMotorPicker();
  }

  // Two small state machines (omni_sim_core/ui/phase.py): what the executive is
  // doing, and where on the field the robot is. Both are drawn as the full set
  // of possible states with the current one lit and the ones already visited
  // kept bright -- a bare current value is a readout, the sequence is the trace,
  // and the trace is the part worth looking at.
  const CONTROL_STATES = ["idle", "planning", "following", "held", "blocked",
                          "arrived", "gave_up", "manual"];
  const MATCH_PHASES = ["start", "ground", "transfer", "storage", "ramp", "l1",
                        "stairs", "l2", "shared", "retry"];
  const BAD_STATES = new Set(["blocked", "gave_up"]);

  function renderChips(el, all, cur, seen) {
    el.innerHTML = "";
    for (const name of all) {
      const c = document.createElement("span");
      c.className = "chip" + (name === cur ? " on" : seen.has(name) ? " seen" : "")
                  + (name === cur && BAD_STATES.has(name) ? " bad" : "");
      c.textContent = name;
      el.appendChild(c);
    }
  }

  function renderStates() {
    const s = simState;
    const box = $("stateBox");
    if (!s || !s.control || !s.phase) { box.hidden = true; return; }
    box.hidden = false;
    renderChips($("ctrlChips"), CONTROL_STATES, s.control.current,
                new Set(s.control.history.map((h) => h[1])));
    renderChips($("phaseChips"), MATCH_PHASES, s.phase.current,
                new Set(s.phase.history.map((h) => h[1])));
    const trace = s.phase.history.map(([t, v]) => `${v}@${t.toFixed(1)}s`).join(" → ");
    $("trace").textContent = trace;
  }

  // Swapping the drive motor rebuilds the plant in the node and keeps the pose,
  // so the same manoeuvre can be re-run back to back on a different drivetrain.
  // The node does NOT write the choice back to config/robot/actuators.yaml --
  // this is for asking "would AK40s climb that ramp?", not for editing the robot.
  let motorPub = null;
  function renderMotorPicker() {
    const m = scene && scene.motors;
    const box = $("motorBox");
    if (!m || !m.available || !m.available.length) { box.hidden = true; return; }
    box.hidden = false;
    const sel = $("motorSel");
    if (sel.options.length !== m.available.length) {
      sel.innerHTML = "";
      for (const name of m.available) {
        const o = document.createElement("option");
        o.value = o.textContent = name;
        sel.appendChild(o);
      }
      sel.onchange = () => {
        if (!motorPub) return;
        $("motorNote").textContent = "再構築中…";
        motorPub.publish(new ROSLIB.Message({ data: sel.value }));
      };
    }
    if (document.activeElement !== sel) sel.value = m.current || "";
    const swapped = m.configured && m.current && m.current !== m.configured;
    $("motorNote").innerHTML =
      `Kt ${m.kt} Nm/A &middot; R ${m.resistance} Ω` +
      (swapped ? ` &middot; <span class="bad">not ${m.configured}</span>` : "");
  }

  function yawFromQuat(q) {
    return Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
  }

  function setPlanStatus(html) { $("planStatus").innerHTML = html; }

  function initMap() {
    if (mapReady) { drawMap(); return; }
    mapReady = true;

    // The scene is re-published every 2 s (not latched -- see sim_node), so a
    // page opened at any time fills in shortly after connecting.
    new ROSLIB.Topic({ ros, name: "/sim/scene", messageType: "std_msgs/msg/String" })
      .subscribe((msg) => {
        const next = JSON.parse(msg.data);
        const changed = JSON.stringify(next) !== JSON.stringify(scene);
        if (!changed) return;
        scene = next;
        FIELD_MM = scene.field.size;
        for (const k of Object.keys(lethalCache)) delete lethalCache[k];
        renderRobotStatus();
        drawMap();
      });

    new ROSLIB.Topic({ ros, name: "/sim/state", messageType: "std_msgs/msg/String" })
      .subscribe((msg) => {
        const next = JSON.parse(msg.data);
        const levelChanged = !simState || simState.level !== next.level;
        simState = next;
        renderRobotStatus();
        if (levelChanged) drawMap();
      });

    new ROSLIB.Topic({ ros, name: "/trajectory", messageType: "nav_msgs/msg/Path" })
      .subscribe((msg) => {
        trajPoints = msg.poses.map((p) => [p.pose.position.x, p.pose.position.y]);
        drawMap();
      });

    // What the robot *believes* its pose is, from wheel odometry. Drawn as a
    // ghost against the true pose: the gap between them is the whole reason an
    // odometry error model exists, and it is invisible in a number.
    new ROSLIB.Topic({ ros, name: "/odom", messageType: "nav_msgs/msg/Odometry" })
      .subscribe((msg) => {
        odomPose = {
          x: msg.pose.pose.position.x,
          y: msg.pose.pose.position.y,
          theta: yawFromQuat(msg.pose.pose.orientation),
        };
      });

    // The estimator's output and the cloud behind it. Both, because an
    // estimate without its spread cannot be judged: a confident wrong answer
    // and a hedged right one look identical as a single dot.
    new ROSLIB.Topic({ ros, name: "/localization/odom", messageType: "nav_msgs/msg/Odometry" })
      .subscribe((msg) => {
        mclPose = {
          x: msg.pose.pose.position.x,
          y: msg.pose.pose.position.y,
          theta: yawFromQuat(msg.pose.pose.orientation),
          spread: Math.sqrt(msg.pose.covariance[0] || 0),
        };
      });

    new ROSLIB.Topic({ ros, name: "/particle_cloud", messageType: "geometry_msgs/msg/PoseArray" })
      .subscribe((msg) => {
        particles = msg.poses.map((p) => [p.position.x, p.position.y,
                                          yawFromQuat(p.orientation)]);
      });

    new ROSLIB.Topic({ ros, name: "/ground_truth/odom", messageType: "nav_msgs/msg/Odometry" })
      .subscribe((msg) => {
        robotPose = {
          x: msg.pose.pose.position.x,
          y: msg.pose.pose.position.y,
          theta: yawFromQuat(msg.pose.pose.orientation),
        };
        drawMap();
      });

    new ROSLIB.Topic({ ros, name: "/plan", messageType: "nav_msgs/msg/Path" })
      .subscribe((msg) => {
        planPoints = msg.poses.map((p) => [p.pose.position.x, p.pose.position.y]);
        if (!planPoints.length) {
          setPlanStatus('<span class="bad">plan failed</span> -- see sim_node log for why');
        } else {
          let length = 0;
          for (let i = 1; i < planPoints.length; i++) {
            const [x0, y0] = planPoints[i - 1], [x1, y1] = planPoints[i];
            length += Math.hypot(x1 - x0, y1 - y0);
          }
          let msg = `<span class="ok">path found</span> -- ${planPoints.length} points, ${length.toFixed(2)} m`;
          // A* relocates a goal that lands inside the inflated region to the
          // nearest cell it can actually stand on. Without saying so, the robot
          // simply stops somewhere other than where you clicked and looks like
          // it gave up part-way -- which is exactly what it looked like.
          const end = planPoints[planPoints.length - 1];
          planEnd = { x: end[0], y: end[1] };
          if (lastGoal) {
            const shift = Math.hypot(end[0] - lastGoal.x, end[1] - lastGoal.y);
            if (shift > 0.1) {
              const infl = scene && scene.robot
                ? `the ${(scene.robot.r_circ / 1000).toFixed(2)} m inflation`
                : "the planner's inflation";
              msg += `<br><span class="bad">goal moved ${shift.toFixed(2)} m</span>` +
                     ` -- where you clicked is inside ${infl} of an obstacle,` +
                     ` so the robot will stop at the orange circle`;
            }
          }
          setPlanStatus(msg);
        }
        drawMap();
      });

    motorPub = new ROSLIB.Topic({ ros, name: "/sim/set_motor",
                                  messageType: "std_msgs/msg/String" });
    // Re-read config/robot/* without restarting the node. The GUI writes those
    // files; before this the only ways to pick an edit up were a restart or a
    // motor swap, which reloaded everything as a side effect.
    const reloadPub = new ROSLIB.Topic({ ros, name: "/sim/reload_robot",
                                         messageType: "std_msgs/msg/String" });
    $("reloadBtn").onclick = () => {
      $("motorNote").textContent = "リロード中…";
      reloadPub.publish(new ROSLIB.Message({ data: "" }));
    };
    goalPub = new ROSLIB.Topic({ ros, name: "/goal_pose", messageType: "geometry_msgs/msg/PoseStamped" });
    $("mapCanvas").onclick = (ev) => {
      const canvas = $("mapCanvas");
      const rect = canvas.getBoundingClientRect();
      const px = (ev.clientX - rect.left) * (canvas.width / rect.width);
      const py = (ev.clientY - rect.top) * (canvas.height / rect.height);
      sendGoal(...canvasToWorld(px, py, canvas));
    };

    drawMap();

    // ?goal=x,y auto-sends a goal once connected -- handy for scripted checks
    // (e.g. a headless-browser smoke test) without needing to simulate a click.
    const autoGoal = new URLSearchParams(location.search).get("goal");
    if (autoGoal) {
      const [gx, gy] = autoGoal.split(",").map(Number);
      if (Number.isFinite(gx) && Number.isFinite(gy)) sendGoal(gx, gy);
    }
  }

  function sendGoal(x, y) {
    lastGoal = { x, y };
    planEnd = null;
    setPlanStatus(`planning to (${x.toFixed(2)}, ${y.toFixed(2)}) ...`);
    goalPub.publish(new ROSLIB.Message({
      header: { frame_id: "map" },
      pose: { position: { x, y, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } },
    }));
    drawMap();
  }

  // connect automatically on load -- open the page, see topics, no click needed
  connect();

  // --- シェルとの接続点 ---------------------------------------------------
  // 元は読み込み直後に connect() を呼んでいた。シェルでは自動走行パネルが
  // 実際に開かれるまで rosbridge を叩かない -- 機体設定だけ見たい人に
  // 接続失敗の赤いステータスを見せる理由がない。
  return {
    start: function () {
      if (started) return;
      started = true;
      connect();
    },
    isStarted: function () { return started && ros !== null; },

    /** 停止ボタン: 全輪トルク 0 を一度だけ送る。
     *
     *  sim を止めるのではなく、走っている機体を止める。ros2can の zero_btn と
     *  同じ位置づけ。sim_node 側は次の cmd_vel/経路追従で上書きするので、
     *  これは「今すぐ抜く」であって非常停止ラッチではない -- ボタンの
     *  ツールチップもそう書いてある。 */
    publishZeroTorque: function () {
      if (!ros) return false;
      const n = (scene && scene.wheels && scene.wheels.length) || 4;
      new ROSLIB.Topic({
        ros: ros,
        name: "/sim/wheel_torque_cmd",
        messageType: "std_msgs/msg/Float64MultiArray",
      }).publish(new ROSLIB.Message({
        layout: { dim: [], data_offset: 0 },
        data: new Array(n).fill(0.0),
      }));
      return true;
    },
    disconnect: function () { if (started) disconnect(); },
    redraw: function () { if (typeof drawMap === "function") drawMap(); },
  };
})();
