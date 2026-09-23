// アプリシェル -- ros2can の MainWindow をブラウザに置き換えたもの。
//
//   QToolBar        上の帯（接続、設定リロード、停止、バージョン）
//   QSplitter       左のリスト + 右のスタック
//   QStackedWidget  パネル。表示中のものだけが生きている
//   QStatusBar      下の一行
//
// ros2can の SizedStackedWidget と同じ考え方で、**開かれるまでパネルを起動
// しない**。機体設定だけ見たい人に rosbridge の接続失敗を見せる理由はないし、
// 自動走行を見たい人に 8 MB の Pyodide を取りに行かせる理由もない。どちらの
// 系統も、相手が落ちていても動く。

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const opened = new Set();
  let active = null;

  // --- QStatusBar --------------------------------------------------------
  function setStatus(kind, text) {
    const msg = $("statusbar-msg");
    msg.textContent = text;
    msg.className = "msg" + (kind === "error" ? " danger" : "");
  }

  // --- 接続 LED ----------------------------------------------------------
  // panel-sim.js は移設前と同じく #status の class と #statusText を書く。
  // シェルはそれを監視して、ツールバーの LED・左のリスト・ステータスバーへ
  // 映す。こうすれば panel-sim 側に手を入れずに済み、差分が「移動」のまま
  // 残る。
  function watchConnection() {
    const status = $("status");
    const led = $("connLed");
    const note = $("nav-sim");
    if (!status) return;

    const sync = function () {
      const cls = status.className || "";
      const text = ($("statusText").textContent || "").trim();
      led.className = "led" + (cls.includes("connected") ? " on"
        : cls.includes("connecting") ? " warn" : " bad");
      note.textContent = cls.includes("connected") ? "接続"
        : cls.includes("connecting") ? "接続中" : "未接続";
      // The right-hand slot always carries the link state -- it is the one
      // thing worth knowing from any panel. The message on the left belongs
      // to whatever panel is in front, so only the sim panel writes it, and
      // only while it is the one showing. Without that guard every rosbridge
      // message (10 Hz) overwrote the chassis panel's own status.
      $("statusbar-right").textContent = "rosbridge: " + (text || "—");
      if (active === "page-sim") {
        setStatus(cls.includes("connected") ? "ok" : "warn",
          cls.includes("connected")
            ? "sim に接続。フィールドをクリックするとゴールを送ります。"
            : "sim に未接続 — " + (text || "…"));
      }
    };
    new MutationObserver(sync).observe(status,
      { attributes: true, childList: true, subtree: true, characterData: true });
    sync();
  }

  // --- QStackedWidget ----------------------------------------------------
  function show(pageId) {
    active = pageId;
    document.querySelectorAll(".page").forEach(function (p) {
      p.dataset.active = String(p.id === pageId);
    });
    document.querySelectorAll(".nav-item").forEach(function (b) {
      b.setAttribute("aria-current", String(b.dataset.page === pageId));
    });
    try { localStorage.setItem("omni.panel", pageId); } catch (e) { /* private mode */ }

    const first = !opened.has(pageId);
    opened.add(pageId);

    if (pageId === "page-sim") {
      if (first) {
        setStatus("", "sim に接続しています…");
        window.OmniSimPanel.start();
      }
      window.OmniSimPanel.redraw();
    } else if (pageId === "page-chassis") {
      if (first) setStatus("", "Pyodide を起動しています…");
      window.OmniChassisPanel.start(setStatus);
      window.OmniChassisPanel.redraw();
    } else if (pageId === "page-verify") {
      if (first) {
        $("nav-verify").textContent = "実行中";
        window.OmniVerifyPanel.start(function (kind, text) {
          setStatus(kind, text);
          $("nav-verify").textContent =
            kind === "ok" ? "PASS" : kind === "error" ? "FAIL" : "実行中";
        });
      }
    } else if (pageId === "page-about") {
      setStatus("", "構成");
    }
  }

  // --- 構成パネルの中身 --------------------------------------------------
  function fillAbout() {
    const t = $("bundle");
    const core = window.OMNI_SIM_CORE;
    Object.entries(core.hashes).forEach(function ([rel, h]) {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.textContent = rel;
      const td = document.createElement("td");
      td.textContent = h;
      tr.append(th, td);
      t.append(tr);
    });
    [["モータプリセット", Object.keys(core.presets).sort().join("  ")],
     ["機体", core.chassis.drive_wheels.length + " 輪 / 外形 "
              + (core.chassis.footprint.size_m * 1000).toFixed(0) + " mm 角"]]
      .forEach(function ([k, v]) {
        const tr = document.createElement("tr");
        const th = document.createElement("th");
        th.textContent = k;
        const td = document.createElement("td");
        td.textContent = v;
        tr.append(th, td);
        t.append(tr);
      });

    const showSW = function () {
      $("sw-state").textContent = window.OMNI_SW_STATE || "—";
    };
    showSW();
    document.addEventListener("omni-sw", showSW);
  }

  // --- 停止ボタン --------------------------------------------------------
  // ros2can の zero_btn と同じ役割。全輪トルク 0 を一度送るだけで、sim を
  // 止めはしない -- 走っている機体を止める最短の手段がひとつ要る。
  function wireEstop() {
    $("estopBtn").addEventListener("click", function () {
      const ok = window.OmniSimPanel.isStarted()
        && window.OmniSimPanel.publishZeroTorque();
      setStatus(ok ? "ok" : "error",
        ok ? "全輪トルク 0 を送信しました。"
           : "停止を送れません（sim に未接続）。");
    });
  }

  // --- boot --------------------------------------------------------------
  document.querySelectorAll(".nav-item").forEach(function (b) {
    b.addEventListener("click", function () { show(b.dataset.page); });
  });

  $("version").textContent = "core " +
    window.OMNI_SIM_CORE.hashes["omni_sim_core/mechanism/derive.py"].slice(0, 8);

  fillAbout();
  watchConnection();
  wireEstop();

  let start = "page-sim";
  try { start = localStorage.getItem("omni.panel") || start; } catch (e) { /* ok */ }
  if (!document.getElementById(start)) start = "page-sim";
  show(start);
})();
