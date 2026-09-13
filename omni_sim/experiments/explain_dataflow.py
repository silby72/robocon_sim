#!/usr/bin/env python3
"""omni_sim の入力→過程→出力を「説明書」として吐く小スクリプト.

制御を理解する/紹介するための1枚図と、テキスト版マニュアルを生成する。
内容はコードの現状仕様に一致（トピック名・ファイル名・信号名は sim_node.py /
run.py / config/ から取った実物）。

    python experiments/explain_dataflow.py            # results/ に .txt と .png
    python experiments/explain_dataflow.py --no-png   # テキストだけ（matplotlib不要）

系の記号:  [物] 物理定数  [機] 機体仕様  [制] 制御設定  [環] 環境  [R] ROS
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

# --------------------------------------------------------------------------- #
# データ（1か所にまとめておき、テキストと図で共有する）
# --------------------------------------------------------------------------- #
INPUTS = [
    ("[物] 真のモータ", "config/plant_true.yaml", "J,B,Kt,R,L,Ke, 摩擦, コギング", "物"),
    ("[物] アクチュエータ", "config/robot/actuators.yaml", "M3508 データシート (+preset)", "物"),
    ("[機] 機体仕様", "config/robot/chassis.yaml", "0.9m / 15kg / 4輪 位置・角\n半径0.0508 減速比6", "機"),
    ("[制] 制御・実験条件", "config/scenarios/*.yaml", "PIDゲイン, DOB tau_q/次数,\nclock, 外乱, seed, 時間", "制"),
    ("[環] 環境地図", "地図PGM+YAML / field/robocon2027.yaml", "占有格子（ルール寸法から）", "環"),
]

# 過程チェーン: (見出し, 補足)
CHAIN = [
    ("PID", "目標ω→tau_pid"),
    ("− DOB補償", "tau_cmd = tau_pid − d_hat"),
    ("真のプラント\nMotorArray", "摩擦/コギング/飽和つき『本物』"),
    ("f=τ/r  →  Jacobian J^T", "車輪力→機体レンチ"),
    ("RigidBody 3DOF\nRK4 積分", "真値 pose / twist"),
]

OUTPUTS_OFFLINE = ("CSV (run.py)",
                   "t, omega_true, omega_ref,\ntau_cmd, tau_pid, d_hat, d_true",
                   "→ plot_results / sweep / 解析")
OUTPUTS_ROS = ("ROS トピック (sim_node.py)",
               "/scan(LaserScan)  /imu/data_raw(Imu)\n/joint_states  /odom + TF\n"
               "/ground_truth/odom  /clock",
               "→ RViz2 / nav2 / 実機ROSスタック")
OUTPUTS_MAP = ("地図 / 評価 / スイープ",
               "PGM+YAML地図,  results/<ts>/,\n境界CSV",
               "→ RViz2 / 設計判断")

REAL_SYSTEMS = [
    ("ROS", "sim_node は /cmd_vel を受け /odom・/scan・/imu を出す標準トピック。\n"
            "実機スタックの下に差し込む『ダミー実機』になる。"),
    ("serial_bridge", "実機のトルク指令↔CANの座席。sim では /sim/wheel_torque_cmd が\n"
                      "そのアナログ（コンソールUIも serial2can 風）。上の制御を実機なしで検証。"),
    ("Gazebo", "直接は未接続。役割違いの並行選択肢。同じROSトピック語彙・同じ地図形式\n"
               "なので nav2 の下で sim_node ↔ Gazebo は差し替え可能。"),
]

SIGNAL_MAP = [
    ("フィードバックの効き", "omega_ref  vs  omega_true"),
    ("PID の出力", "tau_pid"),
    ("外乱とその推定", "d_true  vs  d_hat （差＝DOBの誤差）"),
    ("DOB の価値", "tau_q/次数を振ると d_true−d_hat が縮む（P4スイープ）"),
    ("モデル誤差の影響", "nominal_error の J比 / B比 を振る"),
    ("離散化・遅延の影響", "dt_motor を変える（ZOHの粗さ）"),
    ("オドメトリ drift", "真値 pose − odom_pose （Encoder量子化由来）"),
]

RUNTIME_IN = "実行時コマンド[R]: /cmd_vel, /sim/wheel_torque_cmd, /sim/disturbance"


# --------------------------------------------------------------------------- #
# テキスト版マニュアル
# --------------------------------------------------------------------------- #
def text_manual() -> str:
    L = []
    W = 78
    L.append("=" * W)
    L.append("  omni_sim  取扱説明書 :  入力 → 過程 → 出力  （制御を理解する地図）")
    L.append("=" * W)
    L.append("")
    L.append("動かし口は2つ:  run.py = YAML→CSV（ROS不要, 制御実験）"
             " /  sim_node.py = ROSトピック入出力")
    L.append("")
    L.append("― 入力 : どの系から来るか " + "-" * (W - 26))
    for title, src, body, _ in INPUTS:
        L.append(f"  {title:<18} {src}")
        for ln in body.split("\n"):
            L.append(f"  {'':<18} {ln}")
    L.append(f"  {RUNTIME_IN}")
    L.append("")
    L.append("― 過程 : 層をどう流れるか " + "-" * (W - 27))
    L.append("  時計 MultiRateClock : dt_sim=積分 / dt_motor=制御(ZOH) / dt_nav=上位")
    L.append("")
    for i, (head, note) in enumerate(CHAIN):
        arrow = "        │\n        ▼\n" if i else ""
        for ln in arrow.split("\n"):
            if ln:
                L.append("  " + ln)
        L.append(f"  [ {head.replace(chr(10), ' ')} ]   {note}")
    L.append("")
    L.append("  DOB は『真のプラント と 名目モデル のズレ』だけを推定して打ち消す（= d_hat）")
    L.append("  外乱 d_true はプラントに注入。車輪はスリップ無で機体に結合 → 真の車輪角")
    L.append("  → Encoder量子化 → Jacobian逆 → odom pose （drift = 真値 − odom）")
    L.append("  センサ: IMU / LiDAR(raycast) / Encoder — レート・遅延・ノイズ・ドロップ")
    L.append("  積分は固定ステップ RK4 → 同一 seed でビット完全一致")
    L.append("")
    L.append("― 出力 : どの系へ行くか " + "-" * (W - 25))
    for title, body, dest in (OUTPUTS_OFFLINE, OUTPUTS_ROS, OUTPUTS_MAP):
        L.append(f"  {title}")
        for ln in body.split("\n"):
            L.append(f"      {ln}")
        L.append(f"      {dest}")
    L.append("")
    L.append("― 実機の各系との関係 " + "-" * (W - 22))
    for name, desc in REAL_SYSTEMS:
        L.append(f"  {name}:")
        for ln in desc.split("\n"):
            L.append(f"      {ln}")
    L.append("")
    L.append("― 制御を読む対応表 （左=概念 / 右=見る信号）" + "-" * (W - 44))
    for concept, sig in SIGNAL_MAP:
        L.append(f"  {concept:<18} : {sig}")
    L.append("=" * W)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 図版（poster）
# --------------------------------------------------------------------------- #
COLORS = {"物": "#f3d9b1", "機": "#cfe8cf", "制": "#cfe0f3", "環": "#e6d4ef",
          "R": "#f6cccc", "proc": "#eeeeee", "out": "#e9e2d0"}


def _box(ax, x, y, w, h, title, body="", fc="#eeeeee", ec="#555555", fs=8.0):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.4",
                                fc=fc, ec=ec, lw=1.2, zorder=2))
    ax.text(x + w / 2, y + h - 0.9, title, ha="center", va="top",
            fontsize=fs, weight="bold", zorder=3)
    if body:
        ax.text(x + w / 2, y + h - 2.3, body, ha="center", va="top",
                fontsize=fs - 1.4, color="#333333", zorder=3, linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1, color="#444444"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))


def _set_cjk_font() -> None:
    """Pick an installed CJK font so Japanese renders (else it draws as tofu)."""
    from matplotlib import font_manager, rcParams
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK JP", "Noto Serif CJK JP", "IPAexGothic",
                 "IPAGothic", "TakaoGothic", "VL Gothic", "MS Gothic",
                 "Hiragino Sans", "Yu Gothic"):
        if name in available:
            rcParams["font.family"] = name
            break
    rcParams["axes.unicode_minus"] = False


def draw_poster(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_cjk_font()
    fig, ax = plt.subplots(figsize=(13.5, 17.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 98.5, "omni_sim  取扱説明書", ha="center", fontsize=18, weight="bold")
    ax.text(50, 96.2, "入力 → 過程 → 出力  ｜  制御を理解する地図（現状仕様）",
            ha="center", fontsize=10.5, color="#444444")

    def band_label(y, s):
        ax.text(2, y, s, ha="left", fontsize=12, weight="bold", color="#1a3a5a")

    # ---- 入力帯 ----------------------------------------------------------
    band_label(93, "① 入力  — どの系から来るか")
    x, w, gap = 2.0, 17.6, 1.1
    for i, (title, src, body, tag) in enumerate(INPUTS):
        bx = x + i * (w + gap)
        _box(ax, bx, 80.5, w, 10.5, title, f"{src}\n{body}", fc=COLORS[tag], fs=7.6)
    ax.text(50, 78.6, RUNTIME_IN, ha="center", fontsize=8, color="#7a2a2a",
            style="italic")

    # arrows down into process
    for i in range(len(INPUTS)):
        bx = x + i * (w + gap) + w / 2
        _arrow(ax, bx, 80.3, bx if i in (2, 3) else 50, 75.6, color="#9aa")

    # ---- 過程帯 ----------------------------------------------------------
    band_label(76.5, "② 過程  — 層をどう流れる（時計が主役, 真値 vs 名目 が肝）")
    # clock strip
    _box(ax, 2, 70.5, 95.2, 3.8,
         "MultiRateClock", "dt_sim = 積分 (RK4固定刻み)   /   dt_motor = 制御 (ZOH)"
         "   /   dt_nav = 上位", fc="#dfe8f0", fs=8.5)

    # chain (left -> right)
    cw, cy, cx0, step = 16.0, 57.0, 3.0, 18.7
    xs = [cx0 + i * step for i in range(len(CHAIN))]
    for (head, note), bx in zip(CHAIN, xs):
        _box(ax, bx, cy, cw, 9.5, head, note, fc=COLORS["proc"], fs=8.0)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, a + cw, cy + 4.7, b, cy + 4.7)
    _arrow(ax, 50, 70.3, 50, cy + 9.6, color="#9aa")   # clock -> chain

    # DOB feedback under the 2nd box
    _box(ax, xs[1], 45.5, cw, 8.5, "DOB",
         "d_hat = 真のプラント と\n名目モデル のズレ を推定", fc="#e7dcf0", fs=8.0)
    _arrow(ax, xs[1] + cw / 2, 54.2, xs[1] + cw / 2, cy)          # DOB -> (−)
    _arrow(ax, xs[2] + cw / 2, cy, xs[1] + cw, 50.0, color="#888")  # omega_meas ->DOB

    # disturbance into plant
    _box(ax, xs[2], 45.5, cw, 8.5, "外乱 d_true",
         "step/ramp/... を注入\n(DisturbanceManager, seed)", fc="#f6dede", fs=8.0)
    _arrow(ax, xs[2] + cw / 2, 54.2, xs[2] + cw / 2, cy)

    # odom row under body
    _box(ax, xs[3], 45.5, cw, 8.5, "Encoder 量子化",
         "真の車輪角を量子化", fc="#eeeeee", fs=8.0)
    _box(ax, xs[4], 45.5, cw, 8.5, "Jacobian逆 → odom",
         "drift = 真値 − odom", fc="#eeeeee", fs=8.0)
    _arrow(ax, xs[4] + cw / 2, cy, xs[3] + cw / 2, 54.2, color="#888")  # truth->enc
    _arrow(ax, xs[3] + cw, 49.7, xs[4], 49.7)

    # sensors strip
    _box(ax, 2, 40.0, 95.2, 4.0, "センサ (sim_node が発行)",
         "IMU / LiDAR(env.raycast DDA で地図にレイキャスト) / Encoder"
         "  —  レート・遅延FIFO・ジッタ・ドロップ・量子化", fc="#e3efe3", fs=8.0)

    # ---- 出力帯 ----------------------------------------------------------
    band_label(37.5, "③ 出力  — どの系へ行くか")
    ow, ox0, ostep = 30.0, 2.0, 32.2
    for i, (title, body, dest) in enumerate(
            (OUTPUTS_OFFLINE, OUTPUTS_ROS, OUTPUTS_MAP)):
        bx = ox0 + i * ostep
        _box(ax, bx, 25.0, ow, 10.5, title, f"{body}\n\n{dest}",
             fc=COLORS["out"], fs=7.8)
    _arrow(ax, 50, 39.9, 50, 35.7, color="#9aa")

    # ---- 実機系との関係 --------------------------------------------------
    band_label(22.5, "④ 実機の各系との関係")
    rw, rx0, rstep = 30.5, 2.0, 32.4
    for i, (name, desc) in enumerate(REAL_SYSTEMS):
        bx = rx0 + i * rstep
        _box(ax, bx, 12.0, rw, 9.0, name, desc, fc="#f0eef7", fs=8.0)

    # ---- 制御を読む対応表 ------------------------------------------------
    band_label(9.0, "⑤ 制御を読む対応表  （左=概念 / 右=見る信号）")
    ty = 6.6
    for concept, sig in SIGNAL_MAP:
        ax.text(3.5, ty, f"• {concept}", ha="left", fontsize=8.2, weight="bold")
        ax.text(33, ty, sig, ha="left", fontsize=8.0, color="#333333")
        ty -= 1.35

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-png", action="store_true", help="テキストだけ生成")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manual = text_manual()
    print(manual)
    txt = args.out_dir / "omni_sim_manual.txt"
    txt.write_text(manual + "\n", encoding="utf-8")
    print(f"\n[explain] text  -> {txt}")

    if not args.no_png:
        try:
            png = args.out_dir / "omni_sim_manual.png"
            draw_poster(png)
            print(f"[explain] poster -> {png}")
        except Exception as exc:  # matplotlib 無し等
            print(f"[explain] PNG skipped ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
