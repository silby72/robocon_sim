# omni_sim — 現状 (2026-09-22)

方針会議用のスナップショット。`ROADMAP.md` は Phase A–C / P1–P4 の**経緯**の記録で、
2026-09-13 で止まっている。本書は**今のコードが何をするか・何をしないか**を、
判断に使える数値つきで書く。

---

## 1. これは何か

ハードウェア不要の4輪オムニ ロボット シミュレータ。ABU Robocon 2027（ソロ、インドネシア）
のフィールドを対象にした**制御・経路計算の検証環境**。

**テーマは明示的にアンチGazebo** — 純Python＋自前2D描画。LiDARは2Dしか使わないので
フィールドを高さ注釈つき2Dプリミティブに分解し、レイヤごとの占有格子にスライスする。

### 貫かれている不変条件（破ると設計が壊れる）

| | |
|---|---|
| `gui → core` 一方向 | core は PySide6 非依存 |
| `core` は rclpy 非依存 | ROSは `omni_sim_ros/sim_node.py` だけが触る |
| `plant` は `mechanism` を import しない | 逆は可。層の向き |
| **真値と名目の分離** | 制御器が信じるモデルと真のプラントは別オブジェクト・別YAML |
| **「円で計画し、矩形で判定する」** | A* は外接円で膨張、当たり判定は実形状。余裕は `r_circ − r_insc` だけ |
| コメント保持YAML | GUIの保存でコメントが消えない（ruamel） |
| 決定論 | 乱数は必ず明示的な `Generator` 経由 |

---

## 2. 起動

```bash
# テスト（137）
.venv/bin/python -m pytest omni_sim_core/tests -q

# ROS 2 + ブラウザ ダッシュボード
scripts/build_ros.sh                       # colcon を直接叩かない（後述）
ros2 launch omni_sim_ros sim_with_dashboard.launch.py \
    map_yaml:=$PWD/maps/field_2027_ground.yaml
cd omni_sim_ros/web && python3 -m http.server 8777
# → http://localhost:8777/dashboard.html

# 機体設定GUI
python setup_env.py --extras gui && python run_gui.py
```

`scripts/build_ros.sh` が必要な理由：colcon がビルドのたびに壊す2点を直す。
(1) 実行ファイルが `install/<pkg>/bin/` に入るが launch は `lib/<pkg>/` を見る、
(2) エントリポイントの shebang が `/usr/bin/python3` になり `omni_sim_core` が見えない。
後者は importlib の奥から `ModuleNotFoundError` が出るのでパッケージング問題に見えない。

---

## 3. 実装済み・検証済み

### コア（rclpy非依存）

- **プラント**: マルチレート時計 / RK4 / モータ真値モデル / 名目モデル / 外乱注入
- **制御**: PID・DOB（Tustin離散化）・軌道生成（台形/S字）・追従制御
- **駆動系**: ヤコビアン層（**実行列を受け取れる** — 輪ごとの位置・半径・ギア比・reverse）
- **経路計画** `planning/`: CostField 膨張 → A* → 視線短絡 → **C²クランプBスプライン平滑化**
  → 前後2パスの速度プロファイル（曲率由来の横加速度制限）
- **フィールド** `field/`: 高さ注釈プリミティブ → レイヤ別占有格子。
  `ConnectorSpec`（坂）、`SlopeField`（傾斜・重力）
- **評価** `evaluation/`: 真の幾何に対する解析的な最小距離
- **スイープ** `sweep/`: 直積展開・multiprocessing・境界二分探索
- **オドメトリ誤差** `plant/odometry.py`: 半径/トレッド/スリップ誤差＋従動輪ポッド

### ROSノード / ブラウザ

トピック: `/clock /scan /imu/data_raw /joint_states /odom /ground_truth/odom /plan
/trajectory /goal_pose /cmd_vel /sim/scene /sim/state /sim/set_motor /sim/reload_robot
/sim/disturbance /sim/wheel_torque_cmd`、サービス `/sim/reset /sim/pause`。

ダッシュボードはフィールド・壁（レベル別）・落下域・坂の勾配・**実機形状の機体**・
A*折れ線・追従曲線・オドメトリのゴースト・状態遷移チップ・モータ選択を描く。
**描画用の座標を一切持たない** — 全部 `/sim/scene` 経由で判定側と同じ数字。

---

## 4. 測定された事実（判断材料）

### 駆動系 — **ここが一番の争点**

現行構成（M3508 P19 + **外部6:1**、2インチ輪、15 kg、0.9 m角）:

| | |
|---|---|
| 持続可能速度 | **+x 0.511 m/s / 対角 0.361 m/s**（対角が律速。駆動軸±45°で2輪が全速、2輪が遊ぶ） |
| 登坂 | 静止保持 45°(1595 N)、0.36 m/s で 45°(638 N) |
| 以前の計画速度 | 1.2 m/s（= **実力の3.3倍**。全ての軌道が追従不能だった） |

**6:1 はほぼ確実に仮値。** M3508の内部19:1の出力にさらに6:1を掛けているので
輪速度の上限が 0.43 m/s。ロボコン機として遅すぎる。2 m/s 相当なら **1.3:1 前後**。
**この1数値で経路計画の所要時間が全部変わる。**

### フィールド

| | |
|---|---|
| 坂 | 600 mm 上昇 / 1025 mm 走行 = **30.3°**（オムニでは登れない。図面由来の `[F]` 推定） |
| L1リング幅 | 生 1.5 m → r_circ 0.636 で膨張後 **0.228 m** |
| 坂から到達できるL1 | x 2.12..4.96, y 3.14..7.84 |
| 階段ゲートの到達可能部 | **x 4.70..4.96 のみ**（中央仕切り柵がリング中央を塞ぐ） |
| L2に到達できる最大機体 | **1.04 m 角**（0.9 m は余裕0.14 m） |

### 当たり判定

| | |
|---|---|
| 単一レベル | めり込み **0.0 mm / 0.0 cm²**（多角形の面積判定） |
| ゲート横断中 | **20.0 mm = 1セル**（レイヤ ラスタの継ぎ目を埋める許容。そこだけ） |
| 修正前 | **443 mm**（4隅の点判定＋レベルのテレポート） |
| 判定コスト | 126 µs/回、移動5 mmごとに間引き → rtf 影響 約2% |

### オドメトリ（30秒・9.0 m 走行＋自転）

| 推定器 | ドリフト | 経路比 | 方位 |
|---|---|---|---|
| 駆動輪（スリップ0.2%） | 33.4 mm | 0.36 % | −0.35° |
| 従動輪ポッド＋ジャイロ | 50.2 mm | 0.54 % | −0.44° |
| 駆動輪（スリップ**2%**） | **173.8 mm** | 1.86 % | +1.43° |
| ポッド（同上） | **50.2 mm 不変** | 0.54 % | −0.44° |

**ポッドは無条件に良くない。** 軽いスリップなら駆動輪が勝つ。
ポッドが買うのは「駆動スリップ耐性」で、代わりに払うのが「ジャイロ バイアス」。

誤差源の内訳（単独適用）: トレッド誤差0.6% → 56.4 mm/+4.10°（**支配的**）、
スリップ0.2% → 18.0 mm、輪半径±0.4% → 4.6 mm。
**全部足すと33.9 mm** — 打ち消し合うので誤差バジェットは単純加算できない。

---

## 5. このプロジェクトに繰り返し出るパターン（重要）

**「宣言されているが配線されていない機能」が系統的に存在した。** 今セッションで見つけた分:

| 機能 | 状態だった |
|---|---|
| `MotorArray` | `RobotSim` が生成も reset もするが**一度も積分しない**。Kt/R/慣性/トルク制限が全部無効 |
| `/sim/disturbance` | 受信して保存するが `step()` に渡す口がなく**適用されない** |
| `config/robot/*` 全体 | `RobotSim` はライブラリ既定値で構築されていた（駆動系が**5.9倍**ズレ） |
| `odometry[]`（従動輪/光学） | スキーマもGUIもあるが**読むコードがない** |
| 坂 | フィールド定義に**存在しなかった**（描画専用の矩形。2層のグリッドは非連結） |
| `model_error.ratios` | `MotorControlSim` 専用。ロボット側は未使用（これは設計通り） |

いずれも「エラーが出ない」形で壊れていた。**残りの層にも同種がある前提で、
一度系統的な監査をする価値がある**というのが今セッションの示唆。

---

## 6. 人の判断が要る未決定事項

1. **坂の実寸法**（30.3°）。YAML1行だが、登坂力・所要時間・経路長の全部が従属する。
2. **ギア比 6:1**。上と同じく1数値だが影響範囲が最大。
3. **機体サイズ 0.9 m**。L2到達の上限が 1.04 m なので余裕0.14 m。小さくする判断はあるか。
4. **競技の状態機械**。今の `phase` は**場所**の機械（座標しか読んでいない）。
   得点・保持・Sky Block をモデル化するなら定義が要る。
5. **TR/BR 2台構成**。ルール上の役割分担が未反映。

---

## 7. 既知の欠落・制限（影響順）

| # | 内容 | 備考 |
|---|---|---|
| 1 | **自己位置推定器が無い** | オドメトリ誤差は本物になったが、補正する側（LiDAR＋粒子フィルタ等）が未実装。LiDAR・IMU・地図は揃っている |
| 2 | 輪ごとのモータ差を表現できない | `MotorParams` が全輪共通。実際に `m3508_fl` の実測上書きが全輪に適用される（警告は出る） |
| 3 | 重心オフセットのカップリング | `RigidBody` が対角質量行列。`build_mass_matrix` は3×3を出せるのに使えない |
| 4 | 衝突の巻き戻しがステップ全体 | 壁に触れると**壁沿いに滑れない**。違反方向の成分だけ潰すべき |
| 5 | `cmd_vel` の配分が粗い | 輪速度誤差のP制御。飽和時に姿勢が崩れる（`generic_dc` の+x方向で再現） |
| 6 | 坂はトラクション/転倒を見ない | 摩擦モデルが全体に無いため。`m·g·sinα·cosα` の水平成分のみ |
| 7 | ゴール応答 最大1.2秒 / rtf 0.89 | `_on_tick` がシングルスレッドexecutorを塞ぐ |
| 8 | LiDAR動き歪み補正がオフ | ビーム毎レイキャストで多秒停止。ベクトル化すれば使える |
| 9 | Phase D（Sensorsページ） | `sensors.yaml` のスキーマは仮置き済み、GUIページ無し |
| 10 | `ros2_ws/log/` が git 追跡下 | `.gitignore` 行き |

---

## 8. バックログ（コスト感）

**安い（半日以下）**
- `.gitignore`、LiDARベクトル化、`_on_tick` のオーバーラン時ステップ破棄

**中（1–2日）**
- 輪ごとのモータ（#2）／重心カップリング（#3）／衝突の方向別解決（#4）
- `cmd_vel` を機体レンチからの配分に置換（#5）— `wheel_torque_from_body_force` は既にある

**大（数日〜）**
- **自己位置推定器**（#1）。オドメトリ誤差を作った以上、次の自然な一歩
- 競技モデル（得点・物体・Sky Block）
- TR/BR 2台

---

## 9. ファイルの地図

```
omni_sim_core/src/omni_sim_core/
  plant/      motor jacobian body nominal odometry     ← 真のプラント
  control/    pid dob trajectory                        ← 制御
  planning/   cost_field grid_planner shortcut corner smooth orientation
  field/      spec primitives slicer layered_field slope
  evaluation/ footprint geometry_dist evaluator
  mechanism/  schema derive jacobian build robot_config ← 設定→実行可能な形へ
  sensors/    imu lidar encoder
  ui/         console viewer field_layout_2027 leveled_field_2027 web_scene phase
omni_sim_ros/omni_sim_ros/sim_node.py                   ← rclpy はここだけ
omni_sim_ros/web/dashboard.html                         ← 座標を持たない
omni_sim_gui/pages/{chassis,actuator}_page.py           ← 設定を書く側
config/robot/{chassis,actuators,model_error,sensors}.yaml
config/field/robocon2027.yaml                           ← [R]/[F]/[A] の出典注記つき
scripts/{build_ros.sh,measure_corridors.py}
```

`[R]` ルールブック記載 / `[F]` 図から起こした推定 / `[A]` 記載なしの仮定。
**この注記はフィールド定義の全数値に付いている** — 会議で「その数字はどこから来たか」
を問う時はここを見る。
