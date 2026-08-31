# omni_sim / robocon_sim — 進捗と今後の方針

最終更新: 2026-08-30

制御学習用オムニシミュレータを「ロボコン用汎用シミュレータ＋設定GUI」へ拡張中。
本書は進捗の要約と、次に議論する **経路計画（path planning）** の設計たたき台。

---

## 1. これまでの進捗

### ベースシミュレータ（omni_sim, コアは rclpy 非依存）
- マルチレート時計 / 固定ステップRK4 / モータ真値プラント / 名目モデル / 外乱注入
- PID・DOB（プロパ2フィルタをTustin離散化）・軌道生成・追従制御
- ヤコビアン層・3自由度剛体・占有格子・ベクトル化DDAレイキャスト
- センサ（IMU/LiDAR/エンコーダ）・2エントリ（MotorControlSim / RobotSim）
- ROS 2 (Jazzy) ラッパ・serial2can風コンソールUI・matplotlibライブビュー
- 受け入れ9項目＋単体テスト

### Phase A — 機構パラメータ層（`omni_sim_core/mechanism/`）
- `schema`（chassis/actuators/odometry・検証）/ `derive`（データシート→物理量, 推定は明示）
- `jacobian`（駆動ヤコビアン・条件数・特異検出）/ `build`（→ `config/generated/plant_*.yaml`, ruamelでコメント保持）
- `config/robot/*`, `config/presets/motors/*`, `model_error`
- 制御周期分離 `control_rate_hz`（積分dt_simと分離、ZOH）
- シナリオ `plant: {true_model, nominal_model}`（後方互換）
- **override方針**：フラットdict。データシート系キーはderive前、物理系キーはderive後にマージ
- テスト: 30 passing

### Phase B — 機体設定GUI（`omni_sim_gui/`, PySide6優先→PyQt5フォールバック）
- QGraphicsView配置キャンバス（ドラッグ＋10mmスナップ・回転ハンドル・対称配置）
- 数値フィールド双方向同期 / footprint・CoM・輪・reverse・独立オドメトリ(dead_wheel/optical)
- 追加削除＋Undo/Redo / ライブ検証（条件数・アクチュエータ整合・配置妥当性: footprint外/重心が凸包外）
- 保存: 差分ダイアログ→アトミック書込み・コメント保持・表示mm/度→保存SI

### Phase C — アクチュエータGUIページ
- 一覧＋追加削除＋Undo/Redo / プリセット選択 / 上書き色分け＋revert / 物理上書き
- 導出値ライブ表示（Kt/Ke/R/τ_stall/ω_noload/**J_reflectedは参照輪のギアで算出**）・推定バッジ
- chassis の `actuator_ref` と突き合わせ（使用状況表示）

### 貫かれている不変条件
- `gui → core` 一方向（coreはGUI非依存）/ coreは rclpy・PySide6 非依存
- 真値と名目の分離 / コメント保持YAML / SI単位

### 環境メモ（この開発機）
- `.venv`（--system-site-packages）＋ ruamel ＋ PyQt5。テストは `.venv/bin/python -m pytest`
- **GUIはWaylandセッションのXwayland `:1` に出す**:
  `DISPLAY=:1 XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.* QT_QPA_PLATFORM=xcb`
  （`:0` はGDMグリーターで不可視）。後片付けの `pkill -f run_gui.py` は自滅するので `[r]un_gui.py`

---

## 2. 直近の残タスク
- Phase D: Sensorsページ（`sensors.yaml`: IMU/LiDAR配置）— スキーマは仮置き済み
- Actuatorsの詰め: リネーム / 整合サマリ全体 / 入力バリデーション / 物理上書き追加 / 保存後build_plant実行の確認
- Phase A のスキーマ確定レビュー（アジャイルでB/Cに先行中）

---

## 3. 次の議題：経路計画（path planning）の設計たたき台

### 3.1 用語の切り分け（重要）
現状のシムにある/ない を明確化:
- **経路計画（path planning, 未実装）**: start→goal で占有格子上の障害物回避ルートを計算
- **軌道生成（trajectory generation, 実装済み）**: 経由点列→速度プロファイル（台形/S字）→時間付き軌道
- **追従制御（trajectory following, 実装済み）**: 位置PID＋速度FF（機体座標）

→ 今欠けているのは **グローバルな経路計画器**。その出力（経由点/経路）を既存の
`control/trajectory.py` の軌道生成に渡す形になる。

### 3.2 使える既存インフラ
- `env/occupancy_grid.py`（map_server互換, `generate_rect_field`）
- `env/raycast.py`（ベクトル化DDA・衝突判定に流用可）
- `RobotSim` / `TrajectoryFollower` / マップ `maps/field*.yaml`

### 3.3 オムニ特有の前提
- **ホロノミック**：任意方向へ即時並進可＋回転独立。→ 非ホロノミック拘束が無いので
  経路計画は素直（車体向きθは経路と分離して計画できる）。
- 一方で「向きも指定したい」「機体フットプリントでの衝突」を入れると設計が増える。

### 3.4 決めたい論点（ここをClaudeと詰める）
1. **アルゴリズム**: グリッドA*/Dijkstra/JPS（既知マップ向き・標準）｜サンプリングRRT/RRT*/PRM｜
   ポテンシャル/ベクタフィールド｜最適化(CHOMP/TrajOpt)。ロボコン既知マップなら A*＋平滑化が第一候補。
2. **コスト**: 距離最短 / 障害物からのクリアランス / エネルギー / 時間。多目的の重み。
3. **向き(θ)の扱い**: 経路と分離して別プロファイル / 経路接線に合わせる / 自由。
4. **衝突判定**: 点近似 / フットプリント（占有格子の膨張=inflation層を持つか）。
5. **連結点**: planner → 経由点 → 台形/S字プロファイル → follower の受け渡し仕様。
6. **再計画/動的障害物**: 今回スコープ内か（まずは静的マップで一発計画を提案）。
7. **置き場所**: `omni_sim_core/planning/`（rclpy非依存）新設。GUIは触らない（計画は実行時）。
8. **可視化/評価**: live_view に計画経路＋実走行＋クリアランスを重畳。sweepで比較。

### 3.5 提案する最初のスライス（議論のたたき台）
- `planning/grid_planner.py`: 占有格子に **inflation（機体半径）** → **A***（8近傍, ユークリッド）→
  **経路平滑化**（視線短絡 line-of-sight / もしくはBスプライン）
- 出力を既存 `Trajectory`（台形/S字）に渡し `TrajectoryFollower` で追従
- `experiments/plan_and_run.py` で start/goal を与えて live_view 表示
- テスト: 既知矩形＋障害物で最短経路長・無衝突・inflationの効きを検証

---
（このファイルは議論用のスナップショット。実装が進んだら更新する）
