# モータプリセットの追加

このディレクトリの `*.yaml` は**手で転記したメーカーのデータシート**です。出典は
各ファイル先頭のコメントだけで、それ以外に裏取りはありません。追加するときは
以下を守ってください。

## まず知っておくこと: 書いた値の 2 つは読まれません

`mechanism/derive.derive_motor` が実際に使うのはこれだけです:

| フィールド | 使われ方 |
|---|---|
| `kv_rpm_per_v` | `Kt = Ke = 60 / (2π·kv)` |
| `rated_voltage_v` | `ω_noload = kv·V`、`R = V / I_stall` |
| `stall_current_a` | `R = V / I_stall`、`τ_stall = Kt·(I_stall − I_nl)` |
| `no_load_current_a` | `τ_stall`、`B = Kt·I_nl / ω_noload` |
| `rotor_inertia_kgm2` | そのまま。`null` なら推定（警告 + ESTIMATED） |
| `encoder_cpr` | derive では未使用（エンコーダモデル側で使用） |

**`stall_torque_nm` と `no_load_speed_rpm` は一度も読まれません。** derive は
両方を計算し直します。つまりデータシートのストールトルクをそのまま書いても
シミュレータの挙動は 1 mm も変わりません。

残す意味は**照合**です。書いた値とモデルの計算が食い違うなら、3 つの数字
（kv / I_stall / τ）のどれかが転記ミスです。`test_motor_presets.py` がこれを
検査します。

## 減速機付きは「どちらの軸か」を必ず書く

これが一番やらかす箇所です。既存の `ak40_10_v3.yaml` が実例で、ロータ側の
kv (100 rpm/V) と出力側のストールトルク (24.5 N·m) が混ざっています。
この 2 つを同時に満たす `Kt` は存在しません:

```
Kt = 24.5 / (60 − 1.2) = 0.417 N·m/A  →  kv = 60/(2π·0.417) = 22.9 rpm/V
```

結果、シミュレータは 24.5 N·m ではなく **5.6 N·m** で走っています（0.23×）。
`m3508_c620` と `m2006_c610` は「出力軸換算」とヘッダに明記してあるので
一貫しています。**全部を同じ軸に揃えてから書いてください。**

## 手順

### A. ブラウザで作る（推奨）

`omni_sim_web/index.html` → **機体設定** → モータの一覧から「＋ 新規モータ…」。
入力すると即座に `omni_sim_core` が回り、照合の 2 項目に緑/赤の LED が出ます。
赤なら転記ミスです。**「YAML を書き出す」** でこのディレクトリに置ける形の
ファイルが出ます。

### B. 直接書く

```yaml
# <メーカー名 型番> -- どの軸（ロータ / ギヤ出力）の値かを必ず書く。
# 出典 URL か型番、取得日。
datasheet:
  rated_voltage_v: 24.0
  kv_rpm_per_v: 20.0
  stall_torque_nm: 4.77       # 照合用。derive は Kt·(I_stall−I_nl) で再計算
  stall_current_a: 10.5
  no_load_speed_rpm: 480.0    # 照合用。derive は kv·V で再計算
  no_load_current_a: 0.5
  rotor_inertia_kgm2: 1.0e-4  # 不明なら null（推定され ESTIMATED が付く）
  encoder_cpr: 8192

physical:                     # 任意。実測値はデータシートより優先される
  resistance_ohm: 0.10
```

### 置いたあと

```bash
.venv/bin/python -m pytest omni_sim_core/tests/test_motor_presets.py   # 転記の検査
scripts/sync_web_core.sh                                              # ブラウザ側へ反映
```

前者を通さないと気付けません。後者を忘れるとブラウザの一覧にだけ出てこず、
ページが壊れているように見えます（`test_every_preset_is_reachable_from_the_browser_bundle`
が落ちて教えます）。

## 使い方

`config/robot/actuators.yaml` から名前で参照します。実行中の差し替えは
コンソールの **自動走行** パネルの「駆動モータ」ドロップダウン
（`/sim/set_motor`）でもできます。
