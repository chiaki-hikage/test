# Running CARLA on Lightning.ai

Lightning.ai の T4 GPU Studio 上で CARLA Ubuntu package 版を RenderOffScreen で起動し、
4カメラ画像取得 (PhysicsNeMo egomotion replay) を実行するための手順書。

**前提環境**

| 項目 | 値 |
|---|---|
| GPU | Tesla T4 |
| CARLA | Ubuntu package 版 0.9.16 |
| CARLA package dir | `/teamspace/studios/this_studio/carla_pkg` |
| 作業ディレクトリ | `/teamspace/studios/this_studio/CARLAUE5` |
| Python client | `carla==0.9.16` |

**なぜ Ubuntu package 版を使うか**

Docker 版 CARLA では、コンテナ内に `nvidia_icd.json` が見えず Vulkan が lavapipe (CPU 実装) を掴んでしまい、
RenderThread timeout / Segmentation fault が発生した。
Ubuntu package 版を直接起動することで NVIDIA Vulkan ドライバを正しく参照できる。

---

### 1. 前提確認

環境が正しくセットアップされているかを確認する。

```bash
cd /teamspace/studios/this_studio/carla_pkg

# GPU 確認
nvidia-smi

# NVIDIA Vulkan ライブラリの場所を確認
find /usr -name "libGLX_nvidia.so.0" 2>/dev/null | head

# carla Python パッケージの確認
python - <<'PY'
import carla
print(carla.__version__ if hasattr(carla, "__version__") else "carla imported OK")
PY
```

---

### 2. 既存プロセスの停止

sleep 後の再起動時や、前回のプロセスが残っている場合に実行する。

```bash
pkill -f CarlaUE4 2>/dev/null || true
pkill -f Xvfb     2>/dev/null || true

# ポートが空いているか確認 (何も出なければ OK)
ss -ltnp | grep -E '2000|2001|2002' || true
```

---

### 3. nvidia_icd.json の作成

`/tmp` は sleep 後に消えるため、**起動のたびに毎回作成する**。

```bash
cat > /tmp/nvidia_icd.json << 'EOF'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libGLX_nvidia.so.0",
    "api_version": "1.2.0"
  }
}
EOF

export VK_ICD_FILENAMES=/tmp/nvidia_icd.json
```

---

### 4. Vulkan 確認

NVIDIA Vulkan ドライバが見えていることを確認する。

```bash
VK_ICD_FILENAMES=/tmp/nvidia_icd.json vulkaninfo 2>&1 \
  | grep -Ei "deviceName|driverName|driverInfo|vendorID|deviceID|NVIDIA|lavapipe|llvmpipe" \
  | head -80
```

**期待する出力 (抜粋)**

```
vendorID   = 0x10de
deviceName = Tesla T4
driverName = NVIDIA
```

> **注意**: `lavapipe` や `llvmpipe` が出る場合は、NVIDIA Vulkan ではなく CPU 実装を掴んでいる。
> その状態では `-RenderOffScreen` は失敗しやすい。
> `VK_ICD_FILENAMES` が正しく設定されているか、`libGLX_nvidia.so.0` のパスが正しいか確認すること。

---

### 5. 依存パッケージと Xvfb の起動

```bash
# 初回のみ (インストール済みであればスキップ可)
sudo apt-get update
sudo apt-get install -y xvfb vulkan-tools ffmpeg

# 仮想ディスプレイを起動
Xvfb :99 -screen 0 1920x1080x24 &
sleep 1
export DISPLAY=:99
```

---

### 6. CARLA サーバー起動

**CARLA サーバー専用のターミナルで実行し、このターミナルは開いたままにする。**
Python クライアントや画像取得スクリプトは別ターミナルから実行する。

```bash
cd /teamspace/studios/this_studio/carla_pkg

export VK_ICD_FILENAMES=/tmp/nvidia_icd.json
export DISPLAY=:99

./CarlaUE4.sh \
  -RenderOffScreen \
  -nosound \
  -quality-level=Low \
  -carla-rpc-port=2000 \
  -log
```

> `-RenderOffScreen` は Xvfb + NVIDIA Vulkan を使ってオフスクリーンレンダリングを行う。
> `-nullrhi` (レンダリング完全無効) ではカメラ画像が取得できないため使わない。

---

### 7. 接続確認

別ターミナルで実行する。CARLA サーバーの起動完了まで 30〜60 秒かかることがある。

```bash
python - <<'PY'
import carla

client = carla.Client("localhost", 2000)
client.set_timeout(60.0)

print("client:", client.get_client_version())
print("server:", client.get_server_version())

world = client.get_world()
print("map   :", world.get_map().name)
print("spawn points:", len(world.get_map().get_spawn_points()))
PY
```

**成功条件**

- `client:` と `server:` のバージョンが一致して表示される
- `map:` にマップ名が表示される
- `spawn points:` に正の整数が表示される

---

### 8. Town10HD_Opt を使う場合の注意

現在の route CSV・egomotion CSV はすべて `Town10HD_Opt` に対応している。

- Python スクリプト実行時は `--map Town10HD_Opt --strict-map-check` を推奨する
- `client.load_world()` はなるべく使わず、起動済みの world を `client.get_world()` で利用する方針とする
  (`load_world()` はマップ再ロードを伴い時間がかかる上、CARLA サーバーが不安定になる場合がある)

---

### 9. カメラ画像取得テスト

CARLA サーバーが起動していることを確認してから、別ターミナルで実行する。

まず 10 秒・800×450 で疎通確認し、問題なければ 20 秒・1280×720 に上げる。

```bash
cd /teamspace/studios/this_studio/CARLAUE5

# 疎通確認 (10 秒・800x450)
python scripts/replay_egomotion_capture_images.py \
  --csv outputs_physicsnemo/carla_route/Town10HD_Opt_sp147_normal/egomotion.csv \
  --sample-id town10_sp147_4cam_10sec_rain \
  --output-dir output \
  --map Town10HD_Opt \
  --strict-map-check \
  --absolute-coords \
  --duration-sec 10 \
  --image-width 800 \
  --image-height 450 \
  --weather hard_rain_fog \
  --tire-friction 0.5 \
  --z-offset 0.3
```

> - `--absolute-coords`: egomotion.csv の座標を CARLA ワールド座標として直接使用する
> - `--z-offset 0.3`: 路面埋まり防止のため車両を 0.3 m 上にオフセット
> - `--tire-friction 0.5`: `set_transform` replay のため軌跡には影響しない (見た目・physics 用)
> - 低μ挙動は PhysicsNeMo 側の egomotion に含まれており、CARLA 側では再現不要
> - `--weather hard_rain_fog`: カメラ画像の見た目用

**本番収録 (20 秒・1280×720)**

```bash
python scripts/replay_egomotion_capture_images.py \
  --csv outputs_physicsnemo/carla_route/Town10HD_Opt_sp147_normal/misjudged_low_mu/egomotion.csv \
  --sample-id town10_sp147_misjudged_low_mu_4cam_20sec_rain \
  --output-dir output \
  --map Town10HD_Opt \
  --strict-map-check \
  --absolute-coords \
  --duration-sec 20 \
  --image-width 1280 \
  --image-height 720 \
  --weather hard_rain_fog \
  --tire-friction 0.5 \
  --z-offset 0.3
```

出力ファイル構成:

```
output/town10_sp147_misjudged_low_mu_4cam_20sec_rain/
  images/
    camera_cross_left_120fov/   000000.png  000001.png  ...
    camera_front_wide_120fov/   000000.png  000001.png  ...
    camera_cross_right_120fov/  000000.png  000001.png  ...
    camera_front_tele_30fov/    000000.png  000001.png  ...
  ego_history.csv
  metadata.json
```

---

### 10. 4カメラグリッド動画の生成

ffmpeg で 4 カメラを 2×2 グリッドにまとめた動画を生成する。

```bash
BASE=output/town10_sp147_misjudged_low_mu_4cam_20sec_rain
FPS=10

ffmpeg -y \
  -framerate $FPS -start_number 0 -i "$BASE/images/camera_cross_left_120fov/%06d.png" \
  -framerate $FPS -start_number 0 -i "$BASE/images/camera_front_wide_120fov/%06d.png" \
  -framerate $FPS -start_number 0 -i "$BASE/images/camera_cross_right_120fov/%06d.png" \
  -framerate $FPS -start_number 0 -i "$BASE/images/camera_front_tele_30fov/%06d.png" \
  -filter_complex "
    [0:v]scale=640:360[left];
    [1:v]scale=640:360[front];
    [2:v]scale=640:360[right];
    [3:v]scale=640:360[tele];
    [left][front]hstack=inputs=2[top];
    [right][tele]hstack=inputs=2[bottom];
    [top][bottom]vstack=inputs=2[out]
  " \
  -map "[out]" \
  -c:v libx264 -pix_fmt yuv420p -crf 23 \
  "$BASE/preview_4cam_grid.mp4"

echo "Saved: $BASE/preview_4cam_grid.mp4"
```

グリッドレイアウト:

```
┌─────────────────────────┬─────────────────────────┐
│  camera_cross_left      │  camera_front_wide      │
│  (120° FOV / 左横)      │  (120° FOV / 前方広角)  │
├─────────────────────────┼─────────────────────────┤
│  camera_cross_right     │  camera_front_tele      │
│  (120° FOV / 右横)      │  (30° FOV / 前方望遠)   │
└─────────────────────────┴─────────────────────────┘
```

---

### 11. sleep 前の停止手順

Lightning.ai を sleep させる前に必ず実行する。

```bash
pkill -f CarlaUE4 2>/dev/null || true
pkill -f Xvfb     2>/dev/null || true

# プロセスが残っていないことを確認 (何も出なければ OK)
ps aux | grep -E "CarlaUE4|Xvfb" | grep -v grep || true
ss -ltnp | grep -E '2000|2001|2002' || true
```

---

### 12. sleep 後に再実行が必要なもの

Lightning.ai の sleep からの復帰後、以下は**毎回再設定が必要**。

| 項目 | 理由 |
|---|---|
| `/tmp/nvidia_icd.json` の作成 | `/tmp` は sleep 後に消える |
| `export VK_ICD_FILENAMES=/tmp/nvidia_icd.json` | シェルセッションがリセットされる |
| `export DISPLAY=:99` | シェルセッションがリセットされる |
| Xvfb `:99` の起動 | プロセスが停止している |
| CARLA サーバー (`./CarlaUE4.sh`) の起動 | プロセスが停止している |

復帰時の手順: **セクション 2 → 3 → 4 → 5 → 6 → 7** の順に実行する。

---

### 13. トラブルシュート

#### A. `lavapipe` warning が出る / Vulkan が NVIDIA を掴まない

**症状**: `vulkaninfo` の出力に `lavapipe` や `llvmpipe` が含まれる。

**原因と対処**:
1. `VK_ICD_FILENAMES` が設定されていない → セクション 3 を再実行
2. `libGLX_nvidia.so.0` のパスが異なる → `find /usr -name "libGLX_nvidia.so.0"` でパスを確認し、`library_path` を絶対パスに書き換える

```bash
# フルパスで指定する場合の nvidia_icd.json 例
NV_LIB=$(find /usr -name "libGLX_nvidia.so.0" 2>/dev/null | head -1)
cat > /tmp/nvidia_icd.json << EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "$NV_LIB",
    "api_version": "1.2.0"
  }
}
EOF
```

#### B. `Spawn failed because of collision at spawn position`

**症状**: CARLA の spawn 位置で衝突エラーが発生する。

**原因と対処**:
- ego vehicle を egomotion の最初のポーズに直接 spawn しない
- `try_spawn_actor` で安全な spawn point に一度 spawn してから `set_transform()` で移動する
- `--z-offset 0.3` を指定して路面への埋まりを防ぐ
- `scripts/replay_egomotion_capture_images.py` はこの方式を実装済み

#### C. カメラ画像が保存されない / 全フレームがスキップされる

**症状**: 実行は成功するが `images/` フォルダが空、または `saved_frames=0`。

**確認事項**:
1. CARLA が `-RenderOffScreen` で起動しているか (`-nullrhi` は使わない)
2. `DISPLAY=:99` が設定されているか
3. Xvfb が起動しているか (`ps aux | grep Xvfb`)
4. world tick とセンサーフレームの同期がずれていないか (ログの `[SKIP]` 行を確認)

#### D. CARLA サーバーへの接続がタイムアウトする

**症状**: `client.set_timeout(60.0)` 後に接続エラー。

**確認事項**:
1. CARLA サーバーのターミナルで起動ログを確認 (起動完了まで 30〜60 秒)
2. ポートが使用中でないか: `ss -ltnp | grep 2000`
3. 古い CARLA プロセスが残っていないか: `pkill -f CarlaUE4`
4. `nvidia-smi` で GPU が認識されているか確認
