# 現状のモデル構成整理

## 1. 目的

現状のモデルは、CAN/OBD と GPS 由来の時系列データを用いて、**短時間先の局所運動差分**を物理モデルと整合する形で表現することを目的としている。  
特に、将来的には低摩擦路面などのロングテール条件へ外挿可能な構成を目指しており、その前段として、

- 利用可能な観測量の整理
- yaw rate proxy の構成
- bicycle model ベースの物理コア導入
- 実測差分とのフィット

を進めている。

---

## 2. 入力としている CAN / GPS 情報

現状の学習入力は、1 時刻あたり以下の 6 変数を基本としている。  
さらにこれらを **history_steps 個だけ過去方向に連結**し、モデル入力としている。

### 2.1 基本入力（1時刻あたり）

1. **`ax_mps2`**  
   縦加速度 proxy  
   - 元データ: `Acceleration (ft/s²)`
   - 単位変換後: `m/s²`

2. **`throttle_pct`**  
   スロットル開度  
   - 元データ: `Absolute throttle position (%)`

3. **`rpm`**  
   エンジン回転数  
   - 元データ: `Engine RPM (RPM)`

4. **`vx_mps`**  
   車速  
   - 元データ: `Vehicle speed (MPH)`
   - 単位変換後: `m/s`

5. **`yaw_like_rate_radps`**  
   yaw rate proxy  
   - `Accel (Grav) X/Y/Z` から鉛直方向を推定
   - `Rotation Rate X/Y/Z` をその鉛直方向に射影
   - 定義イメージ:
     $$
     r_{\text{yaw-like}} = -(\omega \cdot \hat{u}_{down})
     $$
   - 単位: `rad/s`

6. **`is_yaw_reliable`**  
   yaw proxy 信頼度フラグ  
   例:
   - `vx >= 5 m/s`
   - `yaw_like_rate` と `r_xy` が有限
   などをもとに構成

---

### 2.2 history 入力

上記 6 変数を、現在時刻 `t` に加えて過去 `t-1, t-2, ...` を連結し、

$$
\text{input_dim} = 6 \times \text{history\_steps}
$$

としている。  
例えば `history_steps = 3` なら、入力次元は

$$
18
$$

となる。

---

## 3. 前処理で構成している観測量・教師信号

前処理では、学習用に以下の派生量を構成している。

### 3.1 GPS / 位置由来

- `x_meas_m`, `y_meas_m`  
  緯度経度からローカル平面へ変換した位置

- `course_rad`  
  位置微分から作った進行方向

- `r_xy_radps`  
  `course` の時間微分から作った yaw rate proxy
  $$
  r_{xy} = \frac{d(\text{course})}{dt}
  $$

- `is_xy_reliable`  
  XY 由来姿勢の信頼フラグ

---

### 3.2 学習用差分教師

- `dx_true_m`
  $$
  \Delta x_t = x_{t+1} - x_t
  $$

- `dy_true_m`
  $$
  \Delta y_t = y_{t+1} - y_t
  $$

- `dpsi_true_rad`
  $$
  \Delta \psi_t = \psi_{t+1} - \psi_t
  $$
  ただし wrap 処理あり

- `dvx_true_mps`
  $$
  \Delta v_{x,t} = v_{x,t+1} - v_{x,t}
  $$

---

## 4. 車両関連の初期パラメータ

モデルには、車両仕様や事前仮定から設定する初期パラメータがある。

### 4.1 固定または初期値として与えるもの

- **`mass_kg_init`**  
  車両質量の初期値  
  - 既存の mass fit 結果や仕様値を利用

- **`wheelbase_m`**  
  ホイールベース

- **`front_weight_fraction`**  
  前輪荷重比  
  これにより
  - `lf`（重心から前輪まで）
  - `lr`（重心から後輪まで）
  を構成する

- **`cf_n_per_rad_init`**  
  前輪コーナリング剛性初期値

- **`cr_n_per_rad_init`**  
  後輪コーナリング剛性初期値

- **`iz_init`**  
  ヨー慣性モーメント初期値  
  例:
  $$
  I_z \approx m \, l_f \, l_r
  $$

---

### 4.2 境界つきで学習する物理パラメータ

現状の本命版では、以下を **明示的な学習パラメータ**として持つ。

1. **`mu`**  
   摩擦係数  
   - 例: `0.35 ~ 1.30`

2. **`Cf`**  
   前輪コーナリング剛性  
   - 初期値に対する scale で学習

3. **`Cr`**  
   後輪コーナリング剛性  
   - 初期値に対する scale で学習

4. **`Iz`**  
   ヨー慣性モーメント  
   - 初期値に対する scale で学習

5. **`mass_kg`**  
   質量  
   - 初期値の周辺で scale として微調整

---

## 5. モデル状態

現状の物理コアでは、状態を次のように置いている。

$$
s = [x, y, \psi, v_x, \beta, r]
$$

- `x, y` : 位置
- `psi` : 車体向き（heading）
- `vx` : 前進速度
- `beta` : スリップ角
- `r` : yaw rate

ただし、現状では `r` の物理予測が不安定だったため、実験的に

$$
r \approx r_{\text{yaw-like}}
$$

として **yaw proxy を直接使う構成**も試している。

---

## 6. モデルに入れている物理法則

## 6.1 bicycle model の横運動

前後輪スリップ角を

$$
\alpha_f = \beta + \frac{l_f r}{v_x} - \delta
$$

$$
\alpha_r = \beta - \frac{l_r r}{v_x}
$$

で定義する。

ここで

- `delta` : 操舵角
- 現状は操舵 CAN が無いため、**曲率ベース proxy**
  $$
  \delta_{\text{proxy}} \approx L \frac{r_{\text{proxy}}}{v_x}
  $$
  を使う方向にしている

---

## 6.2 線形タイヤモデル

前後輪横力を

$$
F_{yf}^{lin} = -C_f \alpha_f
$$

$$
F_{yr}^{lin} = -C_r \alpha_r
$$

で与える。

---

## 6.3 摩擦制限

横力は摩擦限界でクリップする。

$$
|F_{yf}| \le \mu F_{zf}
$$

$$
|F_{yr}| \le \mu F_{zr}
$$

したがって実際に使う横力は、

- 線形タイヤモデルの値
- 摩擦限界 `\mu F_z`

の範囲に制限されたものになる。

---

## 6.4 横運動方程式

スリップ角の時間変化は

$$
\dot{\beta} = \frac{F_{yf}+F_{yr}}{m v_x} - r
$$

で与える。

---

## 6.5 yaw 運動方程式

yaw rate の時間変化は

$$
\dot r = \frac{l_f F_{yf} - l_r F_{yr}}{I_z}
$$

で与える。

ただし現状では `r_pred` が大きく暴れやすかったため、切り分け実験として

- `r` を物理予測せず
- `yaw_like_rate_radps` を直接使う

構成も試している。

---

## 6.6 位置更新

位置差分は

$$
\Delta x = v_x \cos(\psi+\beta)\Delta t
$$

$$
\Delta y = v_x \sin(\psi+\beta)\Delta t
$$

で与える。

ここで `psi + beta` が実際の進行方向を表す。

---

## 6.7 heading 更新

heading 変化は

$$
\Delta \psi = r \Delta t
$$

で与える。

---

## 6.8 縦方向

縦方向は簡易に

$$
\dot v_x \approx a_x^{meas}
$$

として扱い、必要なら小さな residual 補正を加える。

---

## 7. residual / proxy の扱い

現状の本命版では、物理コアを主にしつつ、必要最小限の補正を入れる構成を取っている。

### 7.1 residual の役割
residual は主に

- `beta_dot`
- `r_dot`
- `dvx`

への小補正として入れる

ただし最近の実験では、yaw が不安定だったため `r` は proxy 固定に近づけ、  
residual の主役性は減らしている。

---

### 7.2 操舵角 `delta`
操舵 CAN が無いため、現状では主に

$$
\delta_{\text{proxy}} \approx L \frac{r_{\text{proxy}}}{v_x}
$$

を使っている。  
以前は latent steering を NN で推定していたが、自由度が大きすぎて不安定になりやすかったため、  
現在は観測量に直接結びつく proxy ベースを優先している。

---

## 8. 最終的に CAN データと何を fit しているか

現状で主に実測差分と合わせているのは、次の 3 つである。

1. **`dX`**
   $$
   \Delta x^{pred} \leftrightarrow \Delta x^{true}
   $$

2. **`dY`**
   $$
   \Delta y^{pred} \leftrightarrow \Delta y^{true}
   $$

3. **`dVx`**
   $$
   \Delta v_x^{pred} \leftrightarrow \Delta v_x^{true}
   $$

---

### `dPsi` について
`dPsi` は当初 loss に入れていたが、yaw 系が不安定だったため、現状では主指標から外すことがある。  
ただし補助整合として使う余地はある。

---

### `r` について
`r_pred` を物理予測していた版では

$$
r^{pred} \leftrightarrow r^{proxy}
$$

も fit 対象だった。  
ただし現状では `r = yaw_like_rate` とする実験も行っており、その場合は `r` の予測 loss は実質不要となる。

---

## 9. Loss の入れ方

現状の total loss は、概念的には次のように分かれている。

$$
L =
w_{xy} L_{xy}
+
w_v L_v
+
w_r L_r
+
w_\psi L_\psi
+
w_{phys} L_{bicycle}
+
w_{param} L_{param}
+
w_{reg} L_{reg}
$$

---

## 9.1 `L_xy`
位置差分の一致

$$
L_{xy}
=
\text{Huber}\!\left(\frac{\Delta x^{pred}-\Delta x^{true}}{\sigma_{\Delta x}},0\right)
+
\text{Huber}\!\left(\frac{\Delta y^{pred}-\Delta y^{true}}{\sigma_{\Delta y}},0\right)
$$

現状で最も重視している loss の一つ。

---

## 9.2 `L_v`
縦速度差分の一致

$$
L_v
=
\text{Huber}\!\left(\frac{\Delta v_x^{pred}-\Delta v_x^{true}}{\sigma_{\Delta v_x}},0\right)
$$

`dX,dY` とは別の縦方向情報として補助的に効く。

---

## 9.3 `L_r`
yaw proxy との一致

$$
L_r
=
\text{Huber}\!\left(\frac{(r^{pred}-r^{proxy})\cdot w_{yaw}}{\sigma_r},0\right)
$$

ここで `w_yaw` は `is_yaw_reliable` に相当する重み。

ただし、`r` を proxy 固定する実験ではこの loss は実質 0 にしている。

---

## 9.4 `L_\psi`
heading 変化の整合

$$
L_\psi
=
\text{Huber}\!\left(\frac{(\Delta\psi^{pred}-\Delta\psi^{true})\cdot w_{yaw}}{\sigma_{\Delta\psi}},0\right)
$$

現状では補助的な項であり、切る場合もある。

---

## 9.5 `L_{bicycle}`
物理コアからの逸脱を抑える loss

$$
L_{bicycle}
=
\text{Huber}\!\left(\frac{\dot\beta_{\text{eff}}-\dot\beta_{\text{phys}}}{\sigma_{\dot\beta}},0\right)
+
\text{Huber}\!\left(\frac{\dot r_{\text{eff}}-\dot r_{\text{phys}}}{\sigma_{\dot r}},0\right)
$$

現状の `r` proxy 固定版では、実質的には `beta` 側中心になることがある。

---

## 9.6 `L_{param}`
物理パラメータの prior 罰則

$$
L_{param}
=
\left(\frac{\mu/\mu_0-1}{\sigma_\mu}\right)^2
+
\left(\frac{C_f/C_{f0}-1}{\sigma_{Cf}}\right)^2
+
\left(\frac{C_r/C_{r0}-1}{\sigma_{Cr}}\right)^2
+
\left(\frac{I_z/I_{z0}-1}{\sigma_{Iz}}\right)^2
+
\left(\frac{m/m_0-1}{\sigma_m}\right)^2
$$

これは物理パラメータが初期値・事前想定から離れすぎないようにするための項である。

---

## 9.7 `L_{reg}`
補助入力・補正の暴れを抑える正則化

例:

- `delta_eff` の大きさ
- `delta_eff` の時間差分
- `dvx_eff` が観測加速度から離れすぎないこと

などを抑える。

つまり、**未観測入力や残差自由度が過度に fit のためだけに使われないようにする**項である。

---

## 10. 現状の学習戦略

現状は、主に `dX, dY, dVx` を優先してフィットし、  
その中で物理パラメータや bicycle 構造をなるべく壊さないようにしている段階である。

ただし実験の結果、

- `r` を物理予測すると不安定になりやすい
- `r = yaw_like_rate` とした方が `dX, dY` は大きく改善する

ことが分かっている。

したがって現状は、

- **yaw は proxy ベース**
- **位置差分と縦速度差分を主に fit**
- **物理コアは `beta` と横力の整合に使う**

という、やや実用寄りの構成になっている。

---

## 11. 現時点でのまとめ

現状モデルは、

- CAN / GPS から構成した
  - `vx`
  - `ax`
  - `throttle`
  - `rpm`
  - `yaw_like_rate`
- および車両初期パラメータ
  - `mass`
  - `wheelbase`
  - `Cf, Cr`
  - `Iz`
  - `mu`

を用いて、簡略 dynamic bicycle model を構成している。

その上で、最終的には主に

- `dX`
- `dY`
- `dVx`

を実測差分に合わせるよう学習している。

一方で、

- `r`
- `psi`
- `beta`

については、観測の制約が弱く識別が難しいため、現状は `yaw_like_rate` を直接使うなど安定化寄りの工夫を入れている。

したがって、現状モデルの位置づけは

**「物理構造を持ちながらも、まずは局所差分予測を安定に合わせることを優先した physics-informed 実用版」**

と整理できる。