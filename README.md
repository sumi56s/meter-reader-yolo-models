# meter-reader-yolo-models

固定カメラで撮った計測器の画像から値を読み取るための、学習済み YOLO11n モデル4種です。
計測器の位置は検出しません。**計測器1個分を切り出した画像（ROI）** を入力する前提です。

## モデル一覧

| ファイル | 対象 | タスク | クラス | 推論サイズ |
|---|---|---|---|---|
| `models/needle.pt` | 針型メーター（丸・角） | pose | `meter`（キーポイント2点: 0=針の軸、1=針先） | 256 |
| `models/digits.pt` | 7セグ表示（赤・緑） | detect | 数字 0〜9 | 320 |
| `models/counter.pt` | 機械式カウンタ（数字ドラム） | detect | 数字 0〜9 | 320 |
| `models/float.pt` | 面積式流量計の浮き子 | detect | `float` | 320 |
| `models/tower_clf.pkl` | 積層信号灯（赤・黄・緑の3段）の段ごとの点灯判定 | 分類（scikit-learn 1.9.1） | 点灯/消灯 | - |

いずれも Ultralytics YOLO11n をもとに学習しています。ファイルが壊れていないかは `models/SHA256SUMS.txt` と照合できます。

```
# Windows
certutil -hashfile models\needle.pt SHA256
```

## ダウンロード

ログインは不要です。

- リポジトリ一式: 緑の「Code」ボタン → 「Download ZIP」
- モデル単体: `https://github.com/sumi56s/meter-reader-yolo-models/raw/main/models/needle.pt`

Databricks のノートブックから Volume に保存する例:

```python
import urllib.request
base = "https://github.com/sumi56s/meter-reader-yolo-models/raw/main/models/"
dst = "/Volumes/<catalog>/<schema>/<volume>/meter_models/"
dbutils.fs.mkdirs(dst)
for f in ["needle.pt", "digits.pt", "counter.pt", "float.pt", "tower_clf.pkl", "SHA256SUMS.txt"]:
    urllib.request.urlretrieve(base + f, dst + f)
```

## 使い方

```bash
pip install ultralytics
```

### 針型: 2点から角度を出し、目盛りに換算

```python
import math
from ultralytics import YOLO

r = YOLO("models/needle.pt")("needle_roi.jpg", imgsz=256, verbose=False)[0]
(ax, ay), (tx, ty) = r.keypoints.xy[0][:2].tolist()     # 0=軸, 1=針先
ang = math.degrees(math.atan2(ay - ty, tx - ax))        # 右=0°, 反時計回りが正

# 校正値: 目盛り最小・最大のときの針の角度と値（計測器ごとに設定）
a_min, a_max, v_min, v_max = 225.0, -45.0, 0.0, 1.0
value = v_min + ((a_min - ang) % 360) / ((a_min - a_max) % 360) * (v_max - v_min)
```

### 7セグ・カウンタ: 検出した数字を左から並べる

```python
from ultralytics import YOLO

r = YOLO("models/digits.pt")("digits_roi.jpg", imgsz=320, conf=0.4, verbose=False)[0]
boxes = sorted(zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()), key=lambda b: b[0][0])
text = "".join(r.names[int(c)] for _, c in boxes)       # 例: "1234"
```

小数点とマイナス記号は検出しません。小数点は桁位置を固定で決めてください。
重なった検出が出る場合は、x方向に重なる箱のうち信頼度の低い方を除きます。
`counter.pt` も同じ手順です。

### 浮き子: 高さを目盛りに換算

```python
from ultralytics import YOLO

r = YOLO("models/float.pt")("float_roi.jpg", imgsz=320, verbose=False)[0]
x1, y1, x2, y2 = r.boxes.xyxy[r.boxes.conf.argmax()].tolist()
y = y1                                                   # 読み位置（上端で読む場合）

# 校正値: 目盛り最小・最大の画像上のy座標と値
y_min, y_max, v_min, v_max = 300.0, 20.0, 0.0, 10.0
value = v_min + (y - y_min) / (y_max - y_min) * (v_max - v_min)
```

## 対応している見た目

- 針型: 丸枠・角枠、白・黒・クリームの文字盤、振れ角 90〜300°、細針・先細り・矢じり・尾付き・白針、色付きゾーン、置き針、長方形の広角メータ
- 浮き子: 球・コマ型・円筒・円錐・糸巻き形・ガイド棒付き円盤。ガラス管・アクリルブロック・金属枠の窓
- 未対応: 針が直線的に動くエッジワイズ形、2針式、浮き子が外から見えない金属管式

## 学習データ

プログラムで描画した合成画像と、GitHub 上の公開データで学習しています。

| 用途 | リポジトリ |
|---|---|
| 針型 | jomjol/neural-network-analog-needle-readout、oci-labs/deep-gauge |
| カウンタ | jomjol/neural-network-digital-counter-readout |

上記にはライセンス表記のないリポジトリが含まれます。検証・試作用として使い、
本番で使う場合は自分の現場画像で追加学習することを勧めます。

## 更新履歴

- 2026-11-22
  - `needle.pt` v2: 針の軸がカバーに隠れて見えない差圧計（広角の窓型）に対応。従来の丸型の読みも維持
  - `float.pt` v2: 金属枠の細い窓の流量計（傾いて写るもの）に対応。白い浮き子、球・コマ・円盤・円筒形
  - `tower_clf.pkl` を追加: 段の中心2点（上段・下段）をクリックで決め、各段の点灯を判定する。読み取りコード `tower.py` と組み合わせて使う
- 2026-11-18 初版（needle / digits / counter / float）

## ライセンス

Ultralytics YOLO（AGPL-3.0）をもとに学習した重みのため、AGPL-3.0 で公開します。
