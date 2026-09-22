"""疑似異常画像の生成（設計書 §13）。

正常画像に合成欠陥を載せて (疑似異常画像, マスク) を返す。半導体・外観検査で
有効な scratch / spot / particle / contamination / brightness / blur と、
CutPaste（自己パッチ貼付）を実装する。numpy uint8 HWC 上で動作する。

乱数は呼び出し側から ``numpy.random.Generator`` を渡す（再現性のため）。
"""
import numpy as np
import PIL.Image
import PIL.ImageDraw

try:
    import cv2  # 線・楕円・ブラー描画に使用（無ければ簡易版にフォールバック）
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

ANOMALY_TYPES = (
    "scratch", "spot", "particle", "contamination", "brightness", "blur", "cutpaste",
    "blob",
)

# blob の既定設定。サイズは短辺に対する比（＝外接円半径/短辺）。
# 既定値は実 test 欠陥マスク（116枚）の実測分布に合わせている：
#   欠陥数/画像 中央値1・p90=3、半径r p5=0.005・中央値0.022・p90=0.123
# 色は test の欠陥が単色（グレースケール）なので既定で白〜黒の単色。
BLOB_SHAPES = ("tri", "rect", "penta", "hexa", "circle", "ellipse")
DEFAULT_BLOB_CFG = {
    "count_min": 1,
    "count_max": 3,
    "size_min": 0.005,
    "size_max": 0.13,
    "size_dist": "log",      # log=対数一様（実欠陥の裾広がりに整合） / uniform
    "shapes": BLOB_SHAPES,
    "color_mode": "gray",    # gray=白〜黒の単色 / rgb=ランダム色
    "gray_min": 0,
    "gray_max": 255,
}
# 形状バリアント（S2/S3）の既定。既定値は現行 poly＝従来挙動と完全一致（純粋追加）。
# irregular は実欠陥の形状統計（inpainting train GT 925 枚）に合わせ込む：
#   外接矩形充填率 中央0.59 / 円形度 中央0.38 / 縦横比 中央1.33・p90 2.04
DEFAULT_BLOB_CFG.update({
    "shape_family": "poly",      # poly=従来の凸多角形 / irregular=不定形（S2）
    "aspect_min": 1.0,           # irregular: 縦横比の下限（対数一様）
    "aspect_max": 2.2,           # irregular: 縦横比の上限
    "rough_amp": 0.65,           # irregular: 輪郭の凹凸振幅（半径比）
    "rough_kmax": 10,            # irregular: 輪郭ノイズの最大次数
    "irregular_scale": 0.78,     # irregular: 半径補正（実面積分布を poly と揃えるための係数）
    "scratch_ratio": 0.15,       # irregular: 細長い擦り傷として描く割合
    "edge_blend": "hard",        # hard=ベタ塗り / soft=境界ぼかし合成（S3）
    "edge_sigma_frac": 0.20,     # soft: 境界ぼかし σ＝この比×図形半径（サイズ分布を保つ）
    "edge_sigma_min": 0.6,       # soft: σ の下限[px]
    "edge_sigma_max": 3.0,       # soft: σ の上限[px]
})

_REGULAR_NVERTS = {"tri": 3, "penta": 5, "hexa": 6}


def _rand_color(rng):
    return tuple(int(c) for c in rng.integers(0, 256, size=3))


def _scratch(img, rng):
    """細い直線状の傷。"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
    length = int(rng.integers(min(h, w) // 4, min(h, w)))
    ang = float(rng.uniform(0, np.pi))
    p2 = (int(p1[0] + length * np.cos(ang)), int(p1[1] + length * np.sin(ang)))
    thick = int(rng.integers(1, 4))
    dark = rng.random() < 0.5
    color = (0, 0, 0) if dark else (255, 255, 255)
    if _HAS_CV2:
        cv2.line(img, p1, p2, color, thick)
        cv2.line(mask, p1, p2, 255, thick)
    else:  # 簡易: 点列で描画
        for t in np.linspace(0, 1, length):
            x = int(p1[0] + t * (p2[0] - p1[0])); y = int(p1[1] + t * (p2[1] - p1[1]))
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = color; mask[y, x] = 255
    return img, mask


def _ellipse(img, rng, fill_color=None, particle=False):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    n = int(rng.integers(3, 12)) if particle else 1
    for _ in range(n):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        if particle:
            ax, ay = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        else:
            ax, ay = int(rng.integers(w // 20, w // 8)), int(rng.integers(h // 20, h // 8))
        ang = int(rng.integers(0, 180))
        color = fill_color if fill_color is not None else _rand_color(rng)
        if _HAS_CV2:
            cv2.ellipse(img, (cx, cy), (ax, ay), ang, 0, 360, color, -1)
            cv2.ellipse(mask, (cx, cy), (ax, ay), ang, 0, 360, 255, -1)
        else:
            y0, y1 = max(0, cy - ay), min(h, cy + ay); x0, x1 = max(0, cx - ax), min(w, cx + ax)
            img[y0:y1, x0:x1] = color; mask[y0:y1, x0:x1] = 255
    return img, mask


def _box_blur(patch, k):
    """cv2 非依存の簡易ボックスブラー（累積和ベース、各チャンネル）。"""
    p = patch.astype(np.float32)
    r = max(1, k // 2)
    pad = np.pad(p, ((r, r), (r, r), (0, 0)), mode="edge")
    csum = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0), (0, 0)), mode="constant")
    h, w = patch.shape[:2]
    ys = np.arange(h)[:, None]; xs = np.arange(w)[None, :]
    y0, y1 = ys, ys + 2 * r + 1; x0, x1 = xs, xs + 2 * r + 1
    total = (csum[y1, x1] - csum[y0, x1] - csum[y1, x0] + csum[y0, x0])
    out = total / ((2 * r + 1) ** 2)
    return np.clip(out, 0, 255).astype(np.uint8)


def _brightness(img, rng):
    """局所の明るさ変化（矩形領域をゲイン）。"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bw, bh = int(rng.integers(w // 8, w // 3)), int(rng.integers(h // 8, h // 3))
    x0, y0 = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
    gain = float(rng.uniform(0.4, 1.8))
    region = img[y0:y0 + bh, x0:x0 + bw].astype(np.float32) * gain
    img[y0:y0 + bh, x0:x0 + bw] = np.clip(region, 0, 255).astype(np.uint8)
    mask[y0:y0 + bh, x0:x0 + bw] = 255
    return img, mask


def _blur(img, rng):
    """局所のぼけ。"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bw, bh = int(rng.integers(w // 8, w // 3)), int(rng.integers(h // 8, h // 3))
    x0, y0 = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
    patch = img[y0:y0 + bh, x0:x0 + bw]
    k = int(rng.integers(3, 8)) * 2 + 1
    if _HAS_CV2:
        patch = cv2.GaussianBlur(patch, (k, k), 0)
    else:
        patch = _box_blur(patch, k)
    img[y0:y0 + bh, x0:x0 + bw] = patch
    mask[y0:y0 + bh, x0:x0 + bw] = 255
    return img, mask


def _cutpaste(img, rng):
    """別領域のパッチを切り出して貼り付ける（CutPaste）。"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    pw, ph = int(rng.integers(w // 10, w // 4)), int(rng.integers(h // 10, h // 4))
    sx, sy = int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))
    dx, dy = int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))
    patch = img[sy:sy + ph, sx:sx + pw].copy()
    img[dy:dy + ph, dx:dx + pw] = patch
    mask[dy:dy + ph, dx:dx + pw] = 255
    return img, mask


def _shape_vertices(shape, cx, cy, r, ang0, rng):
    """図形の頂点列（回転込み）を返す。円/楕円は多角形近似（回転可能）。"""
    if shape in ("circle", "ellipse"):
        rx = r
        ry = r if shape == "circle" else max(1, int(r * rng.uniform(0.3, 1.0)))
        m = 24
        ca, sa = np.cos(ang0), np.sin(ang0)
        pts = []
        for k in range(m):
            t = 2 * np.pi * k / m
            ex, ey = rx * np.cos(t), ry * np.sin(t)
            pts.append((cx + ex * ca - ey * sa, cy + ex * sa + ey * ca))
        return pts
    if shape == "rect":
        rx = r
        ry = max(1, int(r * rng.uniform(0.3, 1.0)))
        base = [(-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry)]
        ca, sa = np.cos(ang0), np.sin(ang0)
        return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in base]
    n = _REGULAR_NVERTS[shape]
    return [(cx + r * np.cos(ang0 + 2 * np.pi * k / n),
             cy + r * np.sin(ang0 + 2 * np.pi * k / n)) for k in range(n)]


def _irregular_vertices(cx, cy, r, ang0, rng, cfg):
    """不定形の輪郭頂点（S2）。等方円に半径方向の調和ノイズ＋異方性を掛ける。

    r(θ) = r・(1 + Σ_k a_k cos(kθ+φ_k)) を x 方向 √A 倍・y 方向 1/√A 倍に伸ばす
    （面積を保ったまま縦横比 A を作る）。凹凸により凸多角形より充填率・円形度が下がる。
    """
    m = 64
    t = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    rad = np.ones(m, dtype=np.float64)
    amp = float(cfg["rough_amp"])
    for k in range(2, int(cfg["rough_kmax"]) + 1):
        rad += (amp / (k - 1)) * float(rng.uniform(0.3, 1.0)) * np.cos(k * t + float(rng.uniform(0, 2 * np.pi)))
    rad = np.clip(rad, 0.30, 1.80)
    a = float(np.exp(rng.uniform(np.log(max(cfg["aspect_min"], 1e-6)), np.log(max(cfg["aspect_max"], 1e-6)))))
    sx, sy = np.sqrt(a), 1.0 / np.sqrt(a)
    ex, ey = r * rad * np.cos(t) * sx, r * rad * np.sin(t) * sy
    ca, sa = np.cos(ang0), np.sin(ang0)
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in zip(ex, ey)]


def _scratch_polyline(cx, cy, r, ang0, rng, cfg):
    """細長い擦り傷（S2）。折れ線の座標列と線幅を返す。"""
    length = r * float(rng.uniform(3.0, 8.0))
    width = max(1, int(round(r * float(rng.uniform(0.25, 0.7)))))
    n = 6
    ca, sa = np.cos(ang0), np.sin(ang0)
    pts = []
    for k in range(n):
        u = length * (k / (n - 1) - 0.5)
        v = r * float(rng.uniform(-0.6, 0.6))
        pts.append((cx + u * ca - v * sa, cy + u * sa + v * ca))
    return pts, width


def _blob_shape_paint(targets, shape_family, rng, cfg, cx, cy, r, ang0, shape):
    """図形 1 個を (Draw, 塗り値) の各対へ描く（形状ファミリ切替）。"""
    if shape_family == "irregular":
        # 不定形は凸多角形より実面積が大きく出るため、poly と面積分布を揃える補正を掛ける
        r = max(1, int(round(r * float(cfg["irregular_scale"]))))
    if shape_family == "irregular" and rng.random() < float(cfg["scratch_ratio"]):
        pts, width = _scratch_polyline(cx, cy, r, ang0, rng, cfg)
        ipts = [(int(x), int(y)) for x, y in pts]
        for draw, fill in targets:
            draw.line(ipts, fill=fill, width=width, joint="curve")
        return r
    if shape_family == "irregular":
        verts = _irregular_vertices(cx, cy, r, ang0, rng, cfg)
    else:
        verts = _shape_vertices(shape, cx, cy, r, ang0, rng)
    pts = [(int(x), int(y)) for x, y in verts]
    for draw, fill in targets:
        draw.polygon(pts, fill=fill)
    return r


def _blob(img, rng, cfg=None):
    """雑な合成欠陥：単色の図形をランダムに複数描画する。

    サイズ・個数のランダム上限は ``cfg``（DEFAULT_BLOB_CFG を上書き）で制御する。
    ``shape_family`` で凸多角形(poly, 従来)／不定形(irregular, S2)を切り替え、
    ``edge_blend="soft"`` で境界のみをぼかして合成する（S3。塗り色分布は不変）。
    """
    cfg = {**DEFAULT_BLOB_CFG, **(cfg or {})}
    h, w = img.shape[:2]
    short = min(h, w)
    shapes = tuple(cfg["shapes"])
    family = cfg.get("shape_family", "poly")
    soft = cfg.get("edge_blend", "hard") == "soft"
    if soft:
        from scipy.ndimage import gaussian_filter
    pim = PIL.Image.fromarray(img)
    mim = PIL.Image.fromarray(np.zeros((h, w), np.uint8))
    dimg = PIL.ImageDraw.Draw(pim)
    dmask = PIL.ImageDraw.Draw(mim)
    canvas = np.asarray(pim, dtype=np.float32) if soft else None
    macc = np.zeros((h, w), np.uint8) if soft else None
    n = int(rng.integers(cfg["count_min"], cfg["count_max"] + 1))
    for _ in range(n):
        # poly は従来どおり先に図形種を引く（既存 pack の乱数列と互換を保つ）
        shape = shapes[int(rng.integers(0, len(shapes)))] if family == "poly" else None
        if cfg.get("size_dist", "uniform") == "log":
            lo = np.log(max(float(cfg["size_min"]), 1e-6))
            hi = np.log(max(float(cfg["size_max"]), 1e-6))
            frac = float(np.exp(rng.uniform(lo, hi)))
        else:
            frac = float(rng.uniform(cfg["size_min"], cfg["size_max"]))
        r = max(1, int(frac * short))
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        if cfg.get("color_mode", "rgb") == "gray":
            v = int(rng.integers(int(cfg["gray_min"]), int(cfg["gray_max"]) + 1))
            color = (v, v, v)
        else:
            color = _rand_color(rng)
        ang0 = float(rng.uniform(0, 2 * np.pi))
        if not soft:
            _blob_shape_paint([(dimg, color), (dmask, 255)], family, rng, cfg,
                              cx, cy, r, ang0, shape)
            continue
        # soft：図形ごとに α を作り、σ を半径比例にしてサイズ分布を保ったまま境界だけ滑らかにする
        tim = PIL.Image.fromarray(np.zeros((h, w), np.uint8))
        r_used = _blob_shape_paint([(PIL.ImageDraw.Draw(tim), 255)], family, rng, cfg,
                                   cx, cy, r, ang0, shape)
        sigma = float(np.clip(float(cfg["edge_sigma_frac"]) * r_used,
                              float(cfg["edge_sigma_min"]), float(cfg["edge_sigma_max"])))
        alpha = np.clip(gaussian_filter(np.asarray(tim, dtype=np.float32) / 255.0, sigma=sigma), 0.0, 1.0)
        canvas = canvas * (1.0 - alpha[..., None]) + np.asarray(color, np.float32) * alpha[..., None]
        np.maximum(macc, ((alpha > 0.5) * 255).astype(np.uint8), out=macc)
    if soft:
        return np.clip(canvas, 0, 255).astype(np.uint8), macc
    return np.asarray(pim).copy(), np.asarray(mim).copy()


_GENERATORS = {
    "scratch": lambda img, rng: _scratch(img, rng),
    "spot": lambda img, rng: _ellipse(img, rng),
    "particle": lambda img, rng: _ellipse(img, rng, fill_color=(0, 0, 0), particle=True),
    "contamination": lambda img, rng: _ellipse(img, rng),
    "brightness": lambda img, rng: _brightness(img, rng),
    "blur": lambda img, rng: _blur(img, rng),
    "cutpaste": lambda img, rng: _cutpaste(img, rng),
}


def generate(image: np.ndarray, rng, types=ANOMALY_TYPES, blob_cfg=None):
    """正常画像 [H,W,3] uint8 に疑似異常を 1 種付与し (疑似画像, マスク) を返す。

    ``blob_cfg`` は ``blob`` タイプのサイズ・個数上限等（DEFAULT_BLOB_CFG を上書き）。
    """
    img = image.copy()
    kind = types[int(rng.integers(0, len(types)))]
    if kind == "blob":
        img, mask = _blob(img, rng, blob_cfg)
    else:
        img, mask = _GENERATORS[kind](img, rng)
    return img, mask, kind
