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

# blob（雑な図形の大量描画）の既定設定。サイズは短辺に対する比。
BLOB_SHAPES = ("tri", "rect", "penta", "hexa", "circle", "ellipse")
DEFAULT_BLOB_CFG = {
    "count_min": 3,
    "count_max": 15,
    "size_min": 0.02,
    "size_max": 0.30,
    "shapes": BLOB_SHAPES,
}
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


def _blob(img, rng, cfg=None):
    """雑な合成欠陥：単色の多角形/円/楕円をランダムに複数描画する。

    サイズ・個数のランダム上限は ``cfg``（DEFAULT_BLOB_CFG を上書き）で制御する。
    PIL.ImageDraw で塗るため cv2 非依存。
    """
    cfg = {**DEFAULT_BLOB_CFG, **(cfg or {})}
    h, w = img.shape[:2]
    short = min(h, w)
    shapes = tuple(cfg["shapes"])
    pim = PIL.Image.fromarray(img)
    mim = PIL.Image.fromarray(np.zeros((h, w), np.uint8))
    dimg = PIL.ImageDraw.Draw(pim)
    dmask = PIL.ImageDraw.Draw(mim)
    n = int(rng.integers(cfg["count_min"], cfg["count_max"] + 1))
    for _ in range(n):
        shape = shapes[int(rng.integers(0, len(shapes)))]
        r = max(1, int(rng.uniform(cfg["size_min"], cfg["size_max"]) * short))
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        color = _rand_color(rng)
        ang0 = float(rng.uniform(0, 2 * np.pi))
        pts = [(int(x), int(y)) for x, y in _shape_vertices(shape, cx, cy, r, ang0, rng)]
        dimg.polygon(pts, fill=color)
        dmask.polygon(pts, fill=255)
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
