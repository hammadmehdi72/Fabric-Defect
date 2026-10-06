"""Fabric Defect Inspector engine: YOLO + PatchCore fusion, count-once tracking, grading.

Used by app.py (the Streamlit dashboard). No Streamlit code in here, so it can also be used from a script.
"""
import collections
import pathlib
import random

import cv2
import numpy as np

CELL = 8             # one anomaly-map cell covers 8 x 8 px
MIN_CELLS = 4        # smallest anomalous region that counts
FRAME_H = 640        # frames are processed at this height (the models were trained on 640 px photos)
COLORS = {'Hole': (60, 60, 240), 'Knot': (0, 145, 255), 'Line': (255, 200, 0), 'Stain': (216, 50, 216),
          'unknown': (0, 220, 255)}   # BGR


# ----------------------------------------------------------------------------- models
class Models:
    """Loads YOLO and the PatchCore memory bank. Torch is imported here so the rest of the file works without it."""

    def __init__(self, yolo_path, patchcore_path, device=None):
        import torch
        import torch.nn.functional as F
        import torchvision
        from ultralytics import YOLO
        self.torch, self.F = torch, F
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo = YOLO(str(yolo_path))
        ck = torch.load(str(patchcore_path), map_location='cpu')
        self.bank = ck['bank'].float().to(self.dev)
        self.net = torchvision.models.wide_resnet50_2(
            weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(self.dev).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
        self.names = list(self.yolo.names.values())

    def yolo_detect(self, bgr, conf=0.25):
        """[(class name, confidence, x1, y1, x2, y2), ...] in pixels."""
        r = self.yolo.predict(bgr, imgsz=max(bgr.shape[:2]), conf=conf, verbose=False, device=self.dev)[0]
        return [(self.yolo.names[int(c)], float(p), *map(float, b))
                for b, p, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]

    def anomaly_map(self, bgr):
        """Distance of every 8 x 8 px patch to its nearest normal patch. Shape (H/8, W/8)."""
        torch, F = self.torch, self.F
        with torch.no_grad():
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None].to(self.dev)
            x = (x - self.mean) / self.std
            n = self.net
            x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            l2 = n.layer2(n.layer1(x)); l3 = n.layer3(l2)
            f2 = F.avg_pool2d(l2, 3, 1, 1)
            f3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=f2.shape[-2:], mode='bilinear', align_corners=False)
            f = torch.cat([f2, f3], 1)[0]
            C, h, w = f.shape
            q = f.permute(1, 2, 0).reshape(-1, C)
            d = torch.cat([torch.cdist(q[i:i + 2048], self.bank).min(1).values for i in range(0, q.shape[0], 2048)])
            return d.view(h, w).cpu().numpy()


# ----------------------------------------------------------------------------- geometry
def overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def same_track(a, b):
    """Same defect seen again: the boxes overlap well, or one mostly covers the other."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return False
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) >= 0.2 or inter / min(area_a, area_b) >= 0.5


def regions(amap_s, thr):
    """Anomalous regions in a smoothed map: [(score, x1, y1, x2, y2), ...] in pixels."""
    mask = (amap_s > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= MIN_CELLS:
            out.append((float(amap_s[lab == i].max()), x * CELL, y * CELL, (x + w) * CELL, (y + h) * CELL))
    return out


def grade(rate, a_max=5, b_max=15):
    return 'A' if rate <= a_max else ('B' if rate <= b_max else 'C')


# ----------------------------------------------------------------------------- the inspector
class Inspector:
    """Feed it frames with their position on the roll; it fuses both models and counts each defect once."""

    def __init__(self, models, px_per_m, conf=0.25, pix_thr=2.88, patchcore_every=3, min_hits=2):
        self.m, self.px_per_m = models, px_per_m
        self.conf, self.pix_thr, self.pc_every, self.min_hits = conf, pix_thr, max(1, patchcore_every), min_hits
        self.tracks, self.n = [], 0
        self.end_px = 0
        self.last_known, self.last_unknown, self.last_amap, self.last_amap_off = [], [], None, 0

    # -- tracking
    def _add(self, kind, cls, conf, box):
        for t in self.tracks:
            if same_track(box, t['box']) or (kind != t['kind'] and overlaps(box, t['box'])):
                t['hits'] += 1
                if kind == 'known':
                    if t['kind'] == 'unknown' or conf > t['best']:
                        t['box'], t['best'] = list(box), conf
                    t['kind'] = 'known'; t['votes'][cls] = t['votes'].get(cls, 0) + conf
                return
        self.tracks.append({'box': list(box), 'best': conf if kind == 'known' else 0.0,
                            'votes': {cls: conf} if kind == 'known' else {}, 'hits': 1, 'kind': kind})

    @staticmethod
    def label(t):
        return max(t['votes'], key=t['votes'].get) if t['kind'] == 'known' else 'unknown'

    def counted(self, upto=None):
        return [t for t in self.tracks if t['hits'] >= self.min_hits and (upto is None or t['box'][2] <= upto)]

    # -- one frame
    def process(self, frame, offset, ignore_cols=None, seam_x=()):
        """Run the models on this frame. frame: BGR image, FRAME_H high. offset: x position of the frame's
        left edge on the roll, in px. Returns the annotated frame."""
        W = frame.shape[1]
        known = self.m.yolo_detect(frame, self.conf)
        # a thin box sitting on a join between two photos is the join, not a defect (demo roll only)
        known = [k for k in known if not (k[4] - k[2] <= 64 and any(k[2] - 8 < sx < k[4] + 8 for sx in seam_x))]
        if self.n % self.pc_every == 0:
            amap_s = cv2.GaussianBlur(self.m.anomaly_map(frame).astype(np.float32), (0, 0), 1.0)
            if ignore_cols is not None and len(ignore_cols):
                amap_s[:, ignore_cols] = 0
            unknown = [r for r in regions(amap_s, self.pix_thr) if not any(overlaps(r[1:], k[2:]) for k in known)]
            self.last_amap, self.last_amap_off = amap_s, offset
            self.last_unknown = [(sc, offset + x1, y1, offset + x2, y2) for sc, x1, y1, x2, y2 in unknown]
        else:
            unknown = []
        for cls, conf, x1, y1, x2, y2 in known:
            if x1 > 2 and x2 < W - 2:                       # boxes cut by the frame edge are seen whole in other frames
                self._add('known', cls, conf, (offset + x1, y1, offset + x2, y2))
        for score, x1, y1, x2, y2 in unknown:
            if x1 > 2 and x2 < W - 2:
                self._add('unknown', None, score, (offset + x1, y1, offset + x2, y2))
        # remember what was found, in roll coordinates, so in-between frames can be drawn without the models
        self.last_known = [(cls, conf, offset + x1, y1, offset + x2, y2) for cls, conf, x1, y1, x2, y2 in known]
        self.last_unknown = [u for u in self.last_unknown if not any(overlaps(u[1:], k[2:]) for k in self.last_known)]
        self.n += 1
        self.end_px = max(self.end_px, offset + W)
        return self.render(frame, offset)

    def render(self, frame, offset, hud=None):
        """Draw the latest results on any frame, shifted to where the fabric is now. No models are run."""
        out = frame.copy()
        H, W = frame.shape[:2]
        self.end_px = max(self.end_px, offset + W)
        if self.last_amap is not None:
            a = self.last_amap
            heat = cv2.applyColorMap(np.clip(a / (self.pix_thr * 1.6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            mask = (a > self.pix_thr * 0.85).astype(np.uint8)
            if mask.any():
                aw = a.shape[1] * CELL
                M = np.float32([[1, 0, self.last_amap_off - offset], [0, 1, 0]])     # slide the map with the fabric
                heat = cv2.warpAffine(cv2.resize(heat, (aw, H), interpolation=cv2.INTER_LINEAR), M, (W, H))
                mask = cv2.warpAffine(cv2.resize(mask, (aw, H), interpolation=cv2.INTER_NEAREST), M, (W, H))[..., None]
                out = np.where(mask > 0, cv2.addWeighted(out, 0.6, heat, 0.4, 0), out)
        for cls, conf, x1, y1, x2, y2 in self.last_known:
            x1, x2 = x1 - offset, x2 - offset
            if x2 < 0 or x1 > W:
                continue
            c = COLORS.get(cls, (0, 255, 0))
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            cv2.putText(out, f"{cls} {conf:.2f}", (int(max(x1, 2)), max(int(y1) - 5, 46)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        for score, x1, y1, x2, y2 in self.last_unknown:
            x1, x2 = x1 - offset, x2 - offset
            if x2 < 0 or x1 > W:
                continue
            c = COLORS['unknown']
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            cv2.putText(out, f"unknown {score:.1f}", (int(max(x1, 2)), max(int(y1) - 5, 46)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        if hud:
            cv2.rectangle(out, (0, 0), (W, 30), (20, 20, 20), -1)
            cv2.putText(out, hud, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        return out

    # -- results
    def summary(self, a_max=5, b_max=15, final=False):
        done = self.counted(None if final else self.end_px)
        metres = self.end_px / self.px_per_m if self.px_per_m else 0.0
        rate = len(done) / metres * 100 if metres > 0 else 0.0
        by_type = collections.Counter(self.label(t) for t in done)
        return {'metres': metres, 'defects': len(done), 'known': sum(t['kind'] == 'known' for t in done),
                'unknown': sum(t['kind'] == 'unknown' for t in done), 'rate': rate,
                'grade': grade(rate, a_max, b_max) if metres > 0 else '-', 'by_type': dict(by_type)}

    def hud_text(self, a_max=5, b_max=15):
        s = self.summary(a_max, b_max)
        return (f"{s['metres']:5.2f} m   defects {s['defects']}   known {s['known']}   unknown {s['unknown']}   "
                f"per 100 m {s['rate']:.0f}   grade {s['grade']}")

    def log(self):
        rows = [{'type': self.label(t), 'kind': t['kind'], 'metres_from_start': round(float(t['box'][0] / self.px_per_m), 3),
                 'frames_seen': int(t['hits']), 'confidence': round(float(t['best']), 2) if t['kind'] == 'known' else None}
                for t in self.counted()]
        return sorted(rows, key=lambda r: r['metres_from_start'])


# ----------------------------------------------------------------------------- frame sources
def roll_from_photos(folder, n_images=40, view_w=960, step=4, seed=3):
    """Simulated roll: photos joined side by side, scrolled past a camera in small steps so the video is smooth.
    Yields (frame, offset, ignore_cols, seam_x) for every display frame; run the models on every Nth one.
    Returns (generator, total_frames, roll_length_px)."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    paths = sorted(p for p in pathlib.Path(folder).iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"no images in {folder}")
    random.Random(seed).shuffle(paths)
    tiles, seams, x0 = [], [], 0
    for p in paths[:n_images]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        if h != FRAME_H:
            im = cv2.resize(im, (max(int(round(w * FRAME_H / h / 32)) * 32, 32), FRAME_H))
        tiles.append(im); x0 += im.shape[1]; seams.append(x0)
    roll = np.concatenate(tiles, axis=1)
    offsets = list(range(0, max(roll.shape[1] - view_w, 0) + 1, step))

    def gen():
        for off in offsets:
            frame = np.ascontiguousarray(roll[:, off:off + view_w])
            sx = [s - off for s in seams if 0 < s - off < view_w]
            cols = sorted({c for s in sx for c in range(max(s // CELL - 5, 0), min(s // CELL + 5, view_w // CELL))})
            yield frame, off, cols, sx
    return gen(), len(offsets), roll.shape[1]


def fit_frame(frame):
    """Resize a video frame to FRAME_H high with a width that is a multiple of 32."""
    h, w = frame.shape[:2]
    new_w = max(int(round(w * FRAME_H / h / 32)) * 32, 32)
    return cv2.resize(frame, (new_w, FRAME_H), interpolation=cv2.INTER_AREA if h > FRAME_H else cv2.INTER_LINEAR)


class MotionEstimator:
    """Estimates how far the fabric moved between frames (px, along x) with phase correlation."""

    def __init__(self, scale=0.25):
        self.scale, self.prev, self.last = scale, None, 0.0
        self.win = None

    def update(self, frame):
        g = cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.prev is None or self.prev.shape != g.shape:
            self.prev, self.win = g, cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
            return 0.0
        (dx, _dy), resp = cv2.phaseCorrelate(self.prev, g, self.win)
        self.prev = g
        move = -dx / self.scale                               # fabric content moving left = roll position increasing
        if resp < 0.05 or abs(move) > frame.shape[1] * 0.4:   # unreliable match: keep the last speed
            move = self.last
        self.last = move
        return move


def video_frames(path):
    """Yields (frame, offset, None, ()) for every frame of a recorded video. Offset comes from the estimated
    fabric movement. Returns (generator, total_frames, fps)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def gen():
        est, off = MotionEstimator(), 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = fit_frame(frame)
            off += est.update(frame)
            yield frame, max(off, 0.0), None, ()
        cap.release()
    return gen(), total, fps


def to_h264(src, dst):
    """Re-encode to H.264 so browsers can play the video. Uses imageio-ffmpeg's bundled ffmpeg, or one on PATH.
    Returns True if it worked."""
    import shutil, subprocess
    exe = None
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which('ffmpeg')
    if not exe:
        return False
    r = subprocess.run([exe, '-y', '-loglevel', 'error', '-i', str(src), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(dst)], capture_output=True)
    return r.returncode == 0 and pathlib.Path(dst).exists()
