"""Live camera inspection in the style of a conveyor counter.

A fixed camera looks down at fabric moving through the view. Every defect is followed with its own ID
(tracking), drawn with corner brackets and a centre dot, and counted once when it passes through the
counting zone in the middle of the view. Models run in a background thread on the newest frame, so the
picture never waits for them.

Used by app.py. No Streamlit code in here.
"""
import collections
import threading
import time

import cv2
import numpy as np

import inspector as I

CYAN = (235, 200, 0)          # BGR
DARK = (25, 25, 25)


# ----------------------------------------------------------------------------- tracking
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


class Track:
    def __init__(self, tid, kind, cls, conf, box, t):
        self.id, self.kind, self.box = tid, kind, [float(v) for v in box]
        self.votes = collections.Counter({cls: conf}) if kind == 'known' else collections.Counter()
        self.conf, self.hits, self.missed = conf, 1, 0
        self.counted, self.counted_at, self.first_seen = False, None, t
        self.prev_c = self.c = centre(box)

    @property
    def label(self):
        return self.votes.most_common(1)[0][0] if self.kind == 'known' and self.votes else 'unknown'


class ZoneCounter:
    """Follows detections from frame to frame and counts each one once when it reaches the counting zone.

    direction 'horizontal': fabric moves left/right, the zone is a vertical band in the middle.
    direction 'vertical'  : fabric moves up/down, the zone is a horizontal band in the middle."""

    def __init__(self, width, height, direction='horizontal', zone_share=0.16, min_hits=2, max_missed=6, roi=None):
        self.w, self.h, self.direction = width, height, direction
        rx1, ry1, rx2, ry2 = roi or (0, 0, width, height)
        self.roi = (int(rx1), int(ry1), int(rx2), int(ry2))
        half = zone_share / 2
        if direction == 'horizontal':
            mid = (rx1 + rx2) / 2
            self.zone = (int(mid - width * half), int(ry1), int(mid + width * half), int(ry2))
        else:
            mid = (ry1 + ry2) / 2
            self.zone = (int(rx1), int(mid - height * half), int(rx2), int(mid + height * half))
        self.min_hits, self.max_missed = min_hits, max_missed
        self.tracks, self.done, self.next_id = [], [], 1

    def _in_or_across_zone(self, t):
        z = self.zone
        (px, py), (cx, cy) = t.prev_c, t.c
        if self.direction == 'horizontal':
            lo, hi, a, b = z[0], z[2], px, cx
        else:
            lo, hi, a, b = z[1], z[3], py, cy
        inside = lo <= b <= hi
        crossed = (a < lo and b > hi) or (a > hi and b < lo)    # moved straight over the zone between two model frames
        return inside or crossed

    def update(self, dets, t=None):
        """dets: [(kind, cls, conf, x1, y1, x2, y2), ...] for one model frame, in display coordinates."""
        t = time.time() if t is None else t
        used = set()
        for tr in self.tracks:
            best, best_score = None, 0.0
            for j, d in enumerate(dets):
                if j in used:
                    continue
                box = d[3:]
                score = iou(tr.box, box)
                if score < 0.15:                                   # also accept a close centre (fast or small defects)
                    (ax, ay), (bx, by) = tr.c, centre(box)
                    size = max(tr.box[2] - tr.box[0], tr.box[3] - tr.box[1], box[2] - box[0], box[3] - box[1])
                    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                    score = 0.15 * (1 - dist / (0.8 * size)) if dist < 0.8 * size else 0.0
                if score > best_score:
                    best, best_score = j, score
            if best is None:
                tr.missed += 1
                continue
            used.add(best)
            kind, cls, conf, *box = dets[best]
            tr.box = [0.5 * n + 0.5 * o for n, o in zip(box, tr.box)]   # smooth the box so it does not jitter
            tr.prev_c, tr.c = tr.c, centre(tr.box)
            tr.hits += 1; tr.missed = 0
            if kind == 'known':
                tr.kind = 'known'; tr.votes[cls] += conf; tr.conf = max(tr.conf if tr.votes else 0, conf)
        for j, d in enumerate(dets):
            if j not in used:
                kind, cls, conf, *box = d
                self.tracks.append(Track(self.next_id, kind, cls, conf, box, t)); self.next_id += 1
        for tr in self.tracks:
            if not tr.counted and tr.hits >= self.min_hits and self._in_or_across_zone(tr):
                tr.counted, tr.counted_at = True, t
        alive = []
        for tr in self.tracks:
            if tr.missed > self.max_missed:
                if tr.counted:
                    self.done.append(tr)
            else:
                alive.append(tr)
        self.tracks = alive

    def counted(self):
        return self.done + [t for t in self.tracks if t.counted]

    def by_type(self):
        return dict(collections.Counter(t.label for t in self.counted()))


# ----------------------------------------------------------------------------- drawing
def corner_box(img, box, color, thick=3):
    x1, y1, x2, y2 = [int(v) for v in box]
    L = max(10, int(min(x2 - x1, y2 - y1) * 0.25))
    for (x, y, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (x, y), (x + dx * L, y), color, thick, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * L), color, thick, cv2.LINE_AA)


def text(img, s, org, scale, color=(255, 255, 255), thick=2):
    cv2.putText(img, s, (org[0] + 2, org[1] + 2), I.FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)   # shadow
    cv2.putText(img, s, org, I.FONT, scale, color, thick, cv2.LINE_AA)


def draw(frame, counter, camera_name='Camera 01', heat=None, pix_thr=3.0, unfamiliar=False, teaching=None, panel=True):
    out = frame.copy()
    H, W = out.shape[:2]
    rx1, ry1, rx2, ry2 = counter.roi
    if (rx1, ry1, rx2, ry2) != (0, 0, W, H):            # dim everything outside the inspection region
        dim = (out * 0.35).astype(np.uint8)
        dim[ry1:ry2, rx1:rx2] = out[ry1:ry2, rx1:rx2]
        out = dim
        cv2.rectangle(out, (rx1, ry1), (rx2 - 1, ry2 - 1), (0, 190, 255), 2, cv2.LINE_AA)
        cv2.putText(out, "INSPECTION REGION", (rx1 + 8, ry2 - 10), I.FONT, 0.5, (0, 190, 255), 1, cv2.LINE_AA)
    if heat is not None:                                  # optional anomaly heatmap, only where PatchCore is alarmed
        a = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
        m = (a > pix_thr * 0.85)[..., None]
        if m.any():
            hm = cv2.applyColorMap(np.clip(a / (pix_thr * 1.6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            out = np.where(m, cv2.addWeighted(out, 0.6, hm, 0.4, 0), out)
    # counting zone: tinted band with an outline
    x1, y1, x2, y2 = counter.zone
    tint = out.copy(); cv2.rectangle(tint, (x1, y1), (x2, y2), CYAN, -1)
    out = cv2.addWeighted(tint, 0.16, out, 0.84, 0)
    cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), CYAN, 2, cv2.LINE_AA)
    # tracked defects
    for t in counter.tracks:
        if t.hits < counter.min_hits or t.missed > 2:      # only defects seen just now; no stale boxes
            continue
        color = I.COLORS.get(t.label, CYAN)
        corner_box(out, t.box, color, 4 if t.counted else 3)
        cx, cy = [int(v) for v in t.c]
        cv2.circle(out, (cx, cy), 9, DARK, -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 7, CYAN if not t.counted else (80, 220, 80), -1, cv2.LINE_AA)
        tag = f"#{t.id} {t.label}" + (f" {t.conf:.2f}" if t.kind == 'known' else "")
        I.draw_label(out, tag, t.box[0], t.box[1], t.box[3], color, scale=0.55, min_top=58)
    # timestamp and camera name (the full dashboard view shows its own)
    if panel:
        text(out, time.strftime("%m-%d-%Y %a %H:%M:%S"), (16, 40), 1.0, thick=2)
        (tw, _), _ = cv2.getTextSize(camera_name, I.FONT, 0.9, 2)
        text(out, camera_name, (W - tw - 18, H - 20), 0.9)
    # total count panel (left out when the full dashboard view draws its own)
    total = len(counter.counted()) if panel else 0
    by = counter.by_type()
    detail = "   ".join(f"{k} {v}" for k, v in sorted(by.items())) or "no defects yet"
    pw = max(240, cv2.getTextSize(detail, I.FONT, 0.5, 1)[0][0] + 40) if panel else 0
    if panel:
        px, py, ph = 18, H - 122, 104
        bg = out.copy(); cv2.rectangle(bg, (px, py), (px + pw, py + ph), DARK, -1)
        out = cv2.addWeighted(bg, 0.78, out, 0.22, 0)
        cv2.rectangle(out, (px, py), (px + 6, py + ph), CYAN, -1)
        cv2.putText(out, "TOTAL COUNT", (px + 20, py + 26), I.FONT, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(out, str(total), (px + 20, py + 72), I.FONT, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, detail, (px + 20, py + 94), I.FONT, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    if unfamiliar:
        cv2.rectangle(out, (0, 54), (W, 98), (30, 30, 160), -1)
        text(out, "Not the fabric the models know. Nothing is counted.", (16, 84), 0.7)
    if teaching is not None:
        cv2.rectangle(out, (0, 54), (W, 98), (40, 120, 40), -1)
        text(out, f"Learning this fabric: keep clean cloth in view ({teaching})", (16, 84), 0.7)
    return out


# ----------------------------------------------------------------------------- engine
class LiveEngine:
    """Runs YOLO + PatchCore in a background thread on the newest frame and feeds the ZoneCounter."""

    def __init__(self, models, width, height, conf=0.25, pix_thr=3.0, patchcore_every=3, max_share=0.2,
                 direction='horizontal', zone_share=0.16, roi_share=(0, 0, 1, 1)):
        self.m, self.conf, self.pix_thr, self.pc_every, self.max_share = models, conf, pix_thr, max(1, patchcore_every), max_share
        l, t, r, b = roi_share                            # inspection region as shares of the frame: only this part is inspected
        self.roi = (int(width * l), int(height * t), int(width * r), int(height * b))
        self.counter = ZoneCounter(width, height, direction, zone_share, roi=self.roi)
        self.timeline, self.size = [], (width, height)    # (seconds, total) at every count, for the timeline chart
        self.lock, self.job, self.stop, self.error = threading.Lock(), None, False, None
        self.n, self.started_at, self.unfamiliar, self.heat = 0, time.time(), False, None
        self.travel_px, self.paused, self.px_per_m = 0.0, False, None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # -- detection on one frame (also used directly in tests)
    def detect(self, frame):
        rx1, ry1, rx2, ry2 = self.roi
        inside = lambda b: rx1 <= (b[0] + b[2]) / 2 <= rx2 and ry1 <= (b[1] + b[3]) / 2 <= ry2
        known = [k for k in self.m.yolo_detect(frame, self.conf) if inside(k[2:])]
        if self.n % self.pc_every == 0:
            amap = cv2.GaussianBlur(self.m.anomaly_map(frame).astype(np.float32), (0, 0), 1.0)
            c = I.CELL
            roi_cells = amap[ry1 // c:max(ry2 // c, ry1 // c + 1), rx1 // c:max(rx2 // c, rx1 // c + 1)]
            mask = np.zeros_like(amap); mask[ry1 // c:ry2 // c, rx1 // c:rx2 // c] = 1
            amap = amap * mask                            # nothing outside the inspection region can raise an alarm
            self.heat = amap
            self.unfamiliar = (self.max_share is not None and roi_cells.size > 0
                               and float((roi_cells > self.pix_thr).mean()) > self.max_share)
            unknown = [r for r in I.regions(amap, self.pix_thr) if not any(I.overlaps(r[1:], k[2:]) for k in known)]
        else:
            unknown = []
        self.n += 1
        if self.unfamiliar:
            return []
        return [('known', c, p, *b) for c, p, *b in known] + [('unknown', 'unknown', s, *b) for s, *b in unknown]

    def process(self, frame):
        dets = self.detect(frame)
        with self.lock:
            self.counter.update(dets)
            for t in self.counter.tracks:                 # newly counted: keep a close-up and its place for the dashboard
                if t.counted and not hasattr(t, 'thumb'):
                    x1, y1, x2, y2 = [int(v) for v in t.box]
                    pad = max(12, int(0.3 * max(x2 - x1, y2 - y1)))
                    H, W = frame.shape[:2]
                    crop = frame[max(y1 - pad, 0):min(y2 + pad, H), max(x1 - pad, 0):min(x2 + pad, W)]
                    t.thumb = crop.copy() if crop.size else None
                    across = t.c[1] / H if self.counter.direction == 'horizontal' else t.c[0] / W
                    t.where = (self.travel_px / self.px_per_m if self.px_per_m else 0.0, across)   # metres along, share across
                    self.timeline.append((t.counted_at - self.started_at, len(self.counter.counted())))

    def _run(self):
        while not self.stop:
            with self.lock:
                job, self.job = self.job, None
            if job is None or self.paused:
                time.sleep(0.003)
                continue
            try:
                self.process(job)
            except Exception as e:
                self.error = e
                return

    def submit(self, frame):
        with self.lock:
            self.job = frame

    def rate(self):
        return self.n / max(time.time() - self.started_at, 1e-6)

    def render(self, frame, camera_name='Camera 01', show_heat=False, teaching=None, panel=True):
        with self.lock:
            return draw(frame, self.counter, camera_name, self.heat if show_heat else None, self.pix_thr,
                        self.unfamiliar, teaching, panel)

    def snapshot(self):
        """What the dashboard view needs, copied under the lock."""
        with self.lock:
            done = sorted(self.counter.counted(), key=lambda t: t.counted_at)
            return {'counted': [(t.id, t.label, t.kind, t.conf, t.counted_at - self.started_at, getattr(t, 'thumb', None),
                                 getattr(t, 'where', None)) for t in done],
                    'in_view': sum(1 for t in self.counter.tracks if t.hits >= self.counter.min_hits and t.missed <= 2),
                    'timeline': list(self.timeline), 'elapsed': time.time() - self.started_at,
                    'unfamiliar': self.unfamiliar}

    def summary(self, px_per_m, a_max=5, b_max=15):
        with self.lock:
            done = self.counter.counted()
            by = self.counter.by_type()
        metres = self.travel_px / px_per_m if px_per_m else 0.0
        rate = len(done) / metres * 100 if metres > 0.05 else 0.0
        return {'metres': metres, 'defects': len(done), 'known': sum(t.kind == 'known' for t in done),
                'unknown': sum(t.kind == 'unknown' for t in done), 'rate': rate,
                'grade': I.grade(rate, a_max, b_max) if metres > 0.05 else '-', 'by_type': by}

    def log(self):
        with self.lock:
            done = list(self.counter.counted())
        return sorted([{'id': t.id, 'type': t.label, 'kind': t.kind,
                        'seconds_from_start': round(t.counted_at - self.started_at, 1),
                        'confidence': round(float(t.conf), 2) if t.kind == 'known' else None} for t in done],
                      key=lambda r: r['seconds_from_start'])

    def close(self, timeout=10):
        self.stop = True
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()
