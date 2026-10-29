"""Control-room view for live inspection: the camera picture plus counts, a close-up of the latest defect,
recent defects, a timeline and a map of where defects appear, all drawn into one image with OpenCV."""
import time

import cv2
import numpy as np

import inspector as I

W, H = 1600, 920
BG, PANEL, BORDER = (30, 22, 15), (46, 35, 25), (78, 62, 48)
TXT, MUTED, CYAN = (240, 240, 240), (170, 155, 140), (235, 200, 0)
RED, GREEN, AMBER = (70, 70, 235), (90, 205, 80), (0, 185, 255)
F = cv2.FONT_HERSHEY_SIMPLEX
CLASSES = ['Hole', 'Knot', 'Line', 'Stain', 'unknown']


def put(img, s, org, scale, color=TXT, thick=1):
    cv2.putText(img, str(s), (int(org[0]), int(org[1])), F, scale, color, thick, cv2.LINE_AA)


def box(img, x, y, w, h, title=None, fill=PANEL):
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), BORDER, 1)
    if title:
        put(img, title, (x + 14, y + 26), 0.52, MUTED, 1)


def fit_into(img, src, x, y, w, h):
    """Paste src into the rectangle, keeping its shape (letterboxed)."""
    if src is None or src.size == 0:
        return
    s = min(w / src.shape[1], h / src.shape[0])
    nw, nh = max(1, int(src.shape[1] * s)), max(1, int(src.shape[0] * s))
    r = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    img[oy:oy + nh, ox:ox + nw] = r


def mmss(sec):
    sec = int(max(sec, 0))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def grade_color(g):
    return {'A': GREEN, 'B': AMBER, 'C': RED}.get(g, MUTED)


def compose(video, snap, summary, info):
    """video: the annotated camera frame (BGR). snap: LiveEngine.snapshot(). summary: LiveEngine.summary().
    info: dict with camera, fps, model_fps, device, detector, source."""
    img = np.full((H, W, 3), BG, np.uint8)
    counted = snap['counted']

    # header
    cv2.rectangle(img, (20, 16), (66, 62), CYAN, -1)
    put(img, "FDI", (25, 48), 0.6, BG, 2)
    put(img, "FABRIC", (82, 42), 0.95, CYAN, 2)
    put(img, "DEFECT INSPECTOR", (205, 42), 0.95, TXT, 2)
    put(img, f"{info.get('detector', 'YOLO11s + PatchCore')}  |  " +
        info.get('features', "ID tracking  |  counting zone  |  inspection region"),
        (84, 64), 0.45, MUTED)
    badge = info.get('badge', 'LIVE')
    blink = int(time.time() * 2) % 2 == 0 if badge == 'LIVE' else True
    bw = 40 + cv2.getTextSize(badge, F, 0.6, 2)[0][0]
    cv2.rectangle(img, (1390 - bw - 8, 24), (1390, 56), (40, 30, 110) if badge == 'LIVE' else (90, 60, 30), -1)
    cv2.circle(img, (1390 - bw + 10, 40), 7, (RED if badge == 'LIVE' else CYAN) if blink else (40, 40, 140), -1, cv2.LINE_AA)
    put(img, badge, (1390 - bw + 24, 47), 0.6, (200, 200, 255) if badge == 'LIVE' else TXT, 2)
    put(img, mmss(snap['elapsed']), (1415, 48), 0.85, TXT, 2)
    put(img, time.strftime("%d-%m-%Y  %H:%M:%S"), (1415, 70), 0.42, MUTED)

    # camera view
    vx, vy, vw, vh = 20, 82, 1040, 590
    box(img, vx, vy, vw, vh, fill=(10, 10, 10))
    fit_into(img, video, vx + 2, vy + 2, vw - 4, vh - 4)
    label = f"{info.get('camera', 'CAM 01')}  -  {info.get('source', 'FABRIC LINE')}".upper()
    (tw, _), _ = cv2.getTextSize(label, F, 0.5, 1)
    cv2.rectangle(img, (vx + 12, vy + 12), (vx + 32 + tw, vy + 40), (15, 15, 15), -1)
    put(img, label, (vx + 22, vy + 32), 0.5, TXT, 1)

    # KPI tiles
    rx = 1076
    tiles = [("TOTAL DEFECTS", summary['defects'], CYAN), ("IN VIEW NOW", snap['in_view'], TXT),
             ("DEFECTS PER 100 m", f"{summary['rate']:.0f}" if summary['metres'] > 0.05 else "-", AMBER),
             ("QUALITY GRADE", summary['grade'], grade_color(summary['grade']))]
    for i, (name, val, col) in enumerate(tiles):
        x, y = rx + (i % 2) * 256, 82 + (i // 2) * 122
        box(img, x, y, 248, 112, name)
        cv2.rectangle(img, (x + 14, y + 40), (x + 18, y + 96), col, -1)
        put(img, val, (x + 30, y + 92), 1.7, col, 3)

    # defects by type
    by = summary['by_type']
    box(img, rx, 330, 504, 168, "DEFECTS BY TYPE")
    put(img, f"{summary['metres']:.2f} m scanned", (rx + 360, 356), 0.45, MUTED)
    top = max([by.get(c, 0) for c in CLASSES] + [1])
    for i, c in enumerate(CLASSES):
        y = 362 + i * 26
        col = I.COLORS.get(c, CYAN)
        put(img, c.upper(), (rx + 16, y + 12), 0.48, TXT, 1)
        cv2.rectangle(img, (rx + 120, y), (rx + 440, y + 14), (70, 56, 44), -1)
        cv2.rectangle(img, (rx + 120, y), (rx + 120 + int(320 * by.get(c, 0) / top), y + 14), col, -1)
        put(img, by.get(c, 0), (rx + 456, y + 13), 0.55, TXT, 2)

    # live zoom on the latest defect
    box(img, rx, 508, 504, 164, "LATEST DEFECT  -  ZOOM")
    if counted:
        tid, lab, kind, conf, at, thumb, _ = counted[-1]
        col = I.COLORS.get(lab, CYAN)
        cv2.rectangle(img, (rx + 14, 540), (rx + 214, 662), (12, 12, 12), -1)
        fit_into(img, thumb, rx + 16, 542, 196, 118)
        cv2.rectangle(img, (rx + 228, 560), (rx + 490, 604), col, -1)
        dark = 0.114 * col[0] + 0.587 * col[1] + 0.299 * col[2] > 140
        put(img, f"#{tid} {lab.upper()}", (rx + 240, 592), 0.8, (20, 20, 20) if dark else TXT, 2)
        put(img, (f"confidence {conf:.2f}" if kind == 'known' else "flagged by PatchCore"), (rx + 230, 630), 0.5, MUTED)
        put(img, f"counted at {mmss(at)}", (rx + 230, 654), 0.5, MUTED)
    else:
        put(img, "no defect counted yet", (rx + 150, 600), 0.6, MUTED)

    # timeline
    tx, ty, tw_, th = 20, 688, 520, 210
    box(img, tx, ty, tw_, th, "DEFECT TIMELINE")
    put(img, "cumulative count", (tx + 360, ty + 26), 0.42, CYAN)
    gx, gy, gw, gh = tx + 46, ty + 44, tw_ - 66, th - 76
    span = max(30.0, snap['elapsed'])
    tl = snap['timeline']
    ymax = max([n for _, n in tl] + [5])
    for k in range(3):
        yy = gy + gh - int(gh * k / 2)
        cv2.line(img, (gx, yy), (gx + gw, yy), (70, 56, 44), 1)
        put(img, int(ymax * k / 2), (tx + 12, yy + 5), 0.42, MUTED)
    pts, last = [(gx, gy + gh)], 0
    for t, n in tl:
        x = gx + int(gw * t / span)
        pts += [(x, gy + gh - int(gh * last / ymax)), (x, gy + gh - int(gh * n / ymax))]
        last = n
    pts.append((gx + int(gw * snap['elapsed'] / span), gy + gh - int(gh * last / ymax)))
    cv2.polylines(img, [np.array(pts, np.int32)], False, CYAN, 2, cv2.LINE_AA)
    cv2.circle(img, pts[-1], 5, CYAN, -1, cv2.LINE_AA)
    put(img, "0:00", (gx - 6, gy + gh + 20), 0.42, MUTED)
    put(img, mmss(span)[3:], (gx + gw - 30, gy + gh + 20), 0.42, MUTED)

    # recent defects
    lx, ly, lw, lh = 556, 688, 504, 210
    box(img, lx, ly, lw, lh, "RECENT DEFECTS")
    for i, (tid, lab, kind, conf, at, thumb, _) in enumerate(reversed(counted[-3:])):
        y = ly + 40 + i * 56
        col = I.COLORS.get(lab, CYAN)
        cv2.rectangle(img, (lx + 12, y), (lx + lw - 12, y + 48), (58, 45, 34), -1)
        cv2.rectangle(img, (lx + 12, y), (lx + 17, y + 48), col, -1)
        cv2.rectangle(img, (lx + 26, y + 4), (lx + 78, y + 44), (12, 12, 12), -1)
        fit_into(img, thumb, lx + 27, y + 5, 50, 38)
        put(img, f"#{tid}  {lab.upper()}", (lx + 92, y + 22), 0.55, TXT, 1)
        put(img, (f"conf {conf:.2f}" if kind == 'known' else "unknown defect"), (lx + 92, y + 42), 0.45, col, 1)
        put(img, mmss(at), (lx + lw - 110, y + 30), 0.5, MUTED)
    if not counted:
        put(img, "waiting for the first defect", (lx + 130, ly + 120), 0.6, MUTED)

    # roll map: each counted defect at its place along the fabric (metres) and across its width
    mx, my, mw, mh = 1076, 688, 504, 210
    box(img, mx, my, mw, mh, "ROLL MAP")
    lx0 = mx + 130
    for c in CLASSES:                                       # legend
        col = I.COLORS.get(c, CYAN)
        cv2.circle(img, (lx0, my + 21), 5, col, -1, cv2.LINE_AA)
        put(img, c, (lx0 + 9, my + 26), 0.4, MUTED)
        lx0 += 18 + cv2.getTextSize(c, F, 0.4, 1)[0][0] + 12
    ax, ay, aw, ah = mx + 16, my + 40, mw - 32, mh - 78
    total_m = max(float(summary.get('metres', 0.0)), 0.0)
    span = 10.0 if total_m > 10 else max(1.0, total_m)      # show the last 10 m of a long roll
    m0 = max(0.0, total_m - span)
    cv2.rectangle(img, (ax, ay), (ax + aw, ay + ah), (66, 50, 37), -1)        # the fabric
    cv2.rectangle(img, (ax, ay), (ax + aw, ay + ah), BORDER, 1)
    step = next(s_ for s_ in (0.1, 0.25, 0.5, 1, 2, 5) if span / s_ <= 8)
    k = int(np.ceil(m0 / step))
    while k * step <= m0 + span + 1e-9:
        x = ax + int(aw * (k * step - m0) / span)
        cv2.line(img, (x, ay), (x, ay + ah), (88, 70, 55), 1)
        lab = f"{k * step:g} m"
        put(img, lab, (x - cv2.getTextSize(lab, F, 0.4, 1)[0][0] // 2, ay + ah + 18), 0.4, MUTED)
        k += 1
    if total_m > 0:                                         # how far the inspection has got
        cv2.line(img, (ax + int(aw * (total_m - m0) / span), ay - 4), (ax + int(aw * (total_m - m0) / span), ay + ah + 4), CYAN, 2)
    for tid, lab, kind, conf, at, thumb, where in counted:
        if not where or where[0] < m0:
            continue
        x = ax + int(aw * min((where[0] - m0) / span, 1.0))
        y = ay + 6 + int((ah - 12) * min(max(where[1], 0.0), 1.0))
        cv2.circle(img, (x, y), 7, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 5, I.COLORS.get(lab, CYAN), -1, cv2.LINE_AA)
    put(img, "along the fabric", (ax, my + mh - 8), 0.38, MUTED)
    put(img, "dot height = place across the width", (ax + aw - 250, my + mh - 8), 0.38, MUTED)

    # footer
    put(img, f"FPS {info.get('fps', 0):.1f}   |   models {info.get('model_fps', 0):.1f}/s   |   device {info.get('device', '-')}"
             f"   |   {len(counted)} defects counted", (20, H - 6), 0.45, MUTED)
    if snap.get('unfamiliar'):
        put(img, "Not the fabric the models know: counting paused", (1076, H - 6), 0.45, RED, 1)
    return img


class InspectorView:
    """Feeds the control-room view from the roll-position Inspector used by the demo roll and uploaded videos.
    It notices each defect the moment the Inspector counts it, and keeps its time, a close-up and its place."""

    def __init__(self, insp, fps):
        self.insp, self.fps = insp, max(float(fps or 30), 1.0)
        self.events, self.seen, self.timeline = [], {}, []

    def update(self, frame, offset, frame_idx):
        H, W = frame.shape[:2]
        at = frame_idx / self.fps
        for t in self.insp.counted(self.insp.end_px):
            if id(t) in self.seen:
                continue
            self.seen[id(t)] = len(self.events) + 1
            x1, y1, x2, y2 = t['box']
            fx1, fx2 = int(x1 - offset), int(x2 - offset)
            pad = max(12, int(0.3 * max(x2 - x1, y2 - y1)))
            crop = frame[max(int(y1) - pad, 0):min(int(y2) + pad, H), max(fx1 - pad, 0):min(fx2 + pad, W)]
            where = (((x1 + x2) / 2) / self.insp.px_per_m, min(max(((y1 + y2) / 2) / H, 0.0), 1.0))   # metres along, share across
            self.events.append({'id': len(self.events) + 1, 'track': t, 'at': at,
                                'thumb': crop.copy() if crop.size else None, 'where': where})
            self.timeline.append((at, len(self.events)))

    def snapshot(self, frame_idx, offset=0, width=0):
        L = self.insp.label
        counted = [(e['id'], L(e['track']), e['track']['kind'], e['track']['best'], e['at'], e['thumb'], e['where'])
                   for e in self.events]
        in_view = sum(1 for t in self.insp.tracks
                      if t['hits'] >= self.insp.min_hits and t['box'][2] > offset and t['box'][0] < offset + width)
        return {'counted': counted, 'in_view': in_view, 'timeline': list(self.timeline),
                'elapsed': frame_idx / self.fps, 'unfamiliar': False}
