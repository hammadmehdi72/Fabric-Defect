"""Make a dashboard-style demo video from your REAL Roboflow dataset (YOLO format).

Usage (Colab / Kaggle / your PC):
    pip install opencv-python pillow pyyaml numpy          # + ultralytics if you use --model
    python make_video_from_dataset.py --data Fabric-Defect-Thesis-1 --split test --n 40
    python make_video_from_dataset.py --data Fabric-Defect-Thesis-1 --model runs/detect/train/weights/best.pt

Without --model the boxes are the dataset's ground-truth labels.
With --model the boxes are your trained YOLO model's predictions (the real demo).
Images are joined side by side into one long "fabric roll" that scrolls past a fixed camera.
Uses ffmpeg if it is installed, otherwise OpenCV's own video writer (works on Windows).

Windows example (run from the Fabric folder):
    python make_video_from_dataset.py --data Textile.yolov11 --split train --n 40 --blend 60
"""
import argparse, glob, os, subprocess
import cv2, numpy as np, yaml
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="Roboflow export folder (contains data.yaml)")
ap.add_argument("--split", default="test", help="train / valid / test")
ap.add_argument("--n", type=int, default=40, help="number of images to use")
ap.add_argument("--model", default=None, help="optional YOLO weights (best.pt)")
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--out", default="fabric_inspector_real_data.mp4")
ap.add_argument("--credit", default="", help="dataset credit line shown at the bottom")
ap.add_argument("--raw", action="store_true", help="clean fabric-only video (no boxes, no dashboard) + ground-truth CSV, for testing your own pipeline")
ap.add_argument("--blend", type=int, default=0, help="px of cross-fade between neighbouring images (e.g. 60)")
ap.add_argument("--m_per_image", type=float, default=0.10, help="fabric length one image covers (metres)")
args = ap.parse_args()

W, H, FPS = 1280, 720, 30
VIEW_H = 360; VIEW_W = 640; SCALE = 1.25; STEP = 6; COUNT_X = VIEW_W // 2
names = yaml.safe_load(open(os.path.join(args.data, "data.yaml")))["names"]
names = list(names.values()) if isinstance(names, dict) else list(names)
PALETTE = [(150, 100, 255), (255, 80, 150), (210, 240, 60), (90, 170, 240), (240, 170, 50), (90, 200, 140)]
COLORS = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}

# ---------- build the strip + defect list ----------
img_paths = sorted(glob.glob(os.path.join(args.data, args.split, "images", "*")))[: args.n]
assert img_paths, "no images found - check --data and --split"
model = None
if args.model:
    from ultralytics import YOLO
    model = YOLO(args.model)

tiles, defects, x_off, img_w = [], [], 0, []
for p in img_paths:
    im = cv2.imread(p)
    if im is None: continue
    h0, w0 = im.shape[:2]; s = VIEW_H / h0; w1 = int(w0 * s)
    im = cv2.resize(im, (w1, VIEW_H)); tiles.append(im); img_w.append(w1)
    if model is not None:
        r = model(im, conf=args.conf, verbose=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            defects.append((model.names[int(b.cls)], x_off + (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(b.conf)))
    else:
        lp = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt"
        if os.path.exists(lp):
            for line in open(lp):
                v = line.split()
                if len(v) < 5: continue
                if len(v) > 5:   # segmentation polygon: class x1 y1 x2 y2 ... -> bounding box
                    xs, ys = list(map(float, v[1::2])), list(map(float, v[2::2]))
                    c, cx, cy = int(v[0]), (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
                else:
                    c, cx, cy, bw, bh = int(v[0]), *map(float, v[1:5])
                defects.append((names[c], x_off + cx * w1, cy * VIEW_H, bw * w1, bh * VIEW_H, 1.0))
    x_off += w1 - args.blend
# join images, cross-fading `blend` px at each seam so it reads as one roll
B = args.blend
strip = tiles[0].astype(np.float32)
for t in tiles[1:]:
    t = t.astype(np.float32)
    if B > 0:
        a = np.linspace(0, 1, B, dtype=np.float32)[None, :, None]
        seam = strip[:, -B:] * (1 - a) + t[:, :B] * a
        strip = np.concatenate([strip[:, :-B], seam, t[:, B:]], axis=1)
    else:
        strip = np.concatenate([strip, t], axis=1)
strip = np.concatenate([strip.astype(np.uint8), np.zeros((VIEW_H, COUNT_X + 20, 3), np.uint8)], axis=1)   # tail so the last defect crosses the line
STRIP_LEN = strip.shape[1]
PX_PER_M = (sum(img_w) / len(tiles)) / args.m_per_image
CLASSES = names

FONT_PAIRS = [("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
              ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
              ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
              ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")]
def font(sz, b=False):
    for reg, bold in FONT_PAIRS:
        try: return ImageFont.truetype(bold if b else reg, sz)
        except OSError: continue
    return ImageFont.load_default()
f_title, f_h, f_b, f_s, f_big = font(26, True), font(18, True), font(16), font(13), font(40, True)
grade = lambda r: "A" if r <= 5 else ("B" if r <= 15 else "C")
BG, PANEL, INK, QUIET, ACC = (17, 20, 26), (28, 32, 40), (235, 238, 242), (150, 158, 170), (90, 200, 140)
src = "YOLO predictions" if model else "ground-truth labels"

def render(off, counted, log, fi):
    view = strip[:, off:off + VIEW_W]
    view = cv2.resize(view, (int(VIEW_W * SCALE), int(VIEW_H * SCALE)))
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((32, 22), "Fabric Defect Inspector", font=f_title, fill=INK)
    d.text((400, 32), f"real dataset images  |  boxes: {src}", font=f_s, fill=QUIET)
    vx, vy = 32, 80
    im.paste(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)), (vx, vy))
    lx = vx + int(COUNT_X * SCALE)
    for yy in range(vy, vy + int(VIEW_H * SCALE), 14):
        d.line([(lx, yy), (lx, yy + 7)], fill=(255, 255, 255), width=2)
    for i, (t, cx, cy, w, h, c) in enumerate(defects):
        x0 = (cx - w / 2 - off) * SCALE; x1 = (cx + w / 2 - off) * SCALE
        x0c, x1c = max(x0, 0), min(x1, VIEW_W * SCALE - 1)
        if x1c - x0c < 4: continue
        y0 = max((cy - h / 2) * SCALE, 0); y1 = min((cy + h / 2) * SCALE, VIEW_H * SCALE - 1); col = COLORS.get(t, (255, 150, 50))
        d.rectangle([vx + x0c, vy + y0, vx + x1c, vy + y1], outline=col, width=3)
        lab = f"#{i+1} {t}" + (f" {c:.2f}" if model else "")
        tw = d.textlength(lab, font=f_s); ty = vy + y0 - 20 if y0 > 22 else vy + max(y0, 0) + 3
        d.rectangle([vx + x0c, ty, vx + x0c + tw + 8, ty + 18], fill=col)
        d.text((vx + x0c + 4, ty + 2), lab, font=f_s, fill=(15, 15, 15))
    d.text((vx, vy + int(VIEW_H * SCALE) + 10), f"frame {fi:04d}  |  {len(tiles)} images  |  {len(counted)} counted", font=f_s, fill=QUIET)
    lgx, lgy = vx, vy + int(VIEW_H * SCALE) + 36
    for t in CLASSES:
        d.rectangle([lgx, lgy + 3, lgx + 12, lgy + 15], fill=COLORS[t]); d.text((lgx + 18, lgy), t, font=f_s, fill=INK)
        lgx += 40 + d.textlength(t, font=f_s)
    d.text((vx, lgy + 30), "Pipeline:  frame -> PatchCore (anomaly?) -> YOLO (" + " / ".join(CLASSES) + "?) -> fuse -> count once -> grade",
           font=f_s, fill=QUIET)
    if args.credit:
        d.text((vx, 690), args.credit, font=f_s, fill=(200, 150, 90))
    px = 864; pw = W - px - 32
    metres = (off + VIEW_W) / PX_PER_M; n = len(counted); rate = n / metres * 100; g = grade(rate); cw = (pw - 12) // 2
    for k, (name, val) in enumerate([("Metres scanned", f"{metres:.2f} m"), ("Defects counted", str(n)),
                                     ("Defects / 100 m", f"{rate:.0f}"), ("Quality grade", g)]):
        cx0 = px + (k % 2) * (cw + 12); cy0 = 80 + (k // 2) * 104
        d.rounded_rectangle([cx0, cy0, cx0 + cw, cy0 + 92], 10, fill=PANEL)
        d.text((cx0 + 14, cy0 + 10), name, font=f_s, fill=QUIET)
        d.text((cx0 + 14, cy0 + 36), val, font=f_big,
               fill={"A": ACC, "B": (240, 190, 60), "C": (235, 90, 90)}[g] if name == "Quality grade" else INK)
    by = 300; d.rounded_rectangle([px, by, px + pw, by + 180], 10, fill=PANEL)
    d.text((px + 14, by + 10), "Defects by type", font=f_h, fill=INK)
    mx = max([sum(1 for i in counted if defects[i][0] == t) for t in CLASSES] + [1])
    for j, t in enumerate(CLASSES[:4]):
        cnt = sum(1 for i in counted if defects[i][0] == t); yy = by + 48 + j * 30
        d.text((px + 14, yy), t, font=f_s, fill=INK); bw = int(200 * cnt / mx)
        d.rectangle([px + 110, yy + 2, px + 110 + max(bw, 2), yy + 16], fill=COLORS[t])
        d.text((px + 116 + bw, yy), str(cnt), font=f_s, fill=INK)
    ly = 492; d.rounded_rectangle([px, ly, px + pw, ly + 184], 10, fill=PANEL)
    d.text((px + 14, ly + 10), "Detection log", font=f_h, fill=INK)
    for j, line in enumerate(log[-6:][::-1]):
        d.text((px + 14, ly + 40 + j * 23), line, font=f_s, fill=INK if j == 0 else QUIET)
    return im

import shutil
class Writer:
    """ffmpeg (H.264) when available, else OpenCV mp4v."""
    def __init__(self, path):
        self.ff = None
        if shutil.which("ffmpeg"):
            self.ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                        "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
                                       stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
            if not self.cv.isOpened(): raise SystemExit(f"could not open {path} for writing")
    def write(self, im):
        a = np.asarray(im, np.uint8)
        if self.ff: self.ff.stdin.write(a.tobytes())
        else: self.cv.write(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    def close(self):
        if self.ff: self.ff.stdin.close(); self.ff.wait()
        else: self.cv.release()
if args.raw:
    # clean scrolling-fabric video at 1280x720, plus a CSV of where every labelled defect is
    import csv
    W, H = 2 * VIEW_W, 2 * VIEW_H
    ff = Writer(args.out)
    fabric_len = STRIP_LEN - (COUNT_X + 20)
    off = 0; n = 0
    while off + VIEW_W <= fabric_len:
        frame = cv2.resize(strip[:, off:off + VIEW_W], (W, H), interpolation=cv2.INTER_CUBIC)
        ff.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); off += STEP; n += 1
    ff.close()
    gt = args.out.rsplit(".", 1)[0] + "_ground_truth.csv"
    with open(gt, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["defect_id", "class", "frame_at_centre", "metres_from_start", "x_in_strip_px", "y_px", "w_px", "h_px"])
        for i, (t, cx, cy, bw, bh, _c) in enumerate(sorted(defects, key=lambda d: d[1])):
            w.writerow([i + 1, t, max(int((cx - COUNT_X) / STEP), 0), round(cx / PX_PER_M, 3), round(cx), round(cy), round(bw), round(bh)])
    print(f"wrote {args.out}: {n} frames ({n / FPS:.1f} s), {fabric_len / PX_PER_M:.2f} m of fabric, {len(defects)} labelled defects -> {gt}")
    raise SystemExit
ff = Writer(args.out)
def card(lines):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); y = 250
    for txt, fnt, col in lines:
        d.text(((W - d.textlength(txt, font=fnt)) / 2, y), txt, font=fnt, fill=col); y += fnt.size + 24
    if args.credit:
        d.text((32, 690), args.credit, font=f_s, fill=(200, 150, 90))
    return im
emit = lambda im, n: [ff.write(im) for _ in range(n)]
emit(card([("Fabric Defect Inspector", font(44, True), INK), (f"{len(tiles)} dataset images  |  classes: " + ", ".join(CLASSES), f_b, QUIET),
           (f"boxes: {src}", f_b, ACC)]), FPS * 3)
counted, log, off, fi = [], [], 0, 0
while off + VIEW_W <= STRIP_LEN:
    for i, (t, cx, *_r) in enumerate(defects):
        if i not in counted and cx - off <= COUNT_X:
            counted.append(i); log.append(f"{(off + COUNT_X) / PX_PER_M:5.2f} m  #{i+1} {t}")
    last = render(off, counted, log, fi); ff.write(last)
    off += STEP; fi += 1
    if fi % 300 == 0: print(f'  {fi} frames...', flush=True)
emit(last, FPS * 2)
metres = sum(img_w) / PX_PER_M; rate = len(counted) / metres * 100
emit(card([("Inspection complete", font(40, True), INK),
           (f"{len(counted)} defects in {metres:.2f} m  ->  {rate:.0f} per 100 m  ->  Grade {grade(rate)}", font(24, True), INK),
           ("   ".join(f"{t}: {sum(1 for i in counted if defects[i][0] == t)}" for t in CLASSES), f_b, QUIET)]), FPS * 4)
ff.close()
print(f"wrote {args.out}: {fi} frames, {len(counted)} defects")
