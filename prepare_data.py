"""Prepare the fabric datasets for training.

Run from the Fabric folder:
    pip install opencv-python numpy pyyaml
    python prepare_data.py --textile Textile.yolov11 --coco "fabric defect.coco" --out prepared

What it makes inside --out:
    textile_yolo/   train | valid | test  (images + labels) + data.yaml   -> YOLO training (Hole, Knot, Line, Stain)
    fabric_yolo/    train | valid | test  (images + labels) + data.yaml   -> COCO set converted to YOLO boxes,
                    class names in English (broken_yarn, stain, defect, pulled_or_thick_thread)
    patchcore/<dataset>/train/good, test/good, test/defect                 -> 256 px crops for PatchCore
    split_summary.csv                                                      -> counts per dataset / split / class

Why it is needed:
  * Roboflow added augmented copies of many photos. All copies of one photo are kept in the SAME split,
    so the model is never tested on a photo it has already seen (no leakage).
  * valid and test get one copy of each photo; train keeps every copy.
  * Photos that appear in both datasets are kept only in Textile.
  * Polygon labels become boxes; the extra COCO category 0 is dropped.
"""
import argparse, collections, csv, glob, json, os, random, shutil
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--textile", required=True, help="Textile.yolov11 folder (has train/images, train/labels)")
ap.add_argument("--coco", default=None, help='"fabric defect.coco" folder (has train/_annotations.coco.json)')
ap.add_argument("--out", default="prepared")
ap.add_argument("--val", type=float, default=0.20)
ap.add_argument("--test", type=float, default=0.10)
ap.add_argument("--crop", type=int, default=256, help="PatchCore crop size in px")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
random.seed(args.seed); rng = np.random.default_rng(args.seed)

base = lambda f: os.path.basename(f).split(".rf.")[0]          # original photo name, shared by its augmented copies

# ---------------- load: {base: [(image path, [(cls, cx, cy, w, h), ...]), ...]} ----------------
def load_textile(root):
    groups = collections.defaultdict(list)
    for img in sorted(glob.glob(os.path.join(root, "train", "images", "*"))):
        lp = os.path.join(root, "train", "labels", os.path.splitext(os.path.basename(img))[0] + ".txt")
        boxes = []
        if os.path.exists(lp):
            for v in (l.split() for l in open(lp) if l.strip()):
                if len(v) > 5:                                    # polygon -> box
                    xs, ys = list(map(float, v[1::2])), list(map(float, v[2::2]))
                    boxes.append((int(v[0]), (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)))
                elif len(v) == 5:
                    boxes.append((int(v[0]), *map(float, v[1:])))
        groups[base(img)].append((img, boxes))
    return groups, ["Hole", "Knot", "Line", "Stain"]

# French class names in the COCO set -> English (label numbers and order are unchanged)
ENGLISH = {"Cassure": "broken_yarn", "Tache": "stain", "defaut": "defect", "Defaut": "defect",
           "fil tire ou gros": "pulled_or_thick_thread"}

def load_coco(root):
    d = json.load(open(os.path.join(root, "train", "_annotations.coco.json")))
    cats = [c for c in d["categories"] if c["id"] != 0]           # drop Roboflow's extra category 0
    idx = {c["id"]: i for i, c in enumerate(cats)}
    anns = collections.defaultdict(list)
    for a in d["annotations"]: anns[a["image_id"]].append(a)
    groups = collections.defaultdict(list)
    for im in sorted(d["images"], key=lambda i: i["file_name"]):
        W, H = im["width"], im["height"]
        boxes = [(idx[a["category_id"]], (a["bbox"][0] + a["bbox"][2] / 2) / W, (a["bbox"][1] + a["bbox"][3] / 2) / H,
                  a["bbox"][2] / W, a["bbox"][3] / H) for a in anns[im["id"]] if a["category_id"] in idx]
        groups[base(im["file_name"])].append((os.path.join(root, "train", im["file_name"]), boxes))
    return groups, [ENGLISH.get(c["name"], c["name"].strip().replace(" ", "_")) for c in cats]

datasets = {}
tex, tex_names = load_textile(args.textile); datasets["textile"] = (tex, tex_names)
if args.coco:
    coco, coco_names = load_coco(args.coco)
    shared = [b for b in coco if b in tex]
    for b in shared: del coco[b]
    print(f"photos in both datasets, kept only in Textile: {len(shared)}")
    datasets["fabric"] = (coco, coco_names)

# ---------------- split by original photo ----------------
def split_groups(groups):
    keys = sorted(groups); random.shuffle(keys)
    n = len(keys); n_test = round(n * args.test); n_val = round(n * args.val)
    return {"test": keys[:n_test], "valid": keys[n_test:n_test + n_val], "train": keys[n_test + n_val:]}

def clean_boxes(boxes):
    out = []
    for c, cx, cy, w, h in boxes:
        x0, y0, x1, y1 = max(cx - w / 2, 0), max(cy - h / 2, 0), min(cx + w / 2, 1), min(cy + h / 2, 1)
        if x1 - x0 > 0.002 and y1 - y0 > 0.002:
            out.append((c, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return out

summary = []
shutil.rmtree(args.out, ignore_errors=True)
for name, (groups, names) in datasets.items():
    root = os.path.join(args.out, f"{name}_yolo"); splits = split_groups(groups)
    for sp, keys in splits.items():
        os.makedirs(os.path.join(root, sp, "images")); os.makedirs(os.path.join(root, sp, "labels"))
        n_img = 0; cls = collections.Counter()
        for k in keys:
            copies = groups[k] if sp == "train" else groups[k][:1]     # valid/test: one copy per photo
            for img, boxes in copies:
                boxes = clean_boxes(boxes)
                stem = os.path.splitext(os.path.basename(img))[0]
                shutil.copy(img, os.path.join(root, sp, "images", os.path.basename(img)))
                with open(os.path.join(root, sp, "labels", stem + ".txt"), "w") as fh:
                    fh.writelines(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes)
                n_img += 1; cls.update(names[b[0]] for b in boxes)
        summary.append([name, sp, len(keys), n_img, sum(cls.values())] + [cls[n] for n in names] + [""] * (4 - len(names)))
        print(f"{name:8s} {sp:5s}: {len(keys):4d} photos, {n_img:4d} images, {sum(cls.values()):4d} boxes  {dict(cls)}")
    with open(os.path.join(root, "data.yaml"), "w") as fh:
        fh.write(f"path: {os.path.abspath(root)}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnc: {len(names)}\nnames: {json.dumps(names)}\n")

    # ---------------- PatchCore crops ----------------
    S = args.crop; pc = os.path.join(args.out, "patchcore", name)
    for sub in ("train/good", "test/good", "test/defect"): os.makedirs(os.path.join(pc, sub))
    counts = collections.Counter()
    def px_boxes(boxes, W, H, pad=12):
        return [(int((cx - w / 2) * W) - pad, int((cy - h / 2) * H) - pad, int((cx + w / 2) * W) + pad, int((cy + h / 2) * H) + pad) for _, cx, cy, w, h in boxes]
    def hits(x, y, bxs):
        return any(x < b[2] and x + S > b[0] and y < b[3] and y + S > b[1] for b in bxs)
    for sp, per_img, dest in (("train", 2, "train/good"), ("test", 2, "test/good")):
        for k in splits[sp]:
            img, boxes = groups[k][0]                                   # one copy per photo, so crops are not near-duplicates
            im = cv2.imread(img)
            if im is None or min(im.shape[:2]) < S: continue
            H, W = im.shape[:2]; bxs = px_boxes(boxes, W, H); got = 0
            for _ in range(40):
                x, y = int(rng.integers(0, W - S + 1)), int(rng.integers(0, H - S + 1))
                if hits(x, y, bxs): continue
                cv2.imwrite(os.path.join(pc, dest, f"{k[:60]}_{got}.png"), im[y:y + S, x:x + S]); got += 1; counts[dest] += 1
                if got >= per_img: break
    for k in splits["test"]:                                            # defect crops centred on each labelled defect
        img, boxes = groups[k][0]; im = cv2.imread(img)
        if im is None or min(im.shape[:2]) < S: continue
        H, W = im.shape[:2]
        for j, (c, cx, cy, w, h) in enumerate(boxes[:3]):
            x = int(np.clip(cx * W - S / 2, 0, W - S)); y = int(np.clip(cy * H - S / 2, 0, H - S))
            cv2.imwrite(os.path.join(pc, "test/defect", f"{k[:60]}_{names[c]}_{j}.png".replace(" ", "_")), im[y:y + S, x:x + S]); counts["test/defect"] += 1
    print(f"{name:8s} patchcore crops: {dict(counts)}")
    summary.append([name, "patchcore", "", "", ""] + [f"{k}={v}" for k, v in counts.items()])

with open(os.path.join(args.out, "split_summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["dataset", "split", "photos", "images", "boxes", "class_1", "class_2", "class_3", "class_4"]); w.writerows(summary)
print(f"\ndone -> {os.path.abspath(args.out)}")
