"""Fabric Defect Inspector dashboard.

Run from this folder:
    streamlit run app.py
"""
import pathlib
import tempfile
import time

import cv2
import pandas as pd
import streamlit as st

import inspector as I

HERE = pathlib.Path(__file__).resolve().parent
st.set_page_config(page_title="Fabric Defect Inspector", layout="wide")


def first_existing(*candidates):
    for c in candidates:
        if pathlib.Path(c).exists():
            return str(c)
    return str(candidates[0])


def find_model(pattern, default_name):
    """Newest file matching the pattern in this folder or the one above (so 'best (1).pt' is found too)."""
    hits = [p for folder in (HERE, HERE.parent) for p in folder.glob(pattern)]
    exact = [p for p in hits if p.name == default_name]
    if exact:
        return str(exact[0])
    return str(max(hits, key=lambda p: p.stat().st_mtime)) if hits else str(HERE / default_name)


@st.cache_data(show_spinner=False)
def saved_threshold(path, mtime):
    """Anomaly threshold stored in the PatchCore file by the speed-up notebook, or 2.88 for the original file."""
    try:
        import torch
        ck = torch.load(path, map_location="cpu")
        return round(float(ck["pix_thr"]), 2) if "pix_thr" in ck else 2.88
    except Exception:
        return 2.88


@st.cache_resource(show_spinner="Loading YOLO and PatchCore (first time takes a minute)...")
def load_models(yolo_path, patchcore_path):
    return I.Models(yolo_path, patchcore_path)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.header("Models")
yolo_path = st.sidebar.text_input("YOLO weights", find_model("best*.pt", "best.pt")).strip().strip('"')
pc_path = st.sidebar.text_input("PatchCore file", find_model("patchcore*.pt", "patchcore_textile_fast.pt")).strip().strip('"')

st.sidebar.header("Source")
source = st.sidebar.radio("Inspect", ["Demo roll (test photos)", "Uploaded video"])
if source == "Demo roll (test photos)":
    photos = st.sidebar.text_input("Folder of test photos",
                                   first_existing(HERE.parent / "prepared" / "textile_yolo" / "test" / "images",
                                                  HERE / "test_images"))
    n_photos = st.sidebar.slider("Photos on the roll", 5, 140, 20)
    m_per_photo = st.sidebar.number_input("Fabric length per photo (m)", 0.01, 5.0, 0.15, 0.01)
    out_fps_choice = st.sidebar.radio("Video smoothness", [30, 60], format_func=lambda f: f"{f} frames per second", horizontal=True)
    model_px = st.sidebar.select_slider("Run the models every ... px of fabric", [16, 32, 64, 96], 32,
                                        help="The video moves a few px per frame so it looks smooth. The models run less often; "
                                             "frames in between reuse the last boxes, moved along with the fabric.")
    roll_step = 120 // out_fps_choice                     # 4 px per frame at 30 fps, 2 px at 60 fps: same fabric speed
    every = max(model_px // roll_step, 1)
    upload, view_m = None, None
else:
    upload = st.sidebar.file_uploader("Fabric video", type=["mp4", "avi", "mov", "mkv"])
    every = st.sidebar.slider("Run the models every Nth frame", 1, 10, 2,
                              help="Frames in between reuse the last boxes, moved along with the fabric.")
    view_m = st.sidebar.number_input("Fabric length visible in one frame (m)", 0.01, 10.0, 0.25, 0.01,
                                     help="Measure with a ruler in the first frame: how many metres of fabric fit across the frame.")

st.sidebar.header("Detection settings")
conf = st.sidebar.slider("YOLO confidence", 0.05, 0.90, 0.25, 0.05)
_pc = pathlib.Path(pc_path)
default_thr = saved_threshold(pc_path, _pc.stat().st_mtime) if _pc.exists() else 2.88
pix_thr = st.sidebar.number_input("Anomaly threshold", 0.5, 10.0, default_thr, 0.05, key=f"thr_{pc_path}",
                                  help="Set on validation photos: 2.88 for the original PatchCore file, 3.01 for the fast one. "
                                       "Higher = fewer unknown-defect alarms.")
pc_every = st.sidebar.slider("Run PatchCore every Nth model frame", 1, 10, 3,
                             help="PatchCore is the slow model. Running it less often is faster; YOLO still runs on every frame.")
st.sidebar.header("Grading (defects per 100 m)")
a_max = st.sidebar.number_input("Grade A up to", 0.0, 1000.0, 5.0, 1.0)
b_max = st.sidebar.number_input("Grade B up to", 0.0, 1000.0, 15.0, 1.0)

# ----------------------------------------------------------------------------- page
st.title("Fabric Defect Inspector")
st.caption("YOLO names known defects (Hole, Knot, Line, Stain). PatchCore flags anything unlike normal fabric as an unknown defect. "
           "Each defect is counted once as it moves past the camera.")

run = st.button("Run inspection", type="primary")
left, right = st.columns([3, 2])
frame_slot = left.empty()
progress = left.progress(0.0)
status = left.empty()
with right:
    c1, c2 = st.columns(2)
    m_metres, m_defects = c1.empty(), c2.empty()
    c3, c4 = st.columns(2)
    m_rate, m_grade = c3.empty(), c4.empty()
    chart_slot = st.empty()
log_slot = st.empty()
download_slot = st.container()


def show(s):
    m_metres.metric("Metres scanned", f"{s['metres']:.2f} m")
    m_defects.metric("Defects counted", s['defects'], f"{s['known']} known, {s['unknown']} unknown", delta_color="off")
    m_rate.metric("Defects per 100 m", f"{s['rate']:.0f}")
    m_grade.metric("Quality grade", s['grade'])
    if s['by_type']:
        chart_slot.bar_chart(pd.DataFrame({'defects': s['by_type']}))


if run:
    if not pathlib.Path(yolo_path).exists() or not pathlib.Path(pc_path).exists():
        st.error("Model file not found. Check the two paths in the sidebar (best.pt and patchcore_textile_fast.pt or patchcore_textile.pt).")
        st.stop()
    models = load_models(yolo_path, pc_path)

    tmp_video, out_fps = None, 30
    if source == "Demo roll (test photos)":
        if not pathlib.Path(photos).is_dir():
            st.error(f"Folder not found: {photos}")
            st.stop()
        frames, total, roll_len = I.roll_from_photos(photos, n_images=n_photos, step=roll_step)
        px_per_m, out_fps = 640 / m_per_photo, out_fps_choice
    else:
        if upload is None:
            st.error("Upload a video in the sidebar first.")
            st.stop()
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(upload.name).suffix)
        tmp_video.write(upload.read()); tmp_video.close()
        frames, total, out_fps = I.video_frames(tmp_video.name)
        px_per_m = None                                   # set from the first frame's width

    tmp = pathlib.Path(tempfile.gettempdir())
    raw_path, out_path = str(tmp / "fabric_inspection_raw.mp4"), str(tmp / "fabric_inspection.mp4")
    insp, writer, shown = None, None, 0
    t0 = time.time()
    for i, (frame, offset, cols, seam_x) in enumerate(frames):
        if insp is None:
            if px_per_m is None:
                px_per_m = frame.shape[1] / view_m
            insp = I.Inspector(models, px_per_m, conf=conf, pix_thr=pix_thr, patchcore_every=pc_every)
            writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (frame.shape[1], frame.shape[0]))
        model_frame = i % every == 0
        if model_frame:
            insp.process(frame, offset, cols, seam_x)     # run YOLO (and PatchCore when due)
        vis = insp.render(frame, offset, hud=insp.hud_text(a_max, b_max))   # every frame: boxes moved with the fabric
        writer.write(vis)
        if model_frame:
            frame_slot.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            show(insp.summary(a_max, b_max))
            rate = insp.n / max(time.time() - t0, 1e-6)
            status.caption(f"frame {i + 1}" + (f" of {total}" if total else "") +
                           f"  |  models: {rate:.1f} frames per second  |  device: {models.dev}")
        if total:
            progress.progress(min((i + 1) / total, 1.0))
    if insp is None:
        st.error("No frames could be read from this source.")
        st.stop()
    writer.release()
    progress.progress(1.0)
    status.caption("Preparing the video...")
    playable = I.to_h264(raw_path, out_path)              # browsers need H.264 to play it on the page
    if not playable:
        out_path = raw_path

    final = insp.summary(a_max, b_max, final=True)
    show(final)
    log = pd.DataFrame(insp.log())
    st.session_state["result"] = {"summary": final, "log": log, "video": open(out_path, "rb").read(), "playable": playable,
                                  "seconds": time.time() - t0, "frames": i + 1, "model_frames": insp.n}

if "result" in st.session_state:
    r = st.session_state["result"]
    if not run:
        show(r["summary"])
    s = r["summary"]
    status.caption(f"Done: {r['frames']} video frames, models run on {r['model_frames']} of them, in {r['seconds']:.0f} s")
    if r["playable"]:
        frame_slot.video(r["video"])                     # smooth playback of the finished inspection
    else:
        left.warning("The video was saved but cannot play on this page. Install the player helper with: "
                     "pip install imageio-ffmpeg  (then run again), or download the video below.")
    log_slot.dataframe(r["log"], hide_index=True)
    with download_slot:
        d1, d2 = st.columns(2)
        d1.download_button("Download annotated video", r["video"], "fabric_inspection.mp4", "video/mp4")
        d2.download_button("Download defect log (CSV)", r["log"].to_csv(index=False).encode(), "defect_log.csv", "text/csv")
else:
    frame_slot.info("Pick a source in the sidebar, then press Run inspection.")
