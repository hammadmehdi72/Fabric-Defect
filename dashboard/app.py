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
import live as L
import hud

HERE = pathlib.Path(__file__).resolve().parent
LIVE_SESSION_KEY = "live_camera_session"
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
live_active = LIVE_SESSION_KEY in st.session_state
st.sidebar.header("Models")
yolo_path = st.sidebar.text_input("YOLO weights", find_model("best*.pt", "best.pt"),
                                  disabled=live_active).strip().strip('"')
pc_path = st.sidebar.text_input("PatchCore file", find_model("patchcore*.pt", "patchcore_textile_fast.pt"),
                                disabled=live_active).strip().strip('"')

st.sidebar.header("Source")
LIVE = "Live camera"
source = st.sidebar.radio("Inspect", ["Demo roll (test photos)", "Uploaded video", LIVE],
                          disabled=live_active)
cam = None
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
elif source == LIVE:
    cam = st.sidebar.text_input("Camera index or stream address", "0", disabled=live_active,
                                help="0 is the built-in webcam, 1 a second camera. A phone can be used with a webcam app: "
                                     "type its stream address here, for example http://192.168.1.5:4747/video")
    view_m = st.sidebar.number_input("Fabric length visible in one frame (m)", 0.01, 10.0, 0.25, 0.01,
                                     disabled=live_active,
                                     help="How many metres of fabric fit across the camera view.")
    upload, every = None, 1
    direction = st.sidebar.radio("Fabric moves", ["left / right", "up / down"], horizontal=True, disabled=live_active,
                                 help="Sets where the counting zone goes: a band across the path of the fabric.")
    zone_pct = st.sidebar.slider("Counting zone width (% of the view)", 6, 40, 16, 2, disabled=live_active)
    camera_name = st.sidebar.text_input("Camera name", "CAM 01")
    dashboard_view = st.sidebar.checkbox("Control-room dashboard view", True,
                                         help="Shows the camera with counts, a close-up of the latest defect, recent defects, "
                                              "a timeline and a defect map in one screen.")
    roi_x = st.sidebar.slider("Inspection region, left to right (%)", 0, 100, (5, 95), 1, disabled=live_active,
                              help="Only this part of the view is inspected. Everything outside it (table, hands, room) is ignored.")
    roi_y = st.sidebar.slider("Inspection region, top to bottom (%)", 0, 100, (5, 95), 1, disabled=live_active)
    show_heat = st.sidebar.checkbox("Show PatchCore heatmap", False)
    max_share = st.sidebar.slider("Stop counting when more than ...% of the view looks unusual", 5, 60, 20, 5,
                                  disabled=live_active,
                                  help="A safety check for the live camera. Real fabric is mostly normal, so a view where a large "
                                       "share looks unusual is probably not fabric (a face, a room) or a fabric the models do not know.") / 100
    st.sidebar.caption("Fix the camera above the fabric, pointing straight down, with bright even light. Fill the whole view "
                       "with cloth. Each defect gets an ID and is counted once when it passes through the counting zone. "
                       "For a new fabric or camera, press 'Learn this fabric' with clean cloth in view first.")
    st.sidebar.caption("The camera must be connected to this computer, which is running Streamlit. Allow camera access in Windows privacy settings.")
else:
    upload = st.sidebar.file_uploader("Fabric video", type=["mp4", "avi", "mov", "mkv"])
    every = st.sidebar.slider("Run the models every Nth frame", 1, 10, 2,
                              help="Frames in between reuse the last boxes, moved along with the fabric.")
    view_m = st.sidebar.number_input("Fabric length visible in one frame (m)", 0.01, 10.0, 0.25, 0.01,
                                     help="Measure with a ruler in the first frame: how many metres of fabric fit across the frame.")

if source != LIVE:
    dashboard_view = st.sidebar.checkbox("Control-room dashboard view", True,
                                         help="Shows the video with counts, a close-up of the latest defect, recent defects, "
                                              "a timeline and a defect map in one screen. The saved video shows the full screen too.")

st.sidebar.header("Detection settings")
conf = st.sidebar.slider("YOLO confidence", 0.05, 0.90, 0.25, 0.05, disabled=live_active)
_pc = pathlib.Path(pc_path)
default_thr = saved_threshold(pc_path, _pc.stat().st_mtime) if _pc.exists() else 2.88
pix_thr = st.sidebar.number_input("Anomaly threshold", 0.5, 10.0, default_thr, 0.05, key=f"thr_{pc_path}",
                                  disabled=live_active,
                                  help="Set on validation photos: 2.88 for the original PatchCore file, 3.01 for the fast one. "
                                       "Higher = fewer unknown-defect alarms.")
pc_every = st.sidebar.slider("Run PatchCore every Nth model frame", 1, 10, 3, disabled=live_active,
                             help="PatchCore is the slow model. Running it less often is faster; YOLO still runs on every frame.")
st.sidebar.header("Grading (defects per 100 m)")
a_max = st.sidebar.number_input("Grade A up to", 0.0, 1000.0, 5.0, 1.0, disabled=live_active)
b_max = st.sidebar.number_input("Grade B up to", 0.0, 1000.0, 15.0, 1.0, disabled=live_active)

# ----------------------------------------------------------------------------- page
st.title("Fabric Defect Inspector")
st.caption("YOLO names known defects (Hole, Knot, Line, Stain). PatchCore flags anything unlike normal fabric as an unknown defect. "
           "Each defect is counted once as it moves past the camera.")

if source == LIVE:
    b1, b2, b3, b4, _ = st.columns([1, 1, 1.3, 1.5, 1])
    start_camera = b1.button("Start camera", type="primary", disabled=live_active)
    stop_camera = b2.button("Stop camera", disabled=not live_active)
    if b3.button("Learn this fabric", disabled=not live_active,
                 help="Show clean cloth with no defects. About 40 frames are captured, PatchCore learns this fabric and "
                      "sets its own alarm level. Saved as patchcore_my_fabric.pt in the dashboard folder."):
        st.session_state[LIVE_SESSION_KEY]["teach"] = []
    if b4.button("Save frame for labelling", disabled=not live_active,
                 help="Saves the current camera frame to the captured_frames folder. Label these in Roboflow to retrain "
                      "YOLO on your own camera."):
        st.session_state[LIVE_SESSION_KEY]["save_frame"] = True
    run = False
else:
    start_camera, stop_camera = False, False
    run = st.button("Run inspection", type="primary")
wide_slot = st.empty()                                   # the control-room view uses the full width
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


def finish_live():
    live = st.session_state.get(LIVE_SESSION_KEY)
    if live is None:
        return
    if not live["camera_released"]:
        live["cap"].release()
        live["camera_released"] = True
    engine = live["engine"]
    if engine is not None and not engine.close(timeout=10):
        live["stopping"] = True
        st.session_state["live_error"] = "The detection worker is still finishing a frame. Please wait before starting again."
        return
    st.session_state.pop(LIVE_SESSION_KEY, None)
    if engine is not None:
        st.session_state["result"] = {
            "summary": engine.summary(live["px_per_m"], a_max, b_max),
            "log": pd.DataFrame(engine.log()),
            "video": None,
            "playable": False,
            "seconds": time.time() - live["started_at"],
            "frames": live["frames"],
            "model_frames": engine.n,
        }
    else:
        st.session_state["live_notice"] = "Camera stopped before a video frame was received."


if stop_camera:
    finish_live()
    st.rerun()

if start_camera and source == LIVE:
    if not pathlib.Path(yolo_path).exists() or not pathlib.Path(pc_path).exists():
        st.error("Model file not found. Check the two paths in the sidebar.")
        st.stop()
    st.session_state.pop("result", None)
    st.session_state.pop("live_error", None)
    cap = I.open_camera(cam)
    if not cap.isOpened():
        cap.release()
        st.error(f"Could not open camera '{cam}'. Check Windows camera privacy permissions, close apps using the camera, "
                 "or try camera index 1 instead of 0.")
        st.stop()
    try:
        models = load_models(yolo_path, pc_path)
    except Exception:
        cap.release()
        raise
    st.session_state[LIVE_SESSION_KEY] = {
        "cap": cap,
        "camera_released": False,
        "models": models,
        "engine": None,
        "px_per_m": None,
        "teach": None,
        "save_frame": False,
        "estimator": I.MotionEstimator(),
        "offset": 0.0,
        "frames": 0,
        "read_failures": 0,
        "stopping": False,
        "started_at": time.time(),
    }
    st.rerun()
elif source != LIVE:
    finish_live()

if source != LIVE and run:
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
    insp, writer, view = None, None, None
    t0 = time.time()
    for i, (frame, offset, cols, seam_x) in enumerate(frames):
        if insp is None:
            if px_per_m is None:
                px_per_m = frame.shape[1] / view_m
            insp = I.Inspector(models, px_per_m, conf=conf, pix_thr=pix_thr, patchcore_every=pc_every)
            if dashboard_view:
                view = hud.InspectorView(insp, out_fps)
        model_frame = i % every == 0
        if model_frame:
            insp.process(frame, offset, cols, seam_x)     # run YOLO (and PatchCore when due)
        if dashboard_view:
            if model_frame:
                view.update(frame, offset, i)
            vis = hud.compose(insp.render(frame, offset), view.snapshot(i, offset, frame.shape[1]),
                              insp.summary(a_max, b_max),
                              {"camera": "CAM 01", "badge": "DEMO" if source == "Demo roll (test photos)" else "REPLAY",
                               "source": "demo roll" if source == "Demo roll (test photos)" else "uploaded video",
                               "features": "roll-position tracking  |  each defect counted once",
                               "fps": (i + 1) / max(time.time() - t0, 1e-6),
                               "model_fps": insp.n / max(time.time() - t0, 1e-6), "device": models.dev})
        else:
            vis = insp.render(frame, offset, hud=insp.hud_text(a_max, b_max))   # every frame: boxes moved with the fabric
        if writer is None:
            writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (vis.shape[1], vis.shape[0]))
        writer.write(vis)
        if model_frame:
            if dashboard_view:
                wide_slot.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), output_format="JPEG")
            else:
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
                                  "seconds": time.time() - t0, "frames": i + 1, "model_frames": insp.n,
                                  "wide": dashboard_view}

live_error = st.session_state.pop("live_error", None)
if live_error:
    st.error(live_error)
live_notice = st.session_state.pop("live_notice", None)
if live_notice:
    st.info(live_notice)

live_running = source == LIVE and LIVE_SESSION_KEY in st.session_state
if live_running:
    frame_slot.info("Connecting to the camera...")
    status.caption("Live camera is starting.")
    if not dashboard_view:
        m_metres.caption("Waiting for camera")
        m_defects.caption("Waiting for camera")
        m_rate.caption("Waiting for camera")
        m_grade.caption("Waiting for camera")

@st.fragment(run_every=0.12 if live_running else None)
def refresh_live_camera():
    live = st.session_state.get(LIVE_SESSION_KEY)
    if live is None:
        return
    if live["stopping"]:
        if live["engine"] is None or not live["engine"].thread.is_alive():
            finish_live()
            st.rerun()
        status.caption("Waiting for the detector to finish its current frame...")
        return
    ok, frame = live["cap"].read()
    if not ok:
        live["read_failures"] += 1
        if live["read_failures"] >= 20:
            finish_live()
            st.session_state["live_error"] = (
                "The camera opened but did not provide video frames. Check Windows camera permissions, "
                "make sure another app is not using it, and try camera index 1 if you have another camera."
            )
            st.rerun()
        frame_slot.warning(f"Waiting for a camera frame ({live['read_failures']}/20)...")
        return

    live["read_failures"] = 0
    frame = I.fit_frame(frame)
    move = live["estimator"].update(frame)
    if live["save_frame"]:
        live["save_frame"] = False
        folder = HERE / "captured_frames"; folder.mkdir(exist_ok=True)
        name = folder / time.strftime("frame_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(str(name), frame)
        st.toast(f"Saved {name.name}")
    if live["engine"] is None:
        live["px_per_m"] = frame.shape[1] / view_m
        live["engine"] = L.LiveEngine(
            live["models"], frame.shape[1], frame.shape[0], conf=conf, pix_thr=pix_thr, patchcore_every=pc_every,
            max_share=max_share, direction="horizontal" if direction == "left / right" else "vertical",
            zone_share=zone_pct / 100,
            roi_share=(roi_x[0] / 100, roi_y[0] / 100, roi_x[1] / 100, roi_y[1] / 100))
        live["engine"].px_per_m = live["px_per_m"]
    engine = live["engine"]
    if engine.error is not None:
        error = engine.error
        finish_live()
        st.session_state["live_error"] = f"Defect detection stopped because the model returned an error: {error}"
        st.rerun()
    teaching = None
    if live["teach"] is not None:                       # collecting clean frames to learn this fabric
        live["teach"].append(frame.copy())
        teaching = f"{len(live['teach'])} of 40 frames"
        if len(live["teach"]) >= 40:
            engine.paused = True
            time.sleep(0.5)                               # let the detector finish its current frame
            status.caption("Learning this fabric...")
            try:
                thr = live["models"].learn_fabric(live["teach"][::1])
                live["models"].save_bank(HERE / "patchcore_my_fabric.pt", thr, note="learned from the live camera")
                engine.pix_thr = thr
                st.toast(f"Learned this fabric. New alarm level {thr:.2f}, saved as patchcore_my_fabric.pt")
            except Exception as e:
                st.session_state["live_error"] = f"Could not learn this fabric: {e}"
            live["teach"] = None
            engine.paused = False
    else:
        if abs(move) >= 1.0:
            engine.travel_px += abs(move)
        engine.submit(frame.copy())
    live["frames"] += 1
    fps = live["frames"] / max(time.time() - live["started_at"], 1e-6)
    if dashboard_view:
        vis = engine.render(frame, camera_name, show_heat, teaching, panel=False)
        canvas = hud.compose(vis, engine.snapshot(), engine.summary(live["px_per_m"], a_max, b_max),
                             {"camera": camera_name, "source": "fabric line", "fps": fps, "model_fps": engine.rate(),
                              "device": live["models"].dev})
        frame_slot.empty()
        wide_slot.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), output_format="JPEG")
        return
    vis = engine.render(frame, camera_name, show_heat, teaching)
    if vis.shape[1] > 960:
        vis = cv2.resize(vis, (960, int(vis.shape[0] * 960 / vis.shape[1])), interpolation=cv2.INTER_AREA)
    frame_slot.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), output_format="JPEG")
    if live["frames"] % 5 == 0:
        show(engine.summary(live["px_per_m"], a_max, b_max))
        status.caption(
            f"Live | picture: {live['frames'] / max(time.time() - live['started_at'], 1e-6):.0f} frames per second"
            f" | models: {engine.rate():.1f} per second | device: {live['models'].dev}"
        )

if live_running:
    refresh_live_camera()

if "result" in st.session_state:
    r = st.session_state["result"]
    if not run and not r.get("wide"):
        show(r["summary"])
    s = r["summary"]
    status.caption(f"Done: {r['frames']} video frames, models run on {r['model_frames']} of them, in {r['seconds']:.0f} s")
    if r["video"] is None:
        frame_slot.info("Camera stopped. The defects counted in this session are listed below.")
    elif r["playable"]:
        (wide_slot if r.get("wide") else frame_slot).video(r["video"])                     # smooth playback of the finished inspection
    else:
        left.warning("The video was saved but cannot play on this page. Install the player helper with: "
                     "pip install imageio-ffmpeg  (then run again), or download the video below.")
    log_slot.dataframe(r["log"], hide_index=True)
    with download_slot:
        d1, d2 = st.columns(2)
        if r["video"] is not None:
            d1.download_button("Download annotated video", r["video"], "fabric_inspection.mp4", "video/mp4")
        d2.download_button("Download defect log (CSV)", r["log"].to_csv(index=False).encode(), "defect_log.csv", "text/csv")
elif not live_running:
    if source == LIVE:
        frame_slot.info("Check the camera settings, then press Start camera.")
    else:
        frame_slot.info("Pick a source in the sidebar, then press Run inspection.")
