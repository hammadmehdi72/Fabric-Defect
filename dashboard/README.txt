Fabric Defect Inspector dashboard
=================================

Files in this folder
  app.py            the Streamlit dashboard
  inspector.py      the detection engine (YOLO + PatchCore fusion, counting, grading)
  requirements.txt  Python packages

You also need, in this folder or in the Fabric folder above it:
  best.pt                 trained YOLO model        (Google Drive: fabric_inspector_yolo)
  patchcore_textile_fast.pt   fast PatchCore memory bank, about 9 times faster   (Google Drive: fabric_inspector_speed)
      or patchcore_textile.pt  the original, slower one                      (Google Drive: fabric_inspector_patchcore)
  If both are there, the dashboard uses the fast one and sets the anomaly threshold to match (3.01).

One-time setup (terminal)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install -r requirements.txt     (run this again after an update: it now includes imageio-ffmpeg)

Run
  cd C:\Users\mehdi\Desktop\Fabric\dashboard
  streamlit run app.py

Then pick a source in the sidebar.
  Demo roll       joins test photos from prepared\textile_yolo\test\images into a moving roll
  Uploaded video  any fabric video; fabric should move sideways across the frame
  Live camera     use the camera connected to the computer running Streamlit, then press "Start camera"

For Live camera, camera index 0 is the default webcam; use 1 for a second camera. Allow desktop apps to use the
camera in Windows Privacy & security > Camera. Press "Stop camera" to finish and see the count and CSV log.
Live camera runs on the computer hosting Streamlit; it cannot directly open a webcam connected to a different browser device.
