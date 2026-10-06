Fabric Defect Inspector dashboard
=================================

Files in this folder
  app.py            the Streamlit dashboard
  inspector.py      the detection engine (YOLO + PatchCore fusion, counting, grading)
  requirements.txt  Python packages

You also need, in this folder or in the Fabric folder above it:
  best.pt                 trained YOLO model        (Google Drive: fabric_inspector_yolo)
  patchcore_textile.pt    PatchCore memory bank     (Google Drive: fabric_inspector_patchcore)

One-time setup (terminal)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install -r requirements.txt

Run
  cd C:\Users\mehdi\Desktop\Fabric\dashboard
  streamlit run app.py

Then pick a source in the sidebar and press "Run inspection".
  Demo roll       joins test photos from prepared\textile_yolo\test\images into a moving roll
  Uploaded video  any fabric video; fabric should move sideways across the frame
