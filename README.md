# Automated Textile and Fabric Defect Inspector

NIAI AI/ML Batch 5 capstone by Hammad Mehdi and Eisha Maqbool.

The system finds, names and counts fabric defects with two models:

- **YOLO11s** detects the four known defects (Hole, Knot, Line, Stain).
- **PatchCore** learns what clean fabric looks like and flags anything unusual, so defects YOLO was never trained on are still caught as "unknown".

A fusion step combines both, counts each defect once as the roll moves, and grades the roll (A, B or C) by defects per 100 m. A Streamlit dashboard shows it running.

The full write-up is in `Fabric_Defect_Inspector_Report.pdf`.

## Results

| What | Result |
|---|---|
| YOLO, test mAP50 | 0.789 |
| PatchCore, AUROC | 0.927 |
| Fusion, defects found on 140 unseen test photos | 93.3% |
| Fusion, false alarms per photo | 0.12 |
| Simulated 6 m roll, defects counted | 53 of 67 |
| Speed on a Colab T4 | about 1.6 frames per second |

Known gaps: the count is about 21% under on the simulated roll, the Line class is weak, the speed is below the 5 fps target, and nothing has been tested on real fabric video yet.

## What is in this repo

| Path | What it is |
|---|---|
| `prepare_data.py` | Splits the datasets by original photo (no leakage) and makes the PatchCore crops |
| `train_yolo_textile.ipynb`, `train_yolo_local.py` | YOLO training (Colab or local GPU) |
| `train_patchcore_textile.ipynb` | PatchCore training (Colab) |
| `fusion_inspector.ipynb`, `fusion_inspector_v2.ipynb` | Fusion, counting and grading |
| `make_video_from_dataset.py` | Builds simulated roll videos from dataset photos |
| `dashboard/` | Streamlit app and the two trained models |
| `Textile.yolov11/`, `fabric defect.coco/` | Datasets as exported from Roboflow |
| `prepared/` | Output of `prepare_data.py` |
| `videos/` | 51 simulated roll videos with ground-truth CSVs |
| `fabric inspector fusion/`, `fabric inspector pathcore/` | Result files from the Colab runs |

## Run the dashboard

```
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

`dashboard/patchcore_textile.pt` is stored with Git LFS, so run `git lfs install` once before cloning.

## Data

Both datasets come from Roboflow Universe and keep their original licences. The roll videos are simulated by joining dataset photos, they are not real camera footage.
