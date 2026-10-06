"""Train YOLO on the Textile dataset on your own PC (RTX 3050, 4 GB).

One-time setup (in a terminal):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install ultralytics pandas

Run from the Fabric folder:
    python train_yolo_local.py

Results go to  runs\\textile_yolo11s\\  (best weights: runs\\textile_yolo11s\\weights\\best.pt)
and the test-set numbers for your report to  runs\\test_metrics.csv
"""
import argparse, pathlib, shutil
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="prepared/textile_yolo/data.yaml")
ap.add_argument("--model", default="yolo11s.pt")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=8, help="8 fits a 4 GB GPU at 640 px; use 4 if you get 'CUDA out of memory'")
ap.add_argument("--name", default="textile_yolo11s")
args = ap.parse_args()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot see the GPU. Reinstall it with the CUDA build:\n"
                         "  pip uninstall -y torch torchvision\n"
                         "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    print("GPU:", torch.cuda.get_device_name(0))

    from ultralytics import YOLO
    data = str(pathlib.Path(args.data).resolve())
    runs = pathlib.Path("runs").resolve()

    model = YOLO(args.model)
    model.train(data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=15,
                workers=2, device=0, amp=True, seed=0,
                project=str(runs), name=args.name, exist_ok=True)

    best_path = runs / args.name / "weights" / "best.pt"
    best = YOLO(str(best_path))
    m = best.val(data=data, split="test", imgsz=args.imgsz, batch=args.batch, workers=2, device=0,
                 project=str(runs), name=args.name + "_test", exist_ok=True)

    import pandas as pd
    rows = [{"class": "all", "precision": m.box.mp, "recall": m.box.mr, "mAP50": m.box.map50, "mAP50-95": m.box.map}]
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        rows.append({"class": best.names[int(c)], "precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap})
    table = pd.DataFrame(rows).round(3)
    table.to_csv(runs / "test_metrics.csv", index=False)
    print("\nTEST-SET RESULTS (photos the model never saw)\n", table.to_string(index=False))

    shutil.copy(best_path, "best.pt")
    print(f"\nbest weights copied to {pathlib.Path('best.pt').resolve()}")
    print(f"plots: {runs / args.name}  (results.png)  and  {runs / (args.name + '_test')}  (confusion matrix)")


if __name__ == "__main__":      # required on Windows, or the data-loading workers crash
    main()
