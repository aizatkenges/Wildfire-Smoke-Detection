# Wildfire Smoke Detection with YOLOv8n

Fine-tuning a pretrained YOLOv8n detector to locate smoke in wildfire-monitoring images. This project compares two input sizes, selects a model using validation data, and reports final results on a held-out test split.

> **Student:** Aizat Kenges  
> **Course:** CSCI1409 Deep Learning — Bonus Project  
> **Term:** Fall 2026

## Results at a glance

| Run | Input size | Validation precision | Validation recall | Validation mAP@0.50 | Validation mAP@0.50:0.95 | Training time |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (selected) | 640 px | 0.879 | 0.884 | 0.924 | 0.543 | 1.735 h |
| Controlled experiment | 512 px | 0.874 | 0.837 | 0.916 | 0.526 | 1.015 h |

The 640 px run scored higher on all four reported validation measures, so it was selected before evaluating the test split. It took about 43 minutes longer to train.

**Held-out test results for the selected 640 px checkpoint (74 images):** precision **0.861**, recall **0.946**, mAP@0.50 **0.947**, mAP@0.50:0.95 **0.554**.

## Dataset

- **Source:** [Wildfire Smoke on Roboflow Universe](https://universe.roboflow.com/brad-dwyer/wildfire-smoke/dataset/1), version 1.
- **Usage terms:** CC BY-NC-SA 4.0. Follow the dataset license when using or redistributing the data.
- **Class:** `smoke` (one class; class ID `0`).
- **Annotation format:** YOLO detection text files. Each object row contains `class_id x_center y_center width height`; the four box values are normalized to the image width or height.
- **Provided splits:**

| Split | Images | Labeled objects |
|---|---:|---:|
| Train | 516 | 516 |
| Validation | 147 | 147 |
| Test | 74 | 74 |

The notebook displays five training examples with their ground-truth boxes and inspects labels. The dataset already supplies the split; no new split seed was used. 

## Model and training

- **Framework:** Ultralytics 8.4.215; PyTorch 2.14.1+cpu; Python 3.12.7.
- **Hardware:** AMD Ryzen 5 5500U CPU; CUDA GPU unavailable.
- **Starting checkpoint:** `yolov8n.pt`, pretrained on COCO (80 classes; no smoke class). The model head was adapted to one class; Ultralytics reported transferring 349 of 355 checkpoint tensor entries.
- **Baseline settings:** 640 px, batch size 8, 30 epochs, CPU, workers 0, seed 42. `optimizer="auto"` selected AdamW with learning rate 0.002. Training took 1.735 hours.
- **Augmentation:** Ultralytics defaults were used, including Mosaic (`mosaic=1.0`); Mosaic is disabled for the final 10 epochs (`close_mosaic=10`).
- **Checkpoint:** `runs/detect/runs/detect/wildfire_baseline-2/weights/best.pt` (selected using validation results). `last.pt` is the checkpoint from the final training epoch.

## Controlled experiment

**Question:** Would reducing the input size from 640 to 512 make CPU training faster, and what performance might be lost on faint or small smoke regions?

Only `imgsz` changed. The experiment reused the dataset splits, starting checkpoint (`yolov8n.pt`), 30 epochs, batch size 8, CPU, workers 0, and seed 42. Its run folder is `runs/detect/wildfire_imgsz512`; training took 1.015 hours. Both saved `best.pt` checkpoints were evaluated on the same 147 validation images with the same evaluation settings. The 512 px run was faster, but the 640 px run performed better on the validation metrics, so 640 px was chosen.

## Evaluation settings and metric meanings

For validation and final test metrics, the notebook used `conf=0.001` and NMS `iou=0.7` for both models. The low confidence floor allows the evaluator to consider candidate boxes when constructing precision-recall curves; `iou=0.7` controls how strongly overlapping predicted boxes are removed as duplicates. The mAP IoU matching threshold is separate: **mAP@0.50** counts a predicted box as a match when its intersection-over-union with a ground-truth box is at least 0.50.

- **Precision:** among predicted smoke boxes, how many are correct.
- **Recall:** among labeled smoke boxes, how many the model finds.
- **mAP@0.50:** average precision across confidence cutoffs, using 0.50 box overlap to count a match.
- **mAP@0.50:0.95:** mAP averaged over stricter overlap thresholds from 0.50 through 0.95.

For the five-image prediction display and the all-test error scan, predictions used `conf=0.25` and NMS `iou=0.7`. The error scan used a separate IoU match threshold of 0.50. After removing three initial false flags from review, **11 possible mismatch images remain**. These are candidates for manual review, not a claim that smoke was entirely missed in every image: some issues are duplicate predictions or boxes that are too wide/narrow compared with the label.

## Reproduce in a notebook

1. Open this repository's Colab notebook (link to be added after the repository is published).
2. Install the recorded framework version and make the Roboflow dataset available in YOLOv8 format. Keep the extracted directory named `Wildfire Smoke.v1-raw.yolov8` beside the notebook, with `data.yaml` at its root.
3. Run the cells in order. The notebook checks the environment and labels, trains the 640 px baseline and 512 px experiment, compares both checkpoints on validation, evaluates the selected checkpoint on test, and displays predictions and mismatch examples.

Core training call for the baseline:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data="Wildfire Smoke.v1-raw.yolov8/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    device="cpu",
    workers=0,
    seed=42,
    project="runs/detect",
    name="wildfire_baseline",
)
```

To reproduce the controlled experiment, use the same call and change only `imgsz=512` and the run name to `wildfire_imgsz512`.

The notebook records the versions used. Install compatible packages in a fresh environment before running; do not update packages mid-project if exact reproduction is needed. The final trained checkpoint and run outputs must be made accessible separately because model weights are not embedded in this README.

## Error review and limitations

The final checkpoint found most labeled test smoke boxes, but the test set is small (74 images) and smoke can be faint, diffuse, and hard to box consistently. Manual review left 11 candidate mismatches, including duplicate detections over a broad plume and predicted boxes with different extents from the annotation. A practical next step is to review and standardize plume-box boundaries, check those 11 cases, and audit near-duplicate or same-source images across splits before additional training.

Other limits: CPU-only training was slow; results apply to this one-class dataset and should not be treated as proof of field readiness. The test set was used only after model selection; do not tune thresholds or training settings against its scores.
