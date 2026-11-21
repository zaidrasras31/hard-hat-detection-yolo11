# Hard Hat Detection with YOLO11s

**Author: Zaid Rasras**

An object detection project that trains YOLO11s to locate helmets and heads in construction-site images. The project includes annotation conversion, dataset preparation, training, evaluation, prediction visualization, and ONNX export.

## Dataset

[Hard Hat Detection — Andrew MVD on Kaggle](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)

The dataset contains 5,000 images with Pascal VOC XML annotations. Its original classes are `helmet`, `head`, and `person`.

The notebook first explores a three-class detector, then trains a final two-class model:

| Class ID | Class |
|---|---|
| 0 | helmet |
| 1 | head |

Each XML bounding box is converted to the normalized YOLO format:

```text
class_id x_center y_center width height
```

## Workflow

1. Parse XML annotations and inspect the object table.
2. Convert bounding boxes to YOLO labels.
3. Train an initial three-class baseline.
4. Prepare the final helmet/head dataset with separate training, validation, and test splits.
5. Train a fresh pretrained YOLO11s model on the two-class task.
6. Select a confidence threshold using validation F1 scores and evaluate on the test split.
7. Export to ONNX, compare sample predictions, and save environment details.

## Final Dataset Split

Images with identical 8×8 average hashes are grouped together before splitting. This reduces some duplicate leakage, but does not guarantee that all similar images or related scenes remain in the same split.

| Split | Images |
|---|---:|
| Training | 3,509 |
| Validation | 746 |
| Test | 745 |

The final dataset is generated under `/kaggle/working/yolo_dataset_v2`, with `images/train`, `images/val`, `images/test`, matching `labels` folders, and `data.yaml`.

## Training Configuration

| Setting | Value |
|---|---|
| Pretrained model | `yolo11s.pt` |
| Epochs | 30 |
| Image size | 640 |
| Batch size | 16 |
| Early-stopping patience | 10 |
| GPU in saved run | NVIDIA Tesla T4 |
| Ultralytics in saved run | 8.4.216 |

The final model is trained using:

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="/kaggle/working/yolo_dataset_v2/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,
    project="/kaggle/working/runs",
    name="hardhat_yolo11s_2class",
    patience=10,
    workers=2
)
```

## Final Test Results

The notebook selects a confidence threshold of approximately **0.452** using the validation mean F1 curve, then evaluates on **745 test images containing 3,761 annotated objects**.

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| All classes | 0.939 | 0.891 | 0.8806 | 0.6017 |
| helmet | 0.9608 | 0.8904 | 0.8912 | 0.608* |
| head | 0.9168 | 0.8917 | 0.8699 | 0.595* |

*Per-class mAP50–95 values are rounded as displayed in the saved evaluation log.*

**Evaluation detail:** the test call passes the selected confidence threshold to the validator. AP is therefore computed after that confidence filtering; it should not be treated as directly comparable to AP measured with the usual low confidence cutoff used to construct a fuller precision–recall curve. These are detection metrics, not classification accuracy.

The initial three-class experiment has much weaker person detection. The final results above belong to the separately trained two-class model and its different split.

## Visual Results and Export

The notebook includes training curves, a confusion matrix, sample predictions, and a helmet/head detection-count overlay for the initial baseline. The final section includes test evaluation and an ONNX comparison.

On the single image used for the ONNX comparison, both models returned two detections with matching confidence scores to four decimal places. This is a basic sample check, not a full numerical equivalence test.

The final trained checkpoint and export are generated at:

```text
/kaggle/working/runs/hardhat_yolo11s_2class/weights/best.pt
/kaggle/working/runs/hardhat_yolo11s_2class/weights/best.onnx
```

Weights are separate files and are not embedded in the notebook.

## How to Run

1. Download or clone this repository and open `hard_hat_detection_yolo11.ipynb` in Kaggle.
2. Attach the Hard Hat Detection dataset linked above.
3. Enable a GPU and internet access for package installation and pretrained-weight downloads.
4. Verify the input paths. The saved notebook uses:

   ```text
   /kaggle/input/datasets/andrewmvd/hard-hat-detection/images
   /kaggle/input/datasets/andrewmvd/hard-hat-detection/annotations
   ```

5. Run the cells in order. The notebook contains both the initial baseline and the final two-class experiment.
6. Download the trained weights, generated plots, and `/kaggle/working/requirements.txt` from the notebook outputs.

The notebook installs Ultralytics itself. Rerunning the final preparation cell deletes and rebuilds its `yolo_dataset_v2` output directory. Repeated training can create suffixed run directories; update checkpoint paths if that occurs.

## Tools

Python, Ultralytics YOLO, PyTorch, OpenCV, NumPy, pandas, scikit-learn, Matplotlib, PyYAML, and ONNX Runtime.

## Limitations

- Helmet and head boxes do not automatically establish each worker’s safety status; the notebook does not associate detections with individual workers or track them.
- Average-hash grouping is a limited duplicate check.
- The initial experiment and final experiment use the same source dataset; the final test split is not a new external dataset.
- Performance in other sites, lighting conditions, or camera views requires additional evaluation.
- The notebook demonstrates still-image inference and export, rather than a deployed webcam application.

## Acknowledgments

Dataset: [Andrew MVD — Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection).

Detection framework and pretrained model: [Ultralytics](https://github.com/ultralytics/ultralytics).

Community notebooks studied during development: *Safety Helmet Detection YOLOv8* and *Helmet Detection with YOLOv8*. Add their original Kaggle URLs and author credits when sharing any adapted code, and follow the relevant licenses.
