# Hard Hat Detection with YOLO11s

**Author: Zaid Rasras**

An object detection project that uses YOLO11s to detect helmets and heads in construction-site images.

📓 [View the project notebook](hard-hat-detection-by-zaidKaggle.ipynb)

## Dataset

This project uses the [Hard Hat Detection dataset by Andrew MVD](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection).

The dataset contains:

- 5,000 images.
- Pascal VOC XML bounding-box annotations.
- Three original classes: `helmet`, `head`, and `person`.

The notebook includes an initial three-class experiment, followed by a final model trained on two classes:

| Class ID | Class |
|---|---|
| 0 | helmet |
| 1 | head |

## Project Workflow

1. Read the XML annotations and organize objects in a DataFrame.
2. Convert Pascal VOC bounding boxes into normalized YOLO labels.
3. Train and evaluate an initial three-class model.
4. Prepare a separate two-class training, validation, and test split.
5. Train the final YOLO11s detector.
6. Select a confidence threshold using validation data.
7. Evaluate the final model on the test split.
8. Export the model to ONNX and compare sample predictions.

## Annotation Format

Each generated YOLO label file contains one line per object:

```text
class_id x_center y_center width height
```

The coordinates and box dimensions are normalized by the image width and height.

## Final Dataset Split

Images with identical 8×8 average hashes are grouped before splitting to reduce some duplicate leakage. This method does not detect every similar image or related scene.

| Split | Images |
|---|---:|
| Training | 3,509 |
| Validation | 746 |
| Test | 745 |

## Model and Training

| Setting | Value |
|---|---|
| Model | YOLO11s |
| Starting weights | `yolo11s.pt` |
| Epochs | 30 |
| Image size | 640 |
| Batch size | 16 |
| Early-stopping patience | 10 |
| GPU | NVIDIA Tesla T4 |

The final training configuration is:

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

The confidence threshold was selected using the validation mean F1 curve, giving approximately **0.452**.

The final evaluation used **745 test images containing 3,761 annotated objects**.

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| All classes | 0.939 | 0.891 | 0.8806 | 0.6017 |
| helmet | 0.9608 | 0.8904 | 0.8912 | 0.608 |
| head | 0.9168 | 0.8917 | 0.8699 | 0.595 |

**Evaluation note:** the test evaluation filters predictions using the selected confidence threshold before calculating AP. These AP values are not directly comparable to evaluations using a much lower confidence cutoff to construct the full precision–recall curve.

These are object detection metrics, not classification accuracy.

## Visualizations and Export

The notebook includes:

- Training curves and a confusion matrix for the initial model.
- Sample bounding-box predictions.
- A helmet/head detection-count overlay.
- Final two-class validation and test results.
- ONNX export and a sample comparison with PyTorch predictions.

The ONNX comparison returned the same detection count and matching confidence scores to four decimal places on one image. This is a basic check rather than comprehensive equivalence testing.

## How to Run

1. Open `hard-hat-detection-by-zaidKaggle.ipynb` in Kaggle.
2. Attach the Hard Hat Detection dataset.
3. Enable GPU acceleration and internet access for installing dependencies and downloading pretrained weights.
4. Check the dataset paths used in the notebook:

   ```text
   /kaggle/input/datasets/andrewmvd/hard-hat-detection/images
   /kaggle/input/datasets/andrewmvd/hard-hat-detection/annotations
   ```

5. Run the cells in order.
6. Download the generated weights, plots, ONNX model, and environment requirements from the Kaggle outputs.

The final model files are generated at:

```text
/kaggle/working/runs/hardhat_yolo11s_2class/weights/best.pt
/kaggle/working/runs/hardhat_yolo11s_2class/weights/best.onnx
```

Model weights are separate files and are not embedded in the notebook.

Repeated training may create a run directory with a different suffix. Update checkpoint paths if needed.

## Technologies

- Python
- Ultralytics YOLO and PyTorch
- OpenCV
- NumPy and pandas
- scikit-learn
- Matplotlib
- PyYAML
- ONNX Runtime

## Limitations

- Detection counts do not establish the safety status of every worker.
- The model does not associate helmets and heads with individual people or track workers.
- Average-hash grouping provides only a limited duplicate check.
- The initial and final experiments use the same source dataset; the final test split is not an external dataset.
- Performance in new locations and camera conditions requires further evaluation.
- This project demonstrates still-image inference and model export, rather than a deployed safety-monitoring application.

## Acknowledgments

- [Andrew MVD — Hard Hat Detection dataset](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
- [Ultralytics — detection framework and pretrained models](https://github.com/ultralytics/ultralytics)
- Community YOLO notebooks studied during project development.

## Author

[Zaid Rasras](https://github.com/zaidrasras31)
