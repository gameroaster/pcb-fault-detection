# TENSORFLOW LITE OBJECT DETECTION -  ASSIGNMENT


## OVERVIEW 
This project demonstrates the end-to-end creation of an object detection model optimized for edge devices using YOLOv8 and TensorFlow Lite.
The workflow includes dataset preparation, model training, evaluation, and conversion to .tflite format for deployment on mobile or embedded platforms.

This project implements TensorFlow Lite object detection models for following real-world use cases:
- PCB Fault Detection

The models detect objects in images and output bounding boxes, class labels, and confidence scores, optimized for edge and mobile deployment using TensorFlow Lite.

### Use Case : PCB Fault Detection

### Objective - Detect PCB manufacturing defects such as:

- Soldering defects
- Missing components
- Damaged components


## DATASET
Source: https://universe.roboflow.com/object-detection-dt-wzpc6/pcb-dataset-defect
- 219 manually selected and annotated images

The dataset is a object detection dataset in a ZIP file.
### NOTE : I cant upload the dataset on github since dataset size is exceeding 25mb
The dataset is automatically split into:
- 90% Training
- 10% Validation

### Annotation tool : Label Studio (open source) 
https://labelstud.io/

### Annotation format: Text (txt)

### Classes
- Hole
- Mouse bite
- Open circuit
- Short
- Spur
- Spurious
  
### The dataset directory structure follows the standard YOLO format:
```bash
  custom_data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

A data.yaml file is programmatically generated in the notebook to define:
- Dataset paths
- Number of classes
- Class names

## Model Architecture

- Model : YOLOv8 Small (yolov8s)
- Framework : Ultralytics YOLO
- Architecture Type : One-stage object detector
- Backbone + Head : CNN-based feature extractor with multi-scale detection heads
- Input Size : 640 × 640

### YOLOv8 is chosen for its:

- High inference speed
- Good accuracy-to-size tradeoff
- Native support for TensorFlow Lite export

## TensorFlow Lite Conversion 
YOLOv8 to TensorFlow Lite Conversion :
This project uses YOLOv8 for object detection and converts the trained model into TensorFlow Lite (TFLite) format for deployment on edge and mobile devices.
Ultralytics provides native support for exporting YOLOv8 models to TFLite, making the conversion process straightforward.

```bash
  pip install ultralytics
  from ultralytics import YOLO
  model = YOLO("yolov8n.pt")
  model.export(format="tflite")
```

## Training Approach

- Pretrained weights : yolov8s.pt
- Training type : Transfer Learning
- Epochs : 60
- Image size : 640
- Optimizer & Loss : Handled internally by Ultralytics YOLO
- Environment : Google Colab with GPU acceleration

Model Evaluation :
- Predictions are generated on validation images.
- Sample detection outputs are visualized directly in the notebook.

This produces a .tflite model suitable for:
- Android application
- Edge devices
- Embedded AI systems

## Output
Each model outputs:

1. Bounding boxes
2. Class labels
3. Confidence scores


![train_batch2](https://github.com/user-attachments/assets/56ff77b6-57b1-4fdc-a7d5-1b2840005027)


![train_batch1](https://github.com/user-attachments/assets/eb2b129e-9aaf-4b91-912d-1086e9a0a0dc)



<img width="1920" height="1080" alt="Screenshot 2026-01-18 141322" src="https://github.com/user-attachments/assets/61deca69-43b8-4ed0-9a23-87ef8a9ca5eb" />


<img width="1920" height="1080" alt="Screenshot 2026-01-18 141559" src="https://github.com/user-attachments/assets/69dbc5eb-728a-4e56-b9a9-53e20890e94b" />


<img width="670" height="819" alt="Screenshot 2026-01-18 141746" src="https://github.com/user-attachments/assets/4dc296ef-c867-4bb5-9b3c-d667b6fee802" />



### Model files:
- my_model.pt
- my_model.tflite


## Known Limitations & Challenges

- Quantization not applied
  - The exported TFLite model is not fully integer-quantized, which may limit performance on very low-power devices.

- Dataset size dependency 
  - Model accuracy is highly dependent on dataset size and class balance.

- Hardware constraints
  - Training requires GPU support for reasonable training time.

- Edge deployment tuning
  - Further optimization (INT8 quantization, pruning) may be required for real-time deployment on microcontrollers.

## Future Improvements
- Apply INT8 quantization for better edge performance
- Train larger YOLOv8 variants for higher accuracy
- Perform extensive validation using mAP metrics
- Integrate with Android / Raspberry Pi inference pipelines


## Conclusion

### This project demonstrates an end-to-end object detection pipeline:
- Data preparation
- Model training
- Optimization
- Deployment using TensorFlow Lite

The solution is suitable for embedded and mobile applications.
