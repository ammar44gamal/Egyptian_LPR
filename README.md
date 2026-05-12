# Egyptian License Plate Recognition (LPR) System



![Python](https://img.shields.io/badge/Python-3.8%252B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%252B-green)
![License-MIT](https://img.shields.io/badge/License-MIT-lightgrey)
<br>

#### A production-ready, deep learning-based system for detecting and recognizing Egyptian license plates. This project implements a robust two-stage pipeline using YOLOv8, specifically designed to handle the unique challenges of Arabic-alphanumeric characters and real-world conditions.


## 🎯 Features 

**Two-Stage Detection** : Separate models for plate localization and character recognition.
<br>

**High Accuracy** : Achieves 98.9% mAP on character recognition with zero confusion between mirror-image Arabic characters.
<br>

**Robust Preprocessing**: Handles blurry, shadowed, and low-quality images with CLAHE, upscaling, and grayscale conversion.
<br>

**Bidirectional Text Support** : Correctly processes mixed LTR (numbers) and RTL (Arabic letters) text using arabic-reshaper and python-bidi.
<br>

**User-Friendly GUI** : Built with CustomTkinter for easy image upload and visualization.
<br>

**Real-World Ready** : Trained on locally sourced Egyptian plate datasets.
<br>


## 📁 Project Structure
<img width="718" height="457" alt="Screenshot 2026-01-24 174658" src="https://github.com/user-attachments/assets/9096ff9d-1fea-4da8-8e96-69309b522c80" />


## ⚙️ Installation
#### Clone the repository:

bash
git clone https://github.com/ammar44gamal/Egyptian_LPR.git

cd egyptian-lpr

#### Create a virtual environment (recommended):

bash

python -m venv venv

source venv\Scripts\activate

#### Install dependencies:

bash
pip install -r requirements.txt


## 📊 Datasets
#### The system uses two separate datasets, both annotated in YOLO format:

**Plate Detection Dataset**: Images of Egyptian cars with bounding boxes around license plates.

--> Source: Egyptian Cars Dataset on Roboflow

**Character Recognition Dataset**: Close-up images of license plates with annotations for each Arabic letter and Eastern Arabic numeral.

--> Source: Egyptian Car Plates Dataset on Roboflow


## 🤖 Training the Models
1. Plate Detector (Stage 1)
bash
python src/train_plate.py
Uses YOLOv8n with COCO pretrained weights

Trained for 30 epochs

Output saved to runs_plate/plate_model/weights/best.pt

2. Character Detector (Stage 2)
bash
python src/train_chars.py
Uses YOLOv8n with COCO pretrained weights

Critical: Horizontal flip augmentation disabled (fliplr=0.0) to prevent confusion between mirror characters

Trained for 50 epochs

Achieved 98.9% mAP

Output saved to runs_chars/char_model_no_flip/weights/best.pt


## 🚀 Usage
Command-Line Interface
bash
python src/main.py --image path/to/your/image.jpg
Desktop GUI Application
bash
python src/gui.py

### The GUI provides:

#### Image upload button
#### Visual display of detected plates and characters
#### Formatted output showing separated numbers and Arabic letters


## 🔧 Technical Highlights

### Preprocessing Pipeline
##### Plate Cropping with Padding: 10-pixel buffer to prevent edge character cutoff


#### Image Enhancement:
##### Grayscale conversion
##### 2x upscaling using cubic interpolation
##### **CLAHE (Contrast Limited Adaptive Histogram Equalization)**:  Used for shadow removal


#### Intelligent Post-Processing:
##### **Class-Agnostic NMS**: Prevents overlapping character detections
##### **Bidirectional Sorting**: Numbers sorted LTR, Arabic letters sorted RTL


#### Text Reshaping: 
##### **Arabic Reshaper**: Arabic characters properly connected and displayed


## Model Architecture

### Plate Detection Model
```python
model = YOLO('yolov8n.pt')
model.train(data='dataset_plates/data.yaml', epochs=30)
```


### Character Recognition Model
```python
model = YOLO('yolov8n.pt')
model.train(data='dataset_characters/data.yaml', epochs=50, fliplr=0.0)
```


## 📈 Performance

| Model               | mAP@0.5 | Precision | Recall | Epochs |
|---------------------|---------|-----------|--------|--------|
| **Plate Detector**  | 96.2%   | 0.95      | 0.94   | 30     |
| **Character Recognizer** | 98.9%   | 0.99      | 0.98   | 50     |

#### 🎯 Key Achievement: Zero confusion between visually similar Arabic characters (e.g., ٦ vs ٢, ج vs ح) through constrained augmentation.



## ⚙️ Key Technical Challenges & Solutions:

Problem: The model confused mirror-image characters like ٦ (6) and ٢ (2).

Solution: Implemented a constrained training protocol, disabling horizontal flips (fliplr=0.0) to teach the model their distinct identities, achieving a 98.9% mAP.

Problem: Low-quality inputs (blurry, shadowed plates).

Solution: Built a robust preprocessing pipeline using CLAHE (Contrast Limited Adaptive Histogram Equalization), upscaling, and grayscale conversion to act as a "shadow killer" before recognition.

Problem: Mixed text direction (LTR numbers & RTL Arabic letters).

Solution: Developed a custom split-sorting algorithm and integrated arabic-reshaper + python-bidi libraries for perfect text reconstruction and display.

Problem: Overlapping detections for characters.

Solution: Used Class-Agnostic Non-Maximum Suppression (NMS) to enforce logical spatial constraints.


## 🛠️ Dependencies

| Library          | Purpose |
|------------------|---------|
| **ultralytics**  | YOLOv8 model training and inference |
| **opencv-python**| Image processing and computer vision |
| **numpy**        | Numerical operations and array manipulation |
| **arabic-reshaper** | Arabic text reshaping for proper display |
| **python-bidi**  | Bidirectional text support |
| **customtkinter**| Modern GUI interface |
| **pillow**       | Image handling in GUI |

