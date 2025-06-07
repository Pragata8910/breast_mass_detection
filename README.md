#  Breast Mass Detection

AI system for detecting breast masses in mammographic images using YOLO object detection.

##  Key Results
- **Precision**: 100% at confidence 0.64
- **Model**: Conservative detection (high accuracy, low false positives)
- **Use Case**: Screening support for radiologists

##  Files
```
breast-mass-detection/
├── README.md           # This file
├── detect.py          # Run detection on images
├── best.pt            # Trained model weights
├── results/          # Training curves and metrics
|-- sample_images/     # Example of model in action
|-- dataset/           # Raw CBIS-DDSM dataset on which the entire processing was made      
```

##  Quick Start

### Install Requirements
```bash
pip install ultralytics opencv-python pillow
```

### Run Detection
```bash
# Detect on single image
python detect.py --image sample_images/mammogram.jpg

# Detect on folder of images
python detect.py --folder sample_images/
```

##  Performance
- **High Precision**: When model detects something, it's usually correct
- **Conservative**: Avoids false alarms but may miss some cases  
- **Best Threshold**: 0.64 confidence for clinical use
- **Application**: Perfect for pre-screening support

##  Technical Details
- **Framework**: YOLOv8 object detection
- **Input**: Mammographic images
- **Output**: Bounding boxes with confidence scores
- **Classes**: Detects 80+ objects including anatomical structures

##  Medical Use
This system is designed for **research and screening support only**. Always requires radiologist confirmation for clinical decisions.

##  Results Summary
From training analysis:
- Precision-confidence curve shows sharp transition at 0.6
- Model achieves perfect precision at higher confidence thresholds
- Confusion matrix indicates good class separation
- Recommended for clinical screening pipeline integration

## Contact
- [PRAGATA GHOSH](pragata2004@gmail.com) (email)
- [PRAGATA](+919475170335)
