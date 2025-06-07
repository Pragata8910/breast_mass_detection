import json
import os
import cv2
import glob
import numpy as np
from datetime import datetime


def detect_boxes(image_path):
    """Detect bounding boxes in mammogram images using adaptive thresholding and contour detection."""

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 5
    )

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 50 < w < 2000 and 50 < h < 2000:
            boxes.append([x, y, w, h])
    return boxes


def create_coco_json(image_dir, output_file):
    list_img = []
    list_annotations = []
    annotation_id = 1

    image_paths = glob.glob(os.path.join(image_dir, "*.png"))

    for image_id, image_path in enumerate(image_paths, 1):
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            height, width = img.shape

            list_img.append(
                {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": os.path.basename(image_path),
                    "license": 1,
                    "date_captured": datetime.now().strftime("%Y-%m-%d"),
                }
            )
            boxes = detect_boxes(image_path)
            for box in boxes:
                x, y, w, h = box
                list_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": [x, y, w, h],
                        "area": float(w * h),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

        data = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0.1",
                "description": "Generated COCO from CBIS-DDSM mammogram images",
                "date_created": datetime.now().strftime("%Y-%m-%d"),
            },
            "licenses": [{"url": "None", "id": 1, "name": "None"}],
            "categories": [{"supercategory": "none", "id": 0, "name": "cancer"}],
            "images": list_img,
            "annotations": list_annotations,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Created COCO JSON at {output_file}")
        print(f"Stats: {len(list_img)} images, {len(list_annotations)} annotations")
