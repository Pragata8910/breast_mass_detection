#!/usr/bin/env python3
"""
Simple Breast Mass Detection Script
Usage: python detect.py --image path/to/image.jpg
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def detect_masses(image_path, model_path="best.pt", confidence=0.5):
    """
    Detect breast masses in mammogram image

    Args:
        image_path: Path to mammogram image
        model_path: Path to trained model (default: best.pt)
        confidence: Detection confidence threshold (default: 0.5)
    """

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Analyzing image: {image_path}")
    results = model(image_path, conf=confidence, save=True)

    detections = 0
    high_conf_detections = 0

    for result in results:
        if result.boxes is not None:
            detections = len(result.boxes)

            print("\n=== DETECTION RESULTS ===")
            print(f"Image: {Path(image_path).name}")
            print(f"Total detections: {detections}")

            for i, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                print(f"Detection {i + 1}: Confidence {conf:.3f}")

                if conf >= 0.64:
                    high_conf_detections += 1

            if high_conf_detections > 0:
                print(
                    f"\nHIGH PRIORITY: {high_conf_detections} high-confidence detection(s)"
                )
                print("   Recommend immediate radiologist review")
            elif detections > 0:
                print(f"\n MEDIUM PRIORITY: {detections} detection(s) found")
                print("   Recommend radiologist review")
            else:
                print("\nLOW PRIORITY: No significant masses detected")
        else:
            print(f"\n✅ No detections found in {Path(image_path).name}")

    print("\nResults saved to: runs/detect/predict/")
    return detections, high_conf_detections


def detect_folder(folder_path, model_path="best.pt", confidence=0.5):
    """Detect masses in all images in a folder"""

    folder = Path(folder_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    image_files = [f for f in folder.glob("*") if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {folder_path}")
        return

    print(f"Found {len(image_files)} images to process")

    total_detections = 0
    total_high_conf = 0

    for image_file in image_files:
        detections, high_conf = detect_masses(image_file, model_path, confidence)
        total_detections += detections
        total_high_conf += high_conf
        print("-" * 50)

    print("\n=== SUMMARY ===")
    print(f"Images processed: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"High confidence detections: {total_high_conf}")


def main():
    parser = argparse.ArgumentParser(description="Breast Mass Detection")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--model", type=str, default="best.pt", help="Model path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")

    args = parser.parse_args()

    if not args.image and not args.folder:
        print("Please provide either --image or --folder argument")
        return

    if args.image:
        detect_masses(args.image, args.model, args.conf)
    elif args.folder:
        detect_folder(args.folder, args.model, args.conf)


if __name__ == "__main__":
    main()
