import json
import os
from pathlib import Path


def coco_to_yolo(coco_json_path, output_dir):
    """Convert COCO format annotations to YOLO format"""
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    output_dir = str(Path(output_dir).expanduser().resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    test_file = Path(output_dir) / "permission_test.txt"
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)

    for image in coco_data["images"]:
        img_id = image["id"]
        img_width = image["width"]
        img_height = image["height"]
        base_name = Path(image["file_name"]).stem

        yolo_file = Path(output_dir) / f"{base_name}.txt"

        annotations = [
            ann for ann in coco_data["annotations"] if ann["image_id"] == img_id
        ]

        yolo_file = str(yolo_file.resolve())

        with open(yolo_file, "w") as f:
            for ann in annotations:
                bbox = ann["bbox"]
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height

                f.write(f"0 {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":
    coco_to_yolo("coco.json", "/Users/pragata/Datas(all)/cbis ddsm/results")
