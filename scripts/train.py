from ultralytics import YOLO
from pathlib import Path

YAML_PATH = "/Users/pragata/Datas(all)/cbis ddsm/dataset/dataset.yaml"
DATASET_ROOT = "/Users/pragata/Datas(all)/cbis ddsm/dataset"

assert Path(YAML_PATH).exists(), f"YAML not found at {YAML_PATH}"
assert Path(DATASET_ROOT).exists(), f"Dataset not found at {DATASET_ROOT}"

# TRAIN
model = YOLO("yolov8s.pt").to("mps")
model.train(
    data=YAML_PATH,
    epochs=100,
    imgsz=640,
    batch=8,
    device="mps",
    workers=0,
    optimizer="AdamW",
    warmup_epochs=3,
    cache="disk",
    single_cls=True,
    amp=True,
    profile=True,
)
