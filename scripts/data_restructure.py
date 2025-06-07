import os
import shutil
from sklearn.model_selection import train_test_split

image_dir = "/Users/pragata/Datas(all)/cbis ddsm/results"
label_dir = "/Users/pragata/Datas(all)/cbis ddsm/results"
output_dir = "/Users/pragata/Datas(all)/cbis ddsm/dataset"

images = [f for f in os.listdir(image_dir) if f.endswith(".png")]
labels = [f.replace(".png", ".txt") for f in images]

train_img, val_img, train_lbl, val_lbl = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

os.makedirs(f"{output_dir}/images/train", exist_ok=True)
os.makedirs(f"{output_dir}/images/val", exist_ok=True)
os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
os.makedirs(f"{output_dir}/labels/val", exist_ok=True)

for img, lbl in zip(train_img, train_lbl):
    shutil.copy(f"{image_dir}/{img}", f"{output_dir}/images/train/{img}")
    shutil.copy(f"{label_dir}/{lbl}", f"{output_dir}/labels/train/{lbl}")

for img, lbl in zip(val_img, val_lbl):
    shutil.copy(f"{image_dir}/{img}", f"{output_dir}/images/val/{img}")
    shutil.copy(f"{label_dir}/{lbl}", f"{output_dir}/labels/val/{lbl}")
