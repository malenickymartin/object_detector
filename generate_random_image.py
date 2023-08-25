from detector import transforms as T
import os
import numpy as np
from PIL import Image
from config import (
    RENDERS_PATH,
    MASKS_PATH,
    MODEL_PATH
)

def get_transform(object_name: str, augs_dataset: str):
    transforms = []
    transforms.append(T.AddRenders(object_name, augs_dataset))
    transforms.append(T.AddBackground())
    return T.Compose(transforms)

dataset_name = "train"
augs_dataset = "augs"
object_name = "mustard"
model_folder = "train-train_aug-augs"

transforms = get_transform(dataset_name, augs_dataset)
images_list = os.listdir(RENDERS_PATH(dataset_name, object_name))
image_num = np.random.randint(0, len(images_list))
print(f"Image number {image_num}")
img = np.array(Image.open(RENDERS_PATH(dataset_name, object_name) / f"render{image_num}.png").convert("RGB"))
mask = np.array(Image.open(MASKS_PATH(dataset_name, object_name) / f"mask{image_num}.png"))

img, _, _ = transforms(img, mask, object_name)

img.save(MODEL_PATH(model_folder) / "test6.png")


