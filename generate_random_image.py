from detector import transforms as T
import os
import numpy as np
from PIL import Image
from config import (
    RENDERS_PATH,
    MASKS_PATH,
    RESULT_PATH
)

def get_transform(object_name):
    transforms = []
    transforms.append(T.AddRenders(object_name))
    transforms.append(T.AddBackground())
    return T.Compose(transforms)

dataset_name = "train"
object_name = "rc-car"
model_folder = "rc-car"

transforms = get_transform(object_name)
images_list = os.listdir(RENDERS_PATH(dataset_name, object_name))
image_num = np.random.randint(0, len(images_list))
print(f"Image number {image_num}")
img = np.array(Image.open(RENDERS_PATH(dataset_name, object_name) / f"render{image_num}.png").convert("RGB"))
mask = np.array(Image.open(MASKS_PATH(dataset_name, object_name) / f"mask{image_num}.png"))

img, _, _ = transforms(img, mask, object_name)

img.save(RESULT_PATH(model_folder) / "test6.png")


