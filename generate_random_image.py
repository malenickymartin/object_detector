from detector import transforms as T
import os
import numpy as np
from PIL import Image
from config import (
    RENDERS_PATH,
    MASKS_PATH,
    OBJECT_PATH
)

def get_transform():
    transforms = []
    transforms.append(T.AddRenders())
    transforms.append(T.AddBackground())
    return T.Compose(transforms)

transforms = get_transform()
object_name = "rc-car"

images_list = os.listdir(RENDERS_PATH(object_name))
image_num = np.random.randint(0, len(images_list))
print(f"Image number {image_num}")
img = Image.open(RENDERS_PATH(object_name) / f"render{image_num}.png").convert("RGB")
target = {"masks" : np.array(Image.open(MASKS_PATH(object_name) / f"mask{image_num}.png"))}
target["masks"] = np.transpose(target["masks"][:,:,None], (2,0,1))

img, target = transforms(img, target, object_name)

img.save(OBJECT_PATH(object_name) / "test1.png")


