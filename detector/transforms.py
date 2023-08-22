from typing import Dict, List

import torch
from torch import nn
from torchvision.transforms import functional as F, transforms as T
import os
import numpy as np
from PIL import Image

from config import (
    BACKGROUNDS_PATH,
    OBJECTS_PATH,
    RENDERS_PATH,
    MASKS_PATH
)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
    
class AddRenders(nn.Module):
    def __init__(self, object_names, max_data_objs = 5, max_other_objs = 5):
        super().__init__()
        self.max_data_objs = max_data_objs
        self.max_other_objs = max_other_objs
        self.object_names = object_names

        objects_dir = os.listdir(OBJECTS_PATH)
        self.other_objects = []
        self.data_objects = []
        for object in objects_dir:
            if RENDERS_PATH(object).is_dir():
                if object in object_names:
                    self.data_objects.append({"label": object,
                                              "renders":list(sorted(RENDERS_PATH(object).iterdir())),
                                              "masks":list(sorted(MASKS_PATH(object).iterdir()))})
                    assert len(self.data_objects[-1]["renders"]) == len(self.data_objects[-1]["masks"])
                else:
                    self.other_objects.append({"label": object,
                                               "renders":list(sorted(RENDERS_PATH(object).iterdir())),
                                               "masks":list(sorted(MASKS_PATH(object).iterdir()))})
                    assert len(self.other_objects[-1]["renders"]) == len(self.other_objects[-1]["masks"])

    def forward(self, render, mask, curr_obj_name):
        mask = mask > 0
        all_masks = mask[:,:,None]
        masks = mask[None, :, :]
        only_object_mask = mask > 0
        labels = [curr_obj_name]

        data_objects = []
        for object in self.data_objects:
            if object["label"] != curr_obj_name:
                data_objects.append(object)
        
        rand_data_objs = np.random.randint(0, min(self.max_data_objs, len(data_objects)) + 1) 
        rand_other_objs = np.random.randint(0, min(self.max_other_objs, len(self.other_objects)) + 1)
        rand_data_objs = np.random.choice(data_objects, rand_data_objs, replace=False)
        rand_other_objs = np.random.choice(self.other_objects, rand_other_objs, replace=False)

        for object in np.concatenate((rand_data_objs, rand_other_objs)):
            render_num = np.random.randint(0, len(object["renders"]))
            new_render = np.array(Image.open(object["renders"][render_num]))
            new_mask = np.array(Image.open(object["masks"][render_num]))
            new_mask = new_mask > 0
            ovelay_rate = np.sum(np.logical_and(only_object_mask, new_mask))/np.sum(only_object_mask)

            if object["label"] in self.object_names:
                masks = np.append(masks, new_mask[None,:,:], axis=0)
                labels.append(object["label"])
                only_object_mask = np.logical_or(only_object_mask, new_mask)

            new_mask = new_mask[:,:,None]

            if np.random.random() < 0.5 and ovelay_rate < 0.75:
                render = np.where(new_mask, new_render, render)
            else:
                render = np.where(all_masks, render, new_render)
            all_masks = np.logical_or(new_mask, all_masks)

        return render, masks, all_masks, labels
    
class AddBackground(nn.Module):
    def forward(self, render, masks, all_masks, labels):
        backgrounds = os.listdir(BACKGROUNDS_PATH)
        background = np.random.choice(backgrounds)
        background = Image.open(BACKGROUNDS_PATH / background)
        render = np.array(render)

        background = background.resize(render.shape[:2][::-1])
        result = np.where(all_masks, render, background)

        result = Image.fromarray(result)

        result = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.05)(result)
        result = T.GaussianBlur(kernel_size=5, sigma=(0.2,2))(result)

        return result, masks, labels
    
class PILToTensor(nn.Module):
    def forward(self, image, masks, labels):
        image = F.pil_to_tensor(image)
        return image, masks, labels

class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image, masks, labels):
        image = F.convert_image_dtype(image, self.dtype)
        return image, masks, labels
