from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
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
    def forward(self, render, mask, object_names, curr_obj_name):
        mask = mask > 0
        all_masks = mask[:,:,None]
        masks = mask[None, :, :]
        only_object_mask = mask > 0
        labels = [curr_obj_name]
        
        objects_dir = os.listdir(OBJECTS_PATH)
        objects = []
        for object in objects_dir:
            if os.path.isdir(RENDERS_PATH(object)) and object != curr_obj_name:
                objects.append({"label": object,"num_renders":len(os.listdir(RENDERS_PATH(object)))})

        for object in np.random.choice(objects, np.random.randint(0, len(objects)+1), replace=False):
            render_num = np.random.randint(0, object["num_renders"])
            new_render = np.array(Image.open(RENDERS_PATH(object["label"]) / f"render{render_num}.png"))
            new_mask = np.array(Image.open(MASKS_PATH(object["label"]) / f"mask{render_num}.png"))
            new_mask = new_mask > 0
            ovelay_rate = np.sum(np.logical_and(only_object_mask, new_mask))/np.sum(only_object_mask)

            if object["label"] in object_names:
                masks = np.append(masks, new_mask[None,:,:], axis=0)
                labels.append(object["label"])
                only_object_mask = np.logical_or(only_object_mask, new_mask)

            new_mask = new_mask[:,:,None]

            if np.random.choice([True, False]) and ovelay_rate < 0.75:
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
