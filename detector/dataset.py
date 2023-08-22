import os
import numpy as np
import torch
from PIL import Image

from config import (
    RENDERS_PATH,
    MASKS_PATH
)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, object_names, transforms):
        self.object_names = object_names
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = []
        self.masks = []
        self.labels = []
        for object_name in object_names:
            imgs = list(sorted(RENDERS_PATH(object_name).iterdir()))
            masks = list(sorted(MASKS_PATH(object_name).iterdir()))
            assert len(imgs) == len(masks)
            self.imgs += imgs
            self.masks += masks
            self.labels += [object_name for _ in range(len(imgs))]

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        curr_obj_name = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)

        if self.transforms is not None:
            img, masks, labels = self.transforms(img, mask, curr_obj_name)

        # get bounding box coordinates for each mask
        num_objs = len(labels)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

            labels[i] = self.object_names.index(labels[i])+1

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.imgs)

    