import os
import numpy as np
import torch
from PIL import Image

from config import (
    RENDERS_PATH,
    MASKS_PATH
)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, object_name, transforms, class_id=1):
        self.object_name = object_name
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        #self.imgs = list(sorted(os.listdir(RENDERS_PATH(object_name))))
        self.imgs = list(sorted(RENDERS_PATH(object_name).iterdir()))
        #self.masks = list(sorted(os.listdir(MASKS_PATH(object_name))))
        self.masks = list(sorted(MASKS_PATH(object_name).iterdir()))
        assert class_id > 0
        self.class_id = class_id

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        mask_path =  self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64) * self.class_id
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

        if self.transforms is not None:
            img, target = self.transforms(img, target, self.object_name)

        return img, target

    def __len__(self):
        return len(self.imgs)
    