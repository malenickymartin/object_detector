import os
import numpy as np
import torch
from PIL import Image
from detector.transforms import Compose

from config import RENDERS_PATH, MASKS_PATH, DATASET_PATH, OBJECT_PATH


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_dataset: str, transforms: Compose):
        self.object_names = sorted(os.listdir(DATASET_PATH(train_dataset)))
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = []
        self.masks = []
        self.labels = []
        for object_name in self.object_names:
            if RENDERS_PATH(train_dataset, object_name).is_dir() and MASKS_PATH(
                train_dataset, object_name
            ):
                imgs = list(sorted(RENDERS_PATH(train_dataset, object_name).iterdir()))
                masks = list(sorted(MASKS_PATH(train_dataset, object_name).iterdir()))
                assert len(imgs) == len(
                    masks
                ), "Number of masks is not the same as number of renders."
                self.imgs += imgs
                self.masks += masks
                self.labels += [object_name for _ in range(len(imgs))]
            else:
                print(
                    f"WARNING: Object {object_name} does not have any renders or masks in {OBJECT_PATH(train_dataset, object_name)} and will be excluded during training"
                )

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        curr_obj_name = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        mask = Image.open(mask_path)
        mask = np.array(mask)

        if self.transforms is not None:
            img, masks, labels = self.transforms(img, mask, curr_obj_name)

        num_objs = len(labels)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

            labels[i] = self.object_names.index(labels[i]) + 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
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
