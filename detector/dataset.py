import torch
from PIL import Image
import json
from torchvision import transforms
import albumentations as A
import numpy as np
from tqdm import tqdm

from config import DATASET_PATH

class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_dataset: str):
        self.transforms = A.Compose(
            [
                A.Flip(p=0.5),
                A.Blur(blur_limit=(3,5), p=0.5),
                A.CLAHE(clip_limit=(1,2), p=0.125),
                A.Downscale(scale_min=0.5, scale_max=0.99, interpolation=1, p=0.125),
                A.ISONoise(color_shift=(0,0.2), intensity=(0.1,1), p=0.25),
                A.RandomBrightnessContrast(brightness_limit=(-0.1,0.25), contrast_limit=(-0.1,0.25), p=0.5),
            ],
            bbox_params=A.BboxParams(format="pascal_voc"),
            p=0.8
        )

        self.imgs_path = []
        self.masks_path = []
        self.labels = []
        self.bboxes = []
        self.area = []

        for scene in tqdm(list((DATASET_PATH(train_dataset) / "train_pbr").iterdir())):
            with open(scene / "scene_gt.json", "r") as f:
                scene_gt = json.load(f)
            with open(scene / "scene_gt_info.json", "r") as f:
                scene_gt_info = json.load(f)

            for view in range(len(scene_gt)):
                labels = []
                bboxes = []
                masks_path = []
                area = []
                for obj in range(len(scene_gt[str(view)])):
                    if scene_gt_info[str(view)][obj]["px_count_visib"] > 300:
                        labels.append(scene_gt[str(view)][obj]["obj_id"])
                        bbox = scene_gt_info[str(view)][obj]["bbox_visib"]
                        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                        bboxes.append(bbox)
                        area.append(scene_gt_info[str(view)][obj]["px_count_visib"])
                        masks_path.append(
                            scene / "mask_visib" / f"{view:06d}_{obj:06d}.png"
                        )
                if labels != []:
                    self.labels.append(labels)
                    self.bboxes.append(bboxes)
                    self.masks_path.append(masks_path)
                    self.area.append(area)
                    self.imgs_path.append(scene / "rgb" / f"{view:06d}.jpg")

    def __getitem__(self, idx):
        # load images and masks
        img = Image.open(self.imgs_path[idx]).convert("RGB")
        img = np.array(img)

        masks = []
        for i in range(len(self.masks_path[idx])):
            masks.append(np.array(Image.open(self.masks_path[idx][i])))

        labels = self.labels[idx]
        bboxes = self.bboxes[idx]
        area = self.area[idx]

        if self.transforms is not None:
            transformed = self.transforms(image=img, masks=masks, bboxes=[bboxes[i]+[labels[i]] for i in range(len(bboxes))])
            img = transformed["image"]
            masks = np.array(transformed["masks"])//255
            bboxes = np.array(transformed["bboxes"])[:,0:4]

        img = transforms.ToTensor()(img)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.imgs_path)
