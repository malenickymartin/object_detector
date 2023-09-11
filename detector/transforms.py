import torch
from torch import nn
from torchvision.transforms import functional as F, transforms as T
import os
import numpy as np
from PIL import Image

from config import BACKGROUNDS_PATH, RENDERS_PATH, MASKS_PATH, DATASET_PATH


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class AddRenders(nn.Module):
    def __init__(
        self,
        train_dataset: str,
        aug_dataset: str,
        amodal: bool = False,
        max_train_objs: int = 3,
        max_aug_objs: int = 5,
    ):
        super().__init__()
        self.amodal = amodal
        self.max_train_objs = max_train_objs
        self.max_aug_objs = max_aug_objs
        self.train_names = sorted(os.listdir(DATASET_PATH(train_dataset)))
        self.aug_names = sorted(os.listdir(DATASET_PATH(aug_dataset)))

        assert set(self.train_names).isdisjoint(
            self.aug_names
        ), "Train and augumentation datasets have object with same name (label)."

        self.aug_objects = []
        self.train_objects = []

        for object in self.train_names:
            self.train_objects.append(
                {
                    "label": object,
                    "renders": list(
                        sorted(RENDERS_PATH(train_dataset, object).iterdir())
                    ),
                    "masks": list(sorted(MASKS_PATH(train_dataset, object).iterdir())),
                }
            )
            assert len(self.train_objects[-1]["renders"]) == len(
                self.train_objects[-1]["masks"]
            ), f"Number of masks and renders dont match for object {object}."

        for object in self.aug_names:
            if RENDERS_PATH(aug_dataset, object).is_dir():
                self.aug_objects.append(
                    {
                        "label": object,
                        "renders": list(
                            sorted(RENDERS_PATH(aug_dataset, object).iterdir())
                        ),
                        "masks": list(
                            sorted(MASKS_PATH(aug_dataset, object).iterdir())
                        ),
                    }
                )
                assert len(self.aug_objects[-1]["renders"]) == len(
                    self.aug_objects[-1]["masks"]
                ), f"Number of masks and renders dont match for object {object}."

    def forward(self, render, mask, curr_obj_name):
        mask = mask > 0
        all_masks = mask[:, :, None]
        masks = mask[None, :, :]
        labels = [curr_obj_name]

        train_objects = [
            obj for obj in self.train_objects if obj["label"] != curr_obj_name
        ]

        rand_train_objs = np.random.randint(
            0, min(self.max_train_objs, len(train_objects)) + 1
        )
        rand_aug_objs = np.random.randint(
            0, min(self.max_aug_objs, len(self.aug_objects)) + 1
        )
        rand_train_objs = np.random.choice(
            train_objects, rand_train_objs, replace=False
        )
        rand_aug_objs = np.random.choice(self.aug_objects, rand_aug_objs, replace=False)

        for object in np.concatenate((rand_train_objs, rand_aug_objs)):
            render_num = np.random.randint(0, len(object["renders"]))
            new_render = np.array(Image.open(object["renders"][render_num]))
            new_mask = np.array(Image.open(object["masks"][render_num]))
            new_mask = new_mask > 0

            overlay_rates_train = np.array([
                    np.sum(train_mask & new_mask) / np.sum(train_mask)
                    for train_mask in masks])

            overlay_rate_all = np.sum(all_masks[:,:,0] & new_mask) / np.sum(new_mask)

            if object["label"] in self.train_names and overlay_rate_all > 0.5 and any(overlay_rates_train > 0.5):
                continue

            if (np.random.random() < 0.5 and overlay_rate_all < 0.5) or any(
                overlay_rates_train > 0.5
            ):
                render = np.where(all_masks, render, new_render)
                if not self.amodal:
                    new_mask = np.where(new_mask, np.logical_not(all_masks[:,:,0]), False)
            else:
                render = np.where(new_mask[:,:,None], new_render, render)
                if not self.amodal:
                    for i in range(len(masks)):
                        masks[i] = np.where(masks[i,:,:], np.logical_not(new_mask), False)

            if object["label"] in self.train_names:
                masks = np.append(masks, new_mask[None, :, :], axis=0)
                labels.append(object["label"])

            all_masks = new_mask[:,:,None] | all_masks

        return render, masks, all_masks, labels


class AddBackground(nn.Module):
    def forward(self, render, masks, all_masks, labels):
        backgrounds = os.listdir(BACKGROUNDS_PATH)
        background = np.random.choice(backgrounds)
        background = Image.open(BACKGROUNDS_PATH / background)
        render = np.array(render)

        background = background.resize(render.shape[:2][::-1])
        result = np.where(all_masks, render, background)
        return result, masks, labels


class ColorDistortion(nn.Module):
    def forward(self, image, masks, labels):
        image = Image.fromarray(image)

        result = T.ColorJitter(brightness=0.7, contrast=0.5, saturation=0.5, hue=0.13)(
            image
        )

        result = T.GaussianBlur(kernel_size=5, sigma=(0.2, 1.5))(result)

        result = T.ToTensor()(result)
        result = result + np.random.uniform(0, 1) * torch.randn_like(result) * 0.03

        random_values = np.random.normal(0, 0.15, result.shape)
        smoothing_iterations = np.random.randint(1, 10)
        for _ in range(smoothing_iterations):
            random_values = (
                random_values
                + np.roll(random_values, shift=(0, 1), axis=(0, 1))
                + np.roll(random_values, shift=(0, -1), axis=(0, 1))
                + np.roll(random_values, shift=(1, 0), axis=(0, 1))
                + np.roll(random_values, shift=(-1, 0), axis=(0, 1))
            ) / 5
        random_values = np.clip(random_values, -1, 1)

        result = result + random_values * np.random.uniform(0, 0.5)

        result = torch.clip(result, 0, 1)

        return result, masks, labels


class PILToTensor(nn.Module):
    def forward(self, image, masks, labels):
        image = F.pil_to_tensor(image)
        return image, masks, labels


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, image, masks, labels):
        image = F.convert_image_dtype(image, self.dtype)
        return image, masks, labels
