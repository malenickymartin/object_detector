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
        max_train_objs: int = 3,
        max_aug_objs: int = 5,
    ):
        super().__init__()
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
            # Test if renders and masks dirs exist was already done in dataset init
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
        only_object_mask = mask > 0
        labels = [curr_obj_name]

        data_objects = []
        for object in self.train_objects:
            if object["label"] != curr_obj_name:
                data_objects.append(object)

        rand_train_objs = np.random.randint(
            0, min(self.max_train_objs, len(data_objects)) + 1
        )
        rand_aug_objs = np.random.randint(
            0, min(self.max_aug_objs, len(self.aug_objects)) + 1
        )
        rand_train_objs = np.random.choice(data_objects, rand_train_objs, replace=False)
        rand_aug_objs = np.random.choice(self.aug_objects, rand_aug_objs, replace=False)

        for object in np.concatenate((rand_train_objs, rand_aug_objs)):
            render_num = np.random.randint(0, len(object["renders"]))
            new_render = np.array(Image.open(object["renders"][render_num]))
            new_mask = np.array(Image.open(object["masks"][render_num]))
            new_mask = new_mask > 0
            ovelay_rate = np.sum(np.logical_and(only_object_mask, new_mask)) / np.sum(
                only_object_mask
            )

            if object["label"] in self.train_names:
                masks = np.append(masks, new_mask[None, :, :], axis=0)
                labels.append(object["label"])
                only_object_mask = np.logical_or(only_object_mask, new_mask)

            new_mask = new_mask[:, :, None]

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

        result = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.05)(
            result
        )
        result = T.GaussianBlur(kernel_size=5, sigma=(0.2, 2))(result)

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
