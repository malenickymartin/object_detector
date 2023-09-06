import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

from happypose.toolbox.datasets.scene_dataset import SceneDataset


class BOP_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset: SceneDataset, my_scenes, my_labels):
        self.im_to_tensor = transforms.ToTensor()
        self.imgs = []
        self.masks = []
        self.labels = []
        self.bboxes = []
        print("INFO: Making BOP Datset")
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            if data.infos.scene_id not in my_scenes:
                continue
            self.imgs.append(data.rgb)
            self.labels.append([])
            self.bboxes.append([])
            self.masks.append([])
            for j in range(len(data.object_datas)):
                label = int(data.object_datas[j].label.split("_")[1])
                if label not in my_labels:
                    continue

                self.masks[-1].append(
                    np.where(
                        data.segmentation == j+1,
                        np.ones(data.segmentation.shape, dtype=np.uint8),
                        np.zeros(data.segmentation.shape, dtype=np.uint8),
                    )
                )
                
                label = int(data.object_datas[j].label.split("_")[1])
                self.labels[-1].append(label)

                self.bboxes[-1].append(data.object_datas[j].bbox_amodal)

    def __getitem__(self, idx):
        # load images and masks

        img = self.imgs[idx]
        boxes = self.bboxes[idx]
        masks = self.masks[idx]
        for mask in masks:
            mask = np.array(mask)
        masks = np.array(masks)
        labels = self.labels[idx]

        num_objs = len(labels)

        img = self.im_to_tensor(img)
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
