import os
import sys
import math
import argparse
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from config import DATASET_PATH, MODEL_PATH

from detector import transforms as T
from detector.dataset import Dataset
from detector import utils

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_transform(train_dataset: str, aug_dataset: str) -> T.Compose:
    transforms = []
    transforms.append(T.AddRenders(train_dataset, aug_dataset))
    transforms.append(T.AddBackground())
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes: int):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def get_dataloaders(args: argparse.ArgumentParser, batch_size: int):
    dataset = Dataset(
        args.train_dataset, get_transform(args.train_dataset, args.aug_dataset)
    )
    dataset_test = Dataset(
        args.train_dataset, get_transform(args.train_dataset, args.aug_dataset)
    )
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(
        dataset, indices[: math.floor(TRAIN_TO_TEST_RATIO * len(indices))]
    )
    dataset_test = torch.utils.data.Subset(
        dataset_test,
        indices[-math.ceil((1 - TRAIN_TO_TEST_RATIO) * len(indices)) + 1 :],
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=utils.collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=utils.collate_fn,
    )

    return data_loader, data_loader_test


TRAIN_TO_TEST_RATIO = 0.7


class MaskRCNN(pl.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = get_model_instance_segmentation(num_classes)
        self.automatic_optimization = True
        self.save_hyperparameters()

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        self.model.train()
        images, targets = batch
        with torch.cuda.amp.autocast(enabled=False):
            loss_dict = self.model(images, targets)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        self.log(
            "train_loss",
            loss_value,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            batch_size=len(images),
            sync_dist=True,
        )

        return losses_reduced

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        self.model.eval()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        self.log(
            "val_loss",
            loss_value,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            batch_size=len(images),
            sync_dist=True,
        )
        return losses_reduced

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


def main(args: argparse.ArgumentParser) -> None:
    output_dir = MODEL_PATH(args.output_dir_name) / args.experiment
    output_dir.mkdir(exist_ok=True, parents=True)
    num_epochs = 10
    batch_size = 16

    object_names = sorted(os.listdir(DATASET_PATH(args.train_dataset)))
    num_classes = len(object_names) + 1

    data_loader, data_loader_test = get_dataloaders(args, batch_size)

    logger = CSVLogger(output_dir, name="", version="")

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir, save_top_k=1, monitor="val_loss"
    )

    model = MaskRCNN(num_classes)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=num_epochs,
        default_root_dir=output_dir,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, data_loader, data_loader_test)
    Path(checkpoint_callback.best_model_path).rename(output_dir / "model.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dataset", type=str, nargs="?", default="bbq")
    parser.add_argument("aug_dataset", type=str, nargs="?", default="ycbv")
    parser.add_argument("--experiment", "-e", type=str, default="test_100")

    args = parser.parse_args()
    train_models = os.listdir(DATASET_PATH(args.train_dataset))
    if len(train_models) == 0:
        args.output_dir_name = f"train-{train_models[0]}_aug-{args.aug_dataset}"
    else:
        args.output_dir_name = f"train-{args.train_dataset}_aug-{args.aug_dataset}"

    main(args)
