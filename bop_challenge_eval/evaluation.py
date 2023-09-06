import torch
import torchvision.models.detection.mask_rcnn
import time
import numpy as np
import argparse

from detector import utils
from detector.run_detector_training import MaskRCNN
from detector.dataset import Dataset
from bop_challenge_eval.coco_eval import CocoEvaluator
from bop_challenge_eval.coco_utils import get_coco_api_from_dataset
from bop_challenge_eval.bop_dataset import BOP_Dataset
from config import (
    MODEL_PATH
)

from happypose.toolbox.datasets.datasets_cfg import make_scene_dataset


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 5, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir", type=str, nargs="?", default="train-three-objects_aug-ycbv"
    )
    parser.add_argument("train_dataset", type=str, nargs="?", default="three-objects")
    parser.add_argument("--experiment", "-e", type=str, default="test_3x1000_podruhe")
    args = parser.parse_args()


    if args.experiment == "":
        ckpt_path = MODEL_PATH(args.model_dir) / "model.ckpt"
    else:
        ckpt_path = MODEL_PATH(args.model_dir) / args.experiment / "model.ckpt"
    model = MaskRCNN.load_from_checkpoint(ckpt_path)


    my_scenes = [50, 51, 54, 56, 57, 59]
    my_labels = [15, 17, 18]
    ds_kwargs = dict(load_depth=False)
    dataset = make_scene_dataset("ycbv.bop19", **ds_kwargs)
    dataset = BOP_Dataset(dataset, my_scenes, my_labels)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        collate_fn=utils.collate_fn,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    evaluate(model, data_loader, device)