import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import argparse
import torch
import os
from typing import Union, Tuple
from pathlib import Path
from detector.train_detector import MaskRCNN

from config import MODEL_PATH, DATASET_PATH

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_bb(
    img: Union[np.ndarray, torch.Tensor],
    model: torch.nn.Module,
    str_labels: list[str],
    device: torch.device = torch.device("cpu"),
    min_score: float = 0.75,
) -> Tuple[list[list], list[str], list[float]]:
    model.to(device)
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    img = img.to(device)

    output = model(img)[0]

    labels = output["labels"].cpu().detach().numpy().tolist()
    scores = output["scores"].cpu().detach().numpy()
    boxes = output["boxes"].cpu().detach().numpy()

    print(f"The number of detected objects is {len(labels)}.")
    print(f"The best match has score of {np.max(scores)}.")
    print(f"Other scores are: {scores}.")

    for i in range(len(labels)):
        labels[i] = str_labels[labels[i] - 1]
        scores[i] = round(scores[i], 3)

    best_boxes = []
    best_labels = []
    best_scores = []
    for i, score in enumerate(scores):
        if score >= min_score:
            best_boxes.append(boxes[i])
            best_labels.append(labels[i])
            best_scores.append(scores[i].item())

    # Enlarge boxes by 5 %
    for box in best_boxes:
        dw = 5 * (box[2] - box[0]) / 100
        dh = 5 * (box[3] - box[1]) / 100
        box[0] = np.clip(box[0] - dw, 0, img.shape[-1] + 1)
        box[1] = np.clip(box[1] - dh, 0, img.shape[-2] + 1)
        box[2] = np.clip(box[2] + dw, 0, img.shape[-1] - 1)
        box[3] = np.clip(box[3] + dh, 0, img.shape[-2] - 1)

    return best_boxes, best_labels, best_scores


def main(
    args: argparse.ArgumentParser, image_path: Union[Path, str], result_path: Path
) -> None:
    if args.experiment == "":
        ckpt_path = MODEL_PATH(args.model_dir) / "model.ckpt"
    else:
        ckpt_path = MODEL_PATH(args.model_dir) / args.experiment / "model.ckpt"

    model = MaskRCNN.load_from_checkpoint(ckpt_path)
    model.eval()

    str_labels = sorted(os.listdir(DATASET_PATH(args.train_dataset)))

    transform = transforms.Compose([transforms.PILToTensor()])
    with open(image_path, "rb") as f:
        img = transform(Image.open(f))

        new_shape = [1] + list(img.shape)
    img = torch.reshape(img, new_shape) / 255
    img = img.to(device)
    output = model(img)[0]

    labels = output["labels"].cpu().detach().numpy().tolist()
    scores = output["scores"].cpu().detach().numpy()
    masks = output["masks"].cpu().detach().numpy()
    boxes = output["boxes"].cpu().detach().numpy()

    print(f"The number of detected objects is {len(masks)}.")
    print(f"The best match has score of {np.max(scores)}.")
    print(f"Other scores are: {scores}.")

    # Enlarge boxes by 5 %
    for box in boxes:
        dw = 5 * (box[2] - box[0]) / 100
        dh = 5 * (box[3] - box[1]) / 100
        box[0] = np.clip(box[0] - dw, 0, img.shape[-1] + 1)
        box[1] = np.clip(box[1] - dh, 0, img.shape[-2] + 1)
        box[2] = np.clip(box[2] + dw, 0, img.shape[-1] - 1)
        box[3] = np.clip(box[3] + dh, 0, img.shape[-2] - 1)

    for i in range(len(labels)):
        labels[i] = str_labels[labels[i] - 1]
        scores[i] = round(scores[i], 3)

    # ALL SAVE
    all_masks = np.sum(masks, axis=0)[0]
    all_masks = (all_masks * 255).astype(np.uint8)
    all_masks = Image.fromarray(all_masks).convert("RGB")
    for i in range(len(boxes)):
        rect = ImageDraw.Draw(all_masks)
        rect.rectangle(boxes[i], fill=None, outline="red")
        lab = ImageDraw.Draw(all_masks)
        lab.text((boxes[i][0] + 3, boxes[i][1] + 2), labels[i], (0, 0, 255))
        sc = ImageDraw.Draw(all_masks)
        sc.text((boxes[i][0] + 3, boxes[i][1] + 12), str(scores[i]), (0, 0, 255))
    all_masks.save(result_path / f"all_results_{args.test_img_num}.png")
    print(f"All masks saved to {result_path / f'all_results_{args.test_img_num}.png'}.")

    # BEST SAVE
    best_boxes = []
    best_masks = []
    best_labels = []
    best_scores = []
    for i, score in enumerate(scores):
        if score >= args.min_score:
            best_boxes.append(boxes[i])
            best_masks.append(masks[i][0])
            best_labels.append(labels[i])
            best_scores.append(scores[i])
    best_mask = np.sum(best_masks, axis=0)

    assert len(best_boxes) > 0, "No object detected on the image."
    best_mask = (best_mask * 255).astype(np.uint8)
    best_mask_filter = best_mask < 255 * 0
    best_mask[best_mask_filter] = 0
    best_mask = Image.fromarray(best_mask).convert("RGB")
    for i in range(len(best_boxes)):
        rect = ImageDraw.Draw(best_mask)
        rect.rectangle(best_boxes[i], fill=None, outline="red")
        lab = ImageDraw.Draw(best_mask)
        lab.text(
            (best_boxes[i][0] + 3, best_boxes[i][1] + 2), best_labels[i], (0, 0, 255)
        )
        sc = ImageDraw.Draw(best_mask)
        sc.text(
            (best_boxes[i][0] + 3, best_boxes[i][1] + 12),
            str(best_scores[i]),
            (0, 0, 255),
        )
    best_mask.save(result_path / f"best_result_{args.test_img_num}.png")
    print(
        f"Best mask saved to: {result_path / f'best_result_{args.test_img_num}.png'}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir", type=str, nargs="?", default="train-rc-car_aug-ycbv"
    )
    parser.add_argument("train_dataset", type=str, nargs="?", default="train")
    parser.add_argument("min_score", type=float, nargs="?", default=0.85)
    parser.add_argument("test_img_num", type=float, nargs="?", default=7)
    parser.add_argument("--experiment", "-e", type=str, default="test_0")
    args = parser.parse_args()

    for i in range(1, args.test_img_num+1):
        args.test_img_num = i
        image_path = MODEL_PATH(args.model_dir) / f"test{args.test_img_num}.png"
        result_path = MODEL_PATH(args.model_dir)

        main(args, image_path, result_path)
