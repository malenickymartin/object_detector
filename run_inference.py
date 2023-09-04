import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import argparse
import torch
import os
from typing import Union, Tuple
from detector.train_detector import MaskRCNN

from config import MODEL_PATH, DATASET_PATH

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_bounding_boxes(
    img: Union[np.ndarray, torch.Tensor],
    model: torch.nn.Module,
    str_labels: list[str] = ["0", "1"],
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

    for i, box_1 in enumerate(best_boxes):
        for j, box_2 in enumerate(best_boxes):
            if i == j or best_labels[i] != best_labels[j]:
                continue
            box_area = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])
            overlay_box = [
                max(box_1[0], box_2[0]),
                max(box_1[1], box_2[1]),
                min(box_1[2], box_2[2]),
                min(box_1[3], box_2[3]),
            ]
            if overlay_box[0] > overlay_box[2] or overlay_box[1] > overlay_box[3]:
                continue
            overlay = (overlay_box[2] - overlay_box[0]) * (
                overlay_box[3] - overlay_box[1]
            )
            overlay_rate = overlay / box_area
            if overlay_rate > 0.9:
                print(
                    f"INFO: {best_labels[j]} with score of {best_scores[j]} was removed, as it's bouding box was most likely subset of another detected {best_labels[j]}."
                )
                best_labels.pop(j)
                best_scores.pop(j)
                best_boxes.pop(j)

    return best_boxes, best_labels, best_scores


def main(args: argparse.ArgumentParser) -> None:
    if args.experiment == "":
        ckpt_path = MODEL_PATH(args.model_dir) / "model.ckpt"
    else:
        ckpt_path = MODEL_PATH(args.model_dir) / args.experiment / "model.ckpt"

    model = MaskRCNN.load_from_checkpoint(ckpt_path)
    model.eval()

    str_labels = sorted(os.listdir(DATASET_PATH(args.train_dataset)))

    transform = transforms.Compose([transforms.PILToTensor()])

    for img_idx in range(1, args.test_img_num + 1):
        image_path = MODEL_PATH(args.model_dir) / f"test{img_idx}.png"
        result_path = MODEL_PATH(args.model_dir)

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

        print()
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
        all_masks.save(result_path / f"all_results_{img_idx}.png")
        print(f"All masks saved to {result_path / f'all_results_{img_idx}.png'}.")

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

        if len(best_boxes) == 0:
            print(f"No object with socre under {args.min_score} detected on the image.")
            continue

        for j, box_1 in enumerate(best_boxes):
            for i, box_2 in enumerate(best_boxes):
                if j == i or best_labels[j] != best_labels[i]:
                    continue
                box_area = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])
                overlay_box = [
                    max(box_1[0], box_2[0]),
                    max(box_1[1], box_2[1]),
                    min(box_1[2], box_2[2]),
                    min(box_1[3], box_2[3]),
                ]
                if overlay_box[0] > overlay_box[2] or overlay_box[1] > overlay_box[3]:
                    continue
                overlay = (overlay_box[2] - overlay_box[0]) * (
                    overlay_box[3] - overlay_box[1]
                )
                overlay_rate = overlay / box_area
                if overlay_rate > 0.9:
                    print(
                        f"INFO: {best_labels[i]} with score of {best_scores[i]} was removed, as it's bouding box was most likely subset of another detected {best_labels[i]}. Check all_results_{img_idx}.png if you want to see the removed bounding box."
                    )
                    best_labels.pop(i)
                    best_masks.pop(i)
                    best_scores.pop(i)
                    best_boxes.pop(i)

        best_mask = np.sum(best_masks, axis=0)
        best_mask = (best_mask * 255).astype(np.uint8)
        best_mask = Image.fromarray(best_mask).convert("RGB")
        for i in range(len(best_boxes)):
            rect = ImageDraw.Draw(best_mask)
            rect.rectangle(best_boxes[i], fill=None, outline="red")
            lab = ImageDraw.Draw(best_mask)
            lab.text(
                (best_boxes[i][0] + 3, best_boxes[i][1] + 2),
                best_labels[i],
                (0, 0, 255),
            )
            sc = ImageDraw.Draw(best_mask)
            sc.text(
                (best_boxes[i][0] + 3, best_boxes[i][1] + 12),
                str(best_scores[i]),
                (0, 0, 255),
            )
        best_mask.save(result_path / f"best_result_{img_idx}.png")
        print(f"Best mask saved to: {result_path / f'best_result_{img_idx}.png'}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir", type=str, nargs="?", default="train-rc-car_aug-ycbv"
    )
    parser.add_argument("train_dataset", type=str, nargs="?", default="rc-car")
    parser.add_argument("min_score", type=float, nargs="?", default=0.85)
    parser.add_argument("test_img_num", type=float, nargs="?", default=8)
    parser.add_argument("--experiment", "-e", type=str, default="test_5000")
    args = parser.parse_args()

    main(args)
