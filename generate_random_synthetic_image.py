from detector import transforms as T
from torchvision.transforms import transforms as TorchTransforms
import os
import numpy as np
import argparse
from PIL import Image
from config import (
    RENDERS_PATH,
    MASKS_PATH,
    MODEL_PATH
)

def get_transform(object_name: str, augs_dataset: str, amodal: bool):
    transforms = []
    transforms.append(T.AddRenders(object_name, augs_dataset, amodal))
    transforms.append(T.AddBackground())
    transforms.append(T.ColorDistortion())
    return T.Compose(transforms)

def main(args):
    dataset_name = args.train_dataset
    augs_dataset = args.aug_dataset
    object_name = args.object_name
    model_dir = args.model_dir
    amodal = args.amodal

    for i in range(1,args.num_images+1):
        result_image_name = "test"+str(i)+".png"

        transforms = get_transform(dataset_name, augs_dataset, amodal)
        images_list = os.listdir(RENDERS_PATH(dataset_name, object_name))
        image_num = np.random.randint(0, len(images_list))
        print(f"Image number {image_num}")
        img = np.array(Image.open(RENDERS_PATH(dataset_name, object_name) / f"render{image_num}.png").convert("RGB"))
        mask = np.array(Image.open(MASKS_PATH(dataset_name, object_name) / f"mask{image_num}.png"))

        img, masks, _ = transforms(img, mask, object_name)

        transform = TorchTransforms.ToPILImage()
        img = transform(img)

        img.save(MODEL_PATH(model_dir) / result_image_name)

        for j, mask in enumerate(masks):
            Image.fromarray(mask).save(MODEL_PATH(model_dir) / f"mask-{i}_{j}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dataset", type=str, nargs="?", default="ycbv")
    parser.add_argument("object_name", type=str, nargs="?", default="sugar")
    parser.add_argument("num_images", type=int, nargs="?", default=5)
    parser.add_argument("--aug_dataset", type=str, nargs="?", default=None)
    parser.add_argument("--amodal", "-a", action="store_true")
    args = parser.parse_args()

    if args.aug_dataset == None:
        args.aug_dataset = ".empty"
        args.model_dir = f"train-{args.train_dataset}"
    else:
        args.model_dir = f"train-{args.train_dataset}_aug-{args.aug_dataset}"

    main(args)
