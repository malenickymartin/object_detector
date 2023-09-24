import numpy as np
import argparse
import albumentations as A
from PIL import Image
from tqdm import tqdm
from config import (
    DATASET_PATH,
)

transforms = A.Compose(
            [
                A.Flip(p=0.5),
                A.Blur(blur_limit=(3,5), p=0.25),
                A.CLAHE(clip_limit=(1,2), p=0.125),
                A.Downscale(scale_min=0.5, scale_max=0.99, interpolation=1, p=0.125),
                A.ISONoise(color_shift=(0,0.2), intensity=(0.1,1), p=0.25),
                A.RandomBrightnessContrast(brightness_limit=(-0.1,0.25), contrast_limit=(-0.1,0.25), p=0.5),
            ],
            p=0.95
        )

def main(args):

    for i in tqdm(range(1,args.num_images+1)):
        rand_scene = np.random.choice(list((DATASET_PATH(args.dataset) / "train_pbr").iterdir()))
        image_path = np.random.choice(list((rand_scene / "rgb").iterdir()))
        img = Image.open(image_path)
        transformed = transforms(image=np.array(img))
        img = transformed["image"]

        img = Image.fromarray(img)
        img.save(DATASET_PATH(args.dataset) / f"synt_im_{i}.png")
    print(f"All images were successfully saved to: {DATASET_PATH(args.dataset)}/synt_im_x.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, nargs="?", default="ycbv")
    parser.add_argument("num-images", type=int, nargs="?", default=1)
    args = parser.parse_args()

    main(args)
