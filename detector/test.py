import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train_dataset", type=str, nargs="?", default="three-objects")
parser.add_argument("aug_dataset", type=str, nargs="?", default="ycbv")
parser.add_argument("--batch_size", "-b", type=int, default=8)
parser.add_argument("--img_per_obj", "-i", type=int, default=1000)
parser.add_argument("--amodal", "-a", action="store_true")
parser.add_argument("--experiment", "-e", type=str, default="test_3x5000_2")

args = parser.parse_args()

print(args)