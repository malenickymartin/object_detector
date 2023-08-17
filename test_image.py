import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import argparse

from config import (
    MODEL_PATH,
    OBJECT_PATH
)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def main(object_name, image_path, min_score, best_result_path, result_path):

    with open(MODEL_PATH(object_name) / "model.pkl", "rb") as f:
        model = torch.load(f, device)

    model.to(device)
        
    transform = transforms.Compose([transforms.PILToTensor()])
    with open(image_path, "rb") as f:
        img = transform(Image.open(f))

        new_shape = [1] + list(img.shape)
    img = torch.reshape(img, new_shape)/255
    img = img.to(device)
    output = model(img)[0]

    scores = output["scores"].cpu().detach().numpy()
    masks = output["masks"].cpu().detach().numpy()
    boxes = output["boxes"].cpu().detach().numpy()

    print(f"{len(masks)} objects detected.")
    print(f"The best match has score of {np.max(scores)}.")

    # Enlarge boxes by 5 %
    for box in boxes:
        w = 5*(box[2]-box[0])/100
        h = 5*(box[3]-box[1])/100
        box[0] = np.clip(box[0]-w, 0, img.shape[-1]+1)
        box[1] = np.clip(box[1]-h, 0, img.shape[-2]+1)
        box[2] = np.clip(box[2]+w, 0, img.shape[-1]-1)
        box[3] = np.clip(box[3]+h, 0, img.shape[-2]-1)

    best_box = None
    if len(scores >= 1) and np.max(scores) > min_score:
        best_mask = masks[np.argmax(scores)][0]
        best_box = boxes[np.argmax(scores)]

    #### LOCAL SAVE ####
    if np.any(best_box) != None:
        best_mask = (best_mask*255).astype(np.uint8)
        best_mask_filter = best_mask < 255*0
        best_mask[best_mask_filter] = 0
        best_mask = Image.fromarray(best_mask)
        rect = ImageDraw.Draw(best_mask)
        rect.rectangle(best_box, fill = None, outline ="red")
        best_mask.save(best_result_path)
        print(f"Best mask saved to: {best_result_path}.")
    
    all_masks = np.sum(masks, axis=0)[0]
    all_masks = (all_masks*255).astype(np.uint8)
    all_masks = Image.fromarray(all_masks)
    for box in boxes:
        rect = ImageDraw.Draw(all_masks)
        rect.rectangle(box, fill = None, outline ="red")
    all_masks.save(result_path)
    print(f"All masks saved to {result_path}.")

    ##### MATPLOTLIB ####
    """
    fig, ax = plt.subplots()
    plt.imshow(all_masks[0], cmap='gray')

    for i in range(len(boxes)):
        box = boxes[i]
        w = box[2]-box[0]
        h = box[3]-box[1]
        rect = patches.Rectangle((box[0], box[1]), w, h, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    """

    return best_box

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("object_name",type=str, nargs='?', default="rc-car")
    parser.add_argument("min_score",type=float, nargs='?', default=0.8)
    args = parser.parse_args()

    image_path = OBJECT_PATH(args.object_name) / "test3.png"
    best_result_path = OBJECT_PATH(args.object_name) / "result_best_3.png"
    result_path = OBJECT_PATH(args.object_name) / "result_3.png"

    main(args.object_name, image_path, args.min_score, best_result_path, result_path)