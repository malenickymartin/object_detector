import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import argparse
import torch

from config import (
    MODEL_PATH,
)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_bb(img, model, min_score, str_labels, device):

    model.to(device)

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
        labels[i] = str_labels[labels[i]-1]
        scores[i] = round(scores[i], 3)
    
    best_boxes = []
    best_labels = []
    for i, score in enumerate(scores):
        if score >= min_score:
            best_boxes.append(boxes[i])
            best_labels.append(labels[i])

    # Enlarge boxes by 5 %
    for box in best_boxes:
        dw = 5*(box[2]-box[0])/100
        dh = 5*(box[3]-box[1])/100
        box[0] = np.clip(box[0]-dw, 0, img.shape[-1]+1)
        box[1] = np.clip(box[1]-dh, 0, img.shape[-2]+1)
        box[2] = np.clip(box[2]+dw, 0, img.shape[-1]-1)
        box[3] = np.clip(box[3]+dh, 0, img.shape[-2]-1)

    return best_boxes, best_labels

def main(object_name, image_path, min_score, best_result_path, result_path, str_labels):

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

    labels = output["labels"].cpu().detach().numpy().tolist()
    scores = output["scores"].cpu().detach().numpy()
    masks = output["masks"].cpu().detach().numpy()
    boxes = output["boxes"].cpu().detach().numpy()

    print(f"The number of detected objects is {len(masks)}.")
    print(f"The best match has score of {np.max(scores)}.")
    print(f"Other scores are: {scores}.")

    # Enlarge boxes by 5 %
    for box in boxes:
        dw = 5*(box[2]-box[0])/100
        dh = 5*(box[3]-box[1])/100
        box[0] = np.clip(box[0]-dw, 0, img.shape[-1]+1)
        box[1] = np.clip(box[1]-dh, 0, img.shape[-2]+1)
        box[2] = np.clip(box[2]+dw, 0, img.shape[-1]-1)
        box[3] = np.clip(box[3]+dh, 0, img.shape[-2]-1)

    for i in range(len(labels)):
        labels[i] = str_labels[labels[i]-1]
        scores[i] = round(scores[i], 3)
    
    best_boxes = []
    best_masks = []
    best_labels = []
    best_scores = []
    for i, score in enumerate(scores):
        if score >= min_score:
            best_boxes.append(boxes[i])
            best_masks.append(masks[i][0])
            best_labels.append(labels[i])
            best_scores.append(scores[i])
    best_mask = np.sum(best_masks, axis=0)

    #### LOCAL SAVE ####

    assert len(best_boxes) > 0, "No object detected on the image."
    best_mask = (best_mask*255).astype(np.uint8)
    best_mask_filter = best_mask < 255*0
    best_mask[best_mask_filter] = 0
    best_mask = Image.fromarray(best_mask).convert('RGB')
    for i in range(len(best_boxes)):
        rect = ImageDraw.Draw(best_mask)
        rect.rectangle(best_boxes[i], fill = None, outline ="red")
        lab = ImageDraw.Draw(best_mask)
        lab.text((best_boxes[i][0]+3, best_boxes[i][1]+2), best_labels[i], (0, 0, 255))
        sc = ImageDraw.Draw(best_mask)
        sc.text((best_boxes[i][0]+3, best_boxes[i][1]+12), str(best_scores[i]), (0, 0, 255))
    best_mask.save(best_result_path)
    print(f"Best mask saved to: {best_result_path}.")
    
    all_masks = np.sum(masks, axis=0)[0]
    all_masks = (all_masks*255).astype(np.uint8)
    all_masks = Image.fromarray(all_masks).convert('RGB')
    for i in range(len(boxes)):
        rect = ImageDraw.Draw(all_masks)
        rect.rectangle(boxes[i], fill = None, outline ="red")
        lab = ImageDraw.Draw(all_masks)
        lab.text((boxes[i][0]+3, boxes[i][1]+2), labels[i], (0, 0, 255))
        sc = ImageDraw.Draw(all_masks)
        sc.text((boxes[i][0]+3, boxes[i][1]+12), str(scores[i]), (0, 0, 255))
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

    return best_boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder",type=str, nargs='?', default="rc-car")
    parser.add_argument("model_lables",type=str, nargs='?', default="rc-car")
    parser.add_argument("min_score",type=float, nargs='?', default=0.85)
    parser.add_argument("test_num",type=float, nargs='?', default=7)
    args = parser.parse_args()

    for i in range(1,7):
        args.test_num = i
        image_path = MODEL_PATH(args.model_folder) / f"test{args.test_num}.png"
        best_result_path = MODEL_PATH(args.model_folder) / f"result_best_{args.test_num}.png"
        result_path = MODEL_PATH(args.model_folder) / f"result_{args.test_num}.png"

        labels = args.model_lables.split(",")

        main(args.model_folder, image_path, args.min_score, best_result_path, result_path, labels)