import argparse
import math

from detector.engine import train_one_epoch, evaluate
from detector import utils
from detector import transforms as T
from detector.dataset import Dataset

from config import (
    MODEL_PATH
)

import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

torchvision.disable_beta_transforms_warning()

backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
backbone.out_channels = 1280
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

TRAIN_TO_TEST_RATIO = 0.7

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def get_transform(object_names):
    transforms = []
    transforms.append(T.AddRenders(object_names))
    transforms.append(T.AddBackground())
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)

def main(object_names, model_folder):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = len(object_names) + 1
    # use our dataset and defined transformations
    dataset = Dataset(object_names, get_transform(object_names))
    dataset_test = Dataset(object_names, get_transform(object_names))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:math.floor(TRAIN_TO_TEST_RATIO*len(indices))])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-math.ceil((1-TRAIN_TO_TEST_RATIO)*len(indices))+1:])
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=8,
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=16, shuffle=False, num_workers=8,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # train it for 10 epochs
    num_epochs = 10

    MODEL_PATH(model_folder).mkdir(exist_ok=True)
    with open(MODEL_PATH(model_folder) / 'log.txt', 'w') as f:
        f.write("This file contains losses for each epoch training and validation.\n\n")

    min_eval_loss = math.inf

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, model_folder, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        eval_loss = evaluate(model, data_loader_test, model_folder, device=device)
        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            with open(MODEL_PATH(model_folder) / "model.pkl", "wb") as f:
                torch.save(model, f)
            print(f"Model saved in epoch {epoch} to file: {MODEL_PATH(model_folder)}/model.pkl, with loss: {min_eval_loss}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("object_names",type=str, nargs='?', default="mustard")
    parser.add_argument("model_folder",type=str, nargs='?', default="mustard")
    args = parser.parse_args()

    object_names = args.object_names.split(",")
    main(object_names, args.model_folder)
