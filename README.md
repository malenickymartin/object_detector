# Synthetic data based object detector 

This repository contains code for generating synthetic images using Panda 3D and 2D detector using torch as well as other usefull tools. The code is dependant on Happypose, pleas follow installation [here](https://github.com/agimus-project/happypose/tree/dev). Aim of this project is to create a 2D detector of seen objects, without need to create large labeled dataset by hand.

## Example

This example will show you every necessery step from rendering datasets and training model to running inference.

First you need to create two directories in "datasets" directory. One containing objects you want your detector to detect, second containing objects 
used for augumentations.
You can see example directories, meshes and textures in "datasets/example-train" and "datasets/example-aug".
The following command will create 25 renders without background and 25 masks, with default image size of 640x480.
```
python3 -m renderer.render_model example-train 25
```
Now make renders for the augumentation object. If you do not want to use other objects as augumentations, skip this command.
```
python3 -m renderer.render_model example-aug 25
```
Next step is to start training the model. With the following command, the detector will train for 10 epochs with batch size 4 and the target 
masks will be amodal. The model will be saved in "models/train-example-train_aug-example-aug/example-experiment". If you did not create 
augumentation dataset in previous step, leave the second argument empty.
```
python3 -m detector.run_detector_training example-train --aug_dataset=example-aug -b 4 --num_epochs=10 -a -e example_experiment
```

If you want to see how well the detector has trained you can use the following commands to create random image and test it:
```
python3 -m generate_random_synthetic_image example-train mustard --aug_dataset=example-aug
python3 -m run_inference example-train --aug_dataset=example-aug -e example_experiment
```
After these two commands, the direcotry "models/train-example-train_aug-example-aug" will contain 5 test images, their ground true masks and results 
(masks predicted by detector).