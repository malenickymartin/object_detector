# Synthetic data based object detector 

This repository contains code for generating synthetic images using Panda 3D and 2D detector using torch as well as other usefull tools. The code is dependant on Happypose, pleas follow installation [here](https://github.com/agimus-project/happypose/tree/dev). Aim of this project is to create a 2D detector of seen objects, without need to create large labeled dataset by hand.

## Example

This example will show complete process from rendering datasets and training model to running inference.

First you need to create two directories in "datasets" directory. One containing objects you want your detector to detect, second containing objects 
used in augumentations. Note that even if you do not want to you augumentations, you must create the directory, but you can leav it empty.
You can see example directories, meshes and textures in "datasets/example-train" and "datasets/example-aug".

The following command will create 25 renders and 25 masks, with default image size.
```
python3 -m renderer.render_model example-train 25
```
Now make renders for the augumentation object.
```
python3 -m renderer.render_model example-aug 25
```
Next step is to start training the model. With the following command, the detector will train for 10 epochs with batch size 4 and the target 
masks will be amodal. The model will be saved in "models/train-example-train_aug-example-aug/example-experiment".
```
python3 -m detector.run_detector_training example-train example-aug -b 4 --num_epochs=10 -a -e example_experiment
```

If you want to see how well the detector has trained you can use the following commands to create random image and test it:
```
python3 -m generate_random_synthetic_image
```