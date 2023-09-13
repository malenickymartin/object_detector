# Synthetic Data Based Object Detector 

This project is designed to help you create a 2D object detector for known objects without the need to manually label a large dataset. We achieve this by leveraging synthetic image generation using Panda 3D and implementing a 2D detector using PyTorch. Additionally, we provide various useful tools to streamline your workflow. 

## Installation
Before proceeding with the installation, make sure you have Happypose installed. If you haven't already, you can follow the installation instructions [here](https://github.com/agimus-project/happypose/tree/dev).
Once Happypose is successfully installed, you can continue with the following commands:
```
conda activate happypose
git clone git@github.com:malenickymartin/object_detector.git
cd object_detector
pip install -r requirements.txt
```


## Example

Let's walk through an example that covers every essential step, from rendering datasets and training the model to running inference. If you ever get lost or need to adjust file paths, see the `config.py` file.

### Step 1: Set Up Your Datasets

1. First, you'll need to create two directories within the `datasets` directory:
   - The first directory should contain objects you want the detector to be trained on (eg. models from [YCBV](https://www.ycbbenchmarks.com/object-models/)).
   - The second directory (optional) can contain objects used for augmentations. These augmentation objects will be added to training images and may partially cover your training objects, allowing the detector to detect partially covered objects. If you don't need augmentations, you can skip creating the second directory.
You can find example directories, meshes, and textures in `datasets/example-train` and `datasets/example-aug`.
2. Populate the `backgrounds` directory with various images (eg. dataset from [VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)) that will serve as backgrounds for the renders in the next step. The number of background images should roughly match (number of epochs) multiplied by (renders per epoch), so that each background is unique during training.

### Step 2: Generate Synthetic Images

Run the following command to generate 100 renders without a background and 100 masks (for better results, consider using a few hundred renders). The default image size is set to 640x480.

```bash
python3 -m renderer.render_model example-train 100
```

If you've created an augmentation dataset, run the following command to generate renders for augumentation objects:

```bash
python3 -m renderer.render_model example-aug 100
```


### Step 3: Train Your Model

It's time to train your object detection model. Execute the following command to train the detector for 10 epochs with a batch size of 4. The target masks will be amodal, and the model will be saved in `models/train-example-train_aug-example-aug/example-experiment`. If you didn't create an augmentation dataset, leave the second argument empty.

```bash
python3 -m detector.run_detector_training example-train --aug_dataset=example-aug --batch-size=4 --num_epochs=10 --amodal --experiment=example_experiment
```

### Step 4: Test Your Detector

To evaluate how well your detector has trained, you can create a random image and test it with the following commands:

```bash
python3 -m generate_random_synthetic_image example-train mustard --aug_dataset=example-aug
python3 -m run_inference example-train --aug_dataset=example-aug --experiment=example_experiment
```

After running these commands, you'll find test images, their ground truth masks, and the results (masks predicted by the detector) in the `models/train-example-train_aug-example-aug` directory.

