# Synthetic Data Based Object Detector 

This project is designed to help you create a 2D object detector for known objects without the need to manually label a large datasets. We achieve this by leveraging synthetic image generation using Blenderproc and implementing a 2D detector using PyTorch. Additionally, we provide various useful tools to streamline your workflow. 

## Installation

To get started, follow these installation steps:
```
git clone git@github.com:malenickymartin/object_detector.git
cd object_detector
pip install -r requirements.txt
```

## Example

Let's walk through an example that covers every essential step, from rendering datasets and training the model to running inference. If you ever get lost or need to adjust file paths, see the `config.py` file.

### Step 1: Set Up Your Datasets

1. First, you'll need to create two directories within the `datasets` directory. If you don't yet have any data or just want to try out the scipts, you can use example directories already prepared and skip this substep and continue to "2. Populate the `cctextures` direcotry". The direcotries names are arbitrary, but should represent your datasets:
   - The first directory should contain subdirecotry called `meshes`, that has objects you want the detector to be trained on inside (eg. models from [YCBV](https://www.ycbbenchmarks.com/object-models/)). Object names should adhere to the following format: obj_"obj-id"."ext". The "obj-id" is a 6-digit number, commencing from 000001 and incrementing by one for each additional object. The "ext" is one of following extensions: `bj, ply, dae, stl, fbx and glb`. The dataset direcotry can optionally include file named `labels.txt`, that has one line for each object the line is in following format: `obj-id:label`, where the "obj-id" is the same as in meshes folder and "label" is arbitrary name, that you want to call the object.
   - The second directory (optional) has the same structure as the first direcotry, but contains objects used for augmentations. These augmentation objects will be added to training renders and may partially cover your training objects. This allows the detector to recognize partially obscured objects. If you don't need augmentations, you can skip creating the second directory.

   You can find example directories, meshes, labels and textures in `datasets/example-train` and `datasets/example-aug`.
2. Populate the `cctextures` directory. To do so, use the following command in `object_detection` directory.
   ```
   blenderproc download cc_textures cctextures
   ```
   This will download all the available textures from Blenderproc. You can terminate the program prematurely, if you are only trying the example, but for real training it is recommended to download all textures.
### Step 2: Generate Synthetic Images

Run the following command to generate 100 renders (4 scenes, each containing 25 images) without a background and 100 masks (for better results, consider using a few hundred or thousand renders). The default image size is set to 640x480. Run the following command to generate renders with augumentations. Note that the first time you use this command blender will be automatically installed, if it was not already.

```
blenderproc run renderer/render_scenes.py example-train example-aug --num-scenes=4 --imgs-per-scene=25
```


### Step 3: Train Your Model

It's time to train your object detection model. Execute the following command to train the detector for 10 epochs with a batch size of 4. The model will be saved in `models/train-example-train/example-experiment`.

```
python3 -m detector.run_detector_training example-train --batch-size=4 --num-epochs=10 --experiment=example_experiment
```

### Step 4: Test Your Detector

To evaluate how well your detector has trained, you can create a random augumented images and test them with the following commands:

```
python3 -m generate_random_synthetic_image example-train 10
python3 -m run_inference example-train --experiment=example_experiment
```

After running these commands, you'll find test images, their ground truth masks, and the results (masks predicted by the detector) in the `models/train-example-train` directory.

