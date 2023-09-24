import os
from pathlib import Path

# This is project directory. It is by default set to this config file path.
PROJECT_PATH = Path(os.path.realpath(__file__)).parent

# This directory contains subdirectory for each used dataset. More in "DATASET_PATH".
DATASETS_PATH = PROJECT_PATH / "datasets"

# This directory contains subdirectories with images, that will used as background texture in renders. The images have specific naming convention, 
# so it is recommended to download the textures as mentioned in README.
TEXTURES_PATH = PROJECT_PATH / "cctextures"

# This directory contains subdirectories for each dataset used for training. 
# All subdirectories will be created automatically by "run_detector_training.py". More in "MODEL_PATH"
MODELS_PATH = PROJECT_PATH / "models"

# This directory contains meshes and labeled training images.
def DATASET_PATH(dataset_name : str) -> Path:
    return DATASETS_PATH / dataset_name

# This directory contains results of training. After the training is finished, there will be "model.ckpt" file containing the trained model, 
# metrics.csv file containing losses for each training and validation epoch and "hparams.yaml", needed for running inference.
def MODEL_PATH(dir_name : str) -> Path:
    return MODELS_PATH / dir_name

# This directory contains mesh file (and texture). Blenderproc currently supports obj, ply, dae, stl, fbx and glb.
# The texture file is present only for ".ply" and has same name as the mash.
# Mesh name formate is: obj_{obj_id:06d}.*
def MESH_PATH(dataset_name : str) -> Path:
    return DATASET_PATH(dataset_name) / "meshes"

# This file contains string labels assigned to their numeric value for given dataset.
def LABELS_PATH(dataset_name : str) -> Path:
    return DATASET_PATH(dataset_name) / "labels.txt"
